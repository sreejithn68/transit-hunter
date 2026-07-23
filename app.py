"""
app.py — Transit Hunter
A web app that pulls a real NASA light curve for any star and searches it for
exoplanet transits. Built with Streamlit + lightkurve; the detection maths lives
in detection.py.

Run locally:  streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from detection import find_transit, fold

st.set_page_config(page_title="Transit Hunter", page_icon="🪐", layout="centered")

st.title("🪐 Transit Hunter")
st.markdown(
    "Search a real NASA light curve for exoplanet transits. Enter a star, and this "
    "tool fetches its brightness measurements from the [MAST archive](https://archive.stsci.edu), "
    "runs Box Least Squares detection, and tells you whether there's a candidate planet."
)

# --- Inputs ---------------------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    target = st.text_input(
        "Star name or ID",
        value="Pi Mensae",
        help="Try a name (Pi Mensae, Kepler-10) or an ID (TIC 261136679, KIC 11904151).",
    )
with col2:
    mission = st.selectbox("Mission", ["Auto", "TESS", "Kepler", "K2"], index=0)

st.caption("Examples: **Pi Mensae** (TESS, known planet) · **Kepler-10** (Kepler) · **TrES-2** (Kepler)")

c1, c2 = st.columns(2)
with c1:
    min_period = st.slider("Min period (days)", 0.5, 10.0, 0.5, 0.5)
with c2:
    max_period = st.slider("Max period (days)", 5.0, 30.0, 20.0, 1.0)

go = st.button("Find transits", type="primary")


# --- Data fetch (cached so repeat searches are instant) -------------------
@st.cache_data(show_spinner=False)
def fetch_light_curve(target: str, mission: str):
    """Download and clean a light curve. Returns (time, flux, label)."""
    import lightkurve as lk

    kwargs = {} if mission == "Auto" else {"mission": mission}
    search = lk.search_lightcurve(target, **kwargs)
    if len(search) == 0:
        raise LookupError(f"No light curve found for '{target}'.")

    # Take up to a few sectors/quarters and stitch them into one series.
    collection = search[: min(len(search), 5)].download_all()
    lc = collection.stitch() if collection is not None else search[0].download()

    # Clean: drop bad points, normalise to ~1.0, flatten out slow stellar wiggle.
    lc = lc.remove_nans().normalize().flatten(window_length=401)
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    label = f"{target} · {getattr(lc, 'mission', mission)}"
    return time, flux, label


if go:
    try:
        with st.spinner(f"Fetching light curve for {target}…"):
            time, flux, label = fetch_light_curve(target.strip(), mission)
    except Exception as e:
        st.error(f"Couldn't load data: {e}")
        st.stop()

    with st.spinner("Searching for transits…"):
        try:
            res = find_transit(time, flux, min_period=min_period, max_period=max_period)
        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.stop()

    # --- Verdict ----------------------------------------------------------
    if res.snr >= 7:
        st.success(res.verdict)
    elif res.snr >= 3:
        st.info(res.verdict)
    else:
        st.warning(res.verdict)

    m1, m2, m3 = st.columns(3)
    m1.metric("Orbital period", f"{res.period:.3f} d")
    m2.metric("Transit depth", f"{res.depth_ppm:,.0f} ppm")
    m3.metric("Signal / noise", f"{res.snr:.1f}")

    # --- Plots ------------------------------------------------------------
    st.subheader("Full light curve")
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(time, flux, ".", ms=1, alpha=0.5, color="#b4531f")
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Normalised brightness")
    ax1.set_title(label)
    st.pyplot(fig1)

    st.subheader("Folded on best period")
    st.caption(
        "If the signal is real, folding the data on the detected period stacks "
        "every transit into one dip at phase 0."
    )
    phase, fflux = fold(time, flux, res.period, res.t0)
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(phase, fflux, ".", ms=1, alpha=0.4, color="#555")
    # binned average for a clean overlay
    bins = np.linspace(-0.5, 0.5, 80)
    which = np.digitize(phase, bins)
    bx, by = [], []
    for b in range(1, len(bins)):
        sel = which == b
        if sel.sum() > 0:
            bx.append(phase[sel].mean())
            by.append(fflux[sel].mean())
    ax2.plot(bx, by, "-", color="#b4531f", lw=1.5)
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Normalised brightness")
    ax2.set_xlim(-0.5, 0.5)
    st.pyplot(fig2)

    with st.expander("What am I looking at?"):
        st.markdown(
            "- **Full light curve** — the star's brightness over time, straight from NASA.\n"
            "- **Folded plot** — the data wrapped on the best-fit period. A clean dip at "
            "phase 0 is the fingerprint of a transiting planet.\n"
            "- **Depth** tells you the planet's size relative to the star; **period** is its "
            "year length; **SNR** is how much we trust the signal over noise.\n\n"
            "This is a first-pass vetting tool — a strong signal here is a *candidate*, not "
            "a confirmed planet. Confirming one takes follow-up observations."
        )

st.divider()
st.caption(
    "Built by Sree · data from NASA MAST via lightkurve · detection by Box Least Squares. "
    "Part of an ongoing project — next up: an ML classifier and a live daily pipeline."
)
