"""
app.py — Transit Hunter
A web app that pulls a real NASA light curve for any star, searches it for
exoplanet transits, and vets each finding against NASA's catalogs.

Pipeline: fetch (lightkurve) → detect (detection.py) → cross-check (catalog.py).

Run locally:  streamlit run app.py
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from detection import find_transit, fold
import catalog

st.set_page_config(page_title="Transit Hunter", page_icon="🪐", layout="centered")

# --- Session state --------------------------------------------------------
if "log" not in st.session_state:
    st.session_state.log = []      # saved detections
if "last" not in st.session_state:
    st.session_state.last = None   # most recent analysis, so results survive reruns

st.title("🪐 Transit Hunter")
st.markdown(
    "Search a real NASA light curve for exoplanet transits — then vet the finding "
    "against NASA's catalogs. Enter a star, and this tool fetches its brightness from the "
    "[MAST archive](https://archive.stsci.edu), runs Box Least Squares detection, checks "
    "whether the signal is already a known planet, and tells you whether it's worth a closer look."
)

with st.expander("🔰 New here? Start with this 30-second explainer", expanded=False):
    st.markdown(
        "**How do we find a planet we can't even see?** We watch its star's brightness.\n\n"
        "When a planet passes *in front* of its star (from our point of view), it blocks a "
        "tiny sliver of light — so the star looks very slightly dimmer for a few hours. Then "
        "the planet moves on and the brightness returns. Because the planet orbits over and "
        "over, this dimming **repeats on a regular schedule**.\n\n"
        "So we look for a dip in brightness that comes back again and again, evenly spaced. "
        "That repeating dip is called a **transit**, and it's the fingerprint of a planet.\n\n"
        "This app hunts for that dip, then asks NASA whether the star is already a known "
        "planet host — so you know if you've re-found something known, or spotted something "
        "worth escalating. Every technical word has a tooltip (hover the **?**) or is in the "
        "**Glossary** at the bottom."
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
    mission = st.selectbox(
        "Mission", ["Auto", "TESS", "Kepler", "K2"], index=0,
        help="Which NASA space telescope's data to use. Kepler and TESS both hunt planets "
             "the same way. 'Auto' just uses whatever data exists for your star.",
    )

st.caption("Examples: **Pi Mensae** (TESS, known planet) · **Kepler-10** (Kepler) · **TrES-2** (Kepler)")

c1, c2 = st.columns(2)
with c1:
    min_period = st.slider(
        "Shortest orbit to look for (days)", 0.5, 10.0, 0.5, 0.5,
        help="The shortest 'year' to search for. 0.5 means: also consider planets that "
             "whip around their star in half a day. Leave the defaults unless you're curious.",
    )
with c2:
    max_period = st.slider(
        "Longest orbit to look for (days)", 5.0, 30.0, 20.0, 1.0,
        help="The longest 'year' to search for. We can only spot orbits shorter than the "
             "observing window, so very long periods aren't reliably detectable here.",
    )

go = st.button("Find transits", type="primary")


# --- Data fetch (cached so repeat searches are instant) -------------------
def _coords_from_lc(lc):
    """Best-effort star coordinates (RA/Dec in degrees) from light-curve metadata."""
    meta = getattr(lc, "meta", {}) or {}
    for rk, dk in (("RA_OBJ", "DEC_OBJ"), ("RA", "DEC"), ("ra", "dec")):
        ra, dec = meta.get(rk), meta.get(dk)
        if ra is not None and dec is not None:
            try:
                return float(ra), float(dec)
            except (TypeError, ValueError):
                pass
    try:
        ra_a, dec_a = getattr(lc, "ra", None), getattr(lc, "dec", None)
        if ra_a is not None and dec_a is not None:
            return float(getattr(ra_a, "value", ra_a)), float(getattr(dec_a, "value", dec_a))
    except (TypeError, ValueError):
        pass
    return None, None


@st.cache_data(show_spinner=False)
def fetch_light_curve(target: str, mission: str):
    """Download and clean a light curve. Returns (time, flux, label, ra, dec)."""
    import lightkurve as lk

    kwargs = {} if mission == "Auto" else {"mission": mission}
    search = lk.search_lightcurve(target, **kwargs)
    if len(search) == 0:
        raise LookupError(f"No light curve found for '{target}'.")

    collection = search[: min(len(search), 5)].download_all()
    lc = collection.stitch() if collection is not None else search[0].download()

    ra, dec = _coords_from_lc(lc)
    lc = lc.remove_nans().normalize().flatten(window_length=401)
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    label = f"{target} · {getattr(lc, 'mission', mission)}"
    return time, flux, label, ra, dec


# --- Run analysis on button press, then stash results so they survive reruns ---
if go:
    try:
        with st.spinner(f"Fetching light curve for {target}…"):
            time, flux, label, ra, dec = fetch_light_curve(target.strip(), mission)
    except Exception as e:
        st.error(f"Couldn't load data: {e}")
        st.stop()

    with st.spinner("Searching for transits…"):
        try:
            res = find_transit(time, flux, min_period=min_period, max_period=max_period)
        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.stop()

    with st.spinner("Cross-checking against NASA's catalogs…"):
        cat = catalog.crosscheck(ra, dec, res.period, res.snr)

    st.session_state.last = {
        "target": target.strip(), "label": label, "ra": ra, "dec": dec,
        "time": time, "flux": flux, "res": res, "cat": cat,
    }


# --- Render results from the stored analysis (not gated on `go`) ----------
last = st.session_state.last
if last:
    res, cat = last["res"], last["cat"]
    time, flux, label = last["time"], last["flux"], last["label"]

    # Detection verdict
    if res.snr >= 7:
        st.success(res.verdict)
    elif res.snr >= 3:
        st.info(res.verdict)
    else:
        st.warning(res.verdict)

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Orbital period", f"{res.period:.3f} d",
        help="How long the planet takes to go around its star once — its 'year'. "
             "3 days means a transit repeats every 3 days.",
    )
    m2.metric(
        "Transit depth", f"{res.depth_ppm:,.0f} ppm",
        help="How much the star dims during a transit, in parts-per-million (1,000 ppm = "
             "0.1% dimmer). A bigger planet blocks more light, so a deeper dip usually means "
             "a bigger planet.",
    )
    m3.metric(
        "Signal / noise", f"{res.snr:.1f}",
        help="How strongly the dip stands out from random jitter. Rough guide: below 3 = "
             "probably noise, 3–7 = worth a look, above 7 = a strong signal.",
    )

    # --- Plots ------------------------------------------------------------
    st.subheader("Full light curve")
    st.caption(
        "**What this is:** the star's brightness (up-down) over time (left-right), straight "
        "from NASA. **What to look for:** a mostly-flat line — that's the steady star — with "
        "occasional tiny drops where a planet may have crossed in front. The dips are usually "
        "too small and too spread out to spot by eye here; that's what the next plot is for."
    )
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(time, flux, ".", ms=1, alpha=0.5, color="#b4531f")
    ax1.set_xlabel("Time (days)  →  each dot is one brightness measurement")
    ax1.set_ylabel("Brightness (1.0 = normal)")
    ax1.set_title(label)
    st.pyplot(fig1)

    st.subheader("Folded on best period")
    st.caption(
        "**What this is:** we took the detected 'year' length and stacked every orbit on top "
        "of each other, so all the repeated dips line up in one place. **What to look for:** a "
        "clear U- or V-shaped dip in the middle (at phase 0). A clean dip here is the "
        "fingerprint of a real, repeating transit. A flat, shapeless cloud means no convincing "
        "planet. The orange line is the averaged trend to help you see the shape."
    )
    phase, fflux = fold(time, flux, res.period, res.t0)
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(phase, fflux, ".", ms=1, alpha=0.4, color="#555")
    bins = np.linspace(-0.5, 0.5, 80)
    which = np.digitize(phase, bins)
    bx, by = [], []
    for b in range(1, len(bins)):
        sel = which == b
        if sel.sum() > 0:
            bx.append(phase[sel].mean())
            by.append(fflux[sel].mean())
    ax2.plot(bx, by, "-", color="#b4531f", lw=1.5)
    ax2.set_xlabel("Phase  (0 = the moment of transit; ±0.5 = halfway around the orbit)")
    ax2.set_ylabel("Brightness (1.0 = normal)")
    ax2.set_xlim(-0.5, 0.5)
    st.pyplot(fig2)

    # --- Vetting: is this already known? ----------------------------------
    st.subheader("Is this already known? (catalog cross-check)")
    if cat.status in (catalog.STATUS_UNCATALOGUED, catalog.STATUS_KNOWN_DIFF) and res.snr >= 7:
        st.success(f"**{cat.headline}**")
    elif cat.status == catalog.STATUS_KNOWN_MATCH:
        st.info(f"**{cat.headline}**")
    else:
        st.warning(f"**{cat.headline}**")
    st.write(cat.detail)

    if cat.known:
        rows = [{"Object": k.name,
                 "Type": k.kind,
                 "Known period (d)": f"{k.period:.3f}" if k.period else "—"}
                for k in cat.known]
        st.caption("Known planets / candidates already catalogued at this star:")
        st.table(pd.DataFrame(rows))

    # --- Escalation guidance for genuinely interesting leads --------------
    if cat.recommend_save and res.snr >= 7 and cat.status in (
        catalog.STATUS_UNCATALOGUED, catalog.STATUS_KNOWN_DIFF):
        with st.expander("🚀 This looks worth escalating — what next? (read before getting excited)"):
            st.markdown(
                "A strong, uncatalogued signal is a *lead*, not a discovery. The overwhelming "
                "majority turn out to be eclipsing binaries, blends, or instrument artifacts. "
                "The honest next steps:\n\n"
                "1. **Sanity-check it yourself** — try a different sector/mission, widen the "
                "period range, and see if the dip survives.\n"
                "2. **Take it to people who vet for a living** — "
                "[Planet Hunters TESS](https://www.zooniverse.org/projects/nora-dot-eisner/planet-hunters-tess) "
                "is a citizen-science community that inspects exactly these signals and has "
                "found ~150 candidates the algorithms missed.\n"
                "3. **Real follow-up lives at "
                "[ExoFOP-TESS](https://exofop.ipac.caltech.edu)** — NASA's repository for "
                "follow-up observations of candidates. Viewing is open; uploading needs "
                "registration and genuine new observations (e.g. your own telescope data), "
                "not archival re-detections.\n\n"
                "This tool does the fast first pass. A human takes it the last mile."
            )

    # --- Save + log -------------------------------------------------------
    st.subheader("Save this detection")
    rec = "✅ Our take: **worth saving.**" if cat.recommend_save else \
          "➖ Our take: **not worth saving** (already known or too weak) — but your call."
    st.markdown(rec)

    if st.button("💾 Save to my log"):
        st.session_state.log.append({
            "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "target": last["target"],
            "ra": last["ra"], "dec": last["dec"],
            "period_d": round(res.period, 4),
            "depth_ppm": round(res.depth_ppm, 1),
            "snr": round(res.snr, 2),
            "catalog_status": cat.status,
            "verdict": cat.headline,
            "recommended": "save" if cat.recommend_save else "ignore",
        })
        st.success("Saved. Scroll to **My detection log** below.")

    with st.expander("📖 Glossary — every technical word, in plain English"):
        st.markdown(
            "**Light curve** — a graph of a star's brightness over time.\n\n"
            "**Transit** — when a planet passes in front of its star from our viewpoint, "
            "dimming it slightly. The event we're hunting for.\n\n"
            "**Orbital period** — how long a planet takes to circle its star once. Its 'year'.\n\n"
            "**Transit depth** — how much the star dims during a transit. Bigger planet → "
            "deeper dip. Shown in **ppm** (parts-per-million): 10,000 ppm = 1%.\n\n"
            "**Phase** — position within one orbit, from -0.5 to +0.5. 'Phase 0' is the middle "
            "of the transit.\n\n"
            "**Folding** — wrapping the light curve onto one orbit's length so all the repeated "
            "dips pile up in the same spot.\n\n"
            "**Signal-to-noise (SNR)** — how much the dip stands out above random jitter. "
            "Higher = more trustworthy.\n\n"
            "**Box Least Squares (BLS)** — the detection method. It slides a box-shaped dip "
            "across thousands of possible orbit lengths to find the best repeating transit.\n\n"
            "**TOI** — TESS Object of Interest: a candidate flagged by NASA's pipeline, not yet "
            "confirmed as a planet.\n\n"
            "**Confirmed planet** — a candidate that follow-up observations have verified.\n\n"
            "**Catalog cross-check** — asking NASA's Exoplanet Archive whether a known planet or "
            "candidate already sits at this star's position.\n\n"
            "**MAST archive** — NASA's public library of space-telescope data.\n\n"
            "**TIC / KIC ID** — catalogue ID numbers for stars (TESS / Kepler Input Catalog)."
        )

# --- My detection log (persists across searches this session) -------------
if st.session_state.log:
    st.divider()
    st.subheader(f"📋 My detection log ({len(st.session_state.log)})")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df, use_container_width=True)
    dcol1, dcol2 = st.columns([1, 1])
    with dcol1:
        st.download_button(
            "⬇️ Download log (CSV)", df.to_csv(index=False),
            file_name="my_transit_detections.csv", mime="text/csv",
        )
    with dcol2:
        if st.button("🗑️ Clear log"):
            st.session_state.log = []
            st.rerun()

st.divider()
st.caption(
    "Built by Sree · data from NASA MAST via lightkurve · detection by Box Least Squares · "
    "catalog cross-check via the NASA Exoplanet Archive. A vetting and learning tool — "
    "not an automated discovery pipeline. Next: a community repository and a scheduled watchlist."
)
