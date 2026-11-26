"""
app.py — Transit Hunter
A web app that pulls real NASA light curves for any star, searches them for
exoplanet transits (several planets per star), vets each signal for the usual
false positives, and cross-checks it against NASA's catalogs.

Pipeline: fetch (lightcurves.py) → detect (detection.py) → vet (vetting.py)
          → cross-check (catalog.py).

Run locally:  streamlit run app.py
"""

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import catalog
import detection
import lightcurves
import vetting

st.set_page_config(page_title="Transit Hunter", page_icon="🪐", layout="centered")

SIGNAL_COLORS = ["#b4531f", "#2a6fb0", "#3a8f4b", "#8e44ad", "#c0392b"]
STATUS_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", "n/a": "➖"}

# --- Session state --------------------------------------------------------
if "log" not in st.session_state:
    st.session_state.log = []
if "last" not in st.session_state:
    st.session_state.last = None

st.title("🪐 Transit Hunter")
st.markdown(
    "Search real NASA light curves for exoplanet transits — then vet every signal "
    "for the usual impostors and check it against NASA's catalogs. Enter a star: this "
    "tool fetches its brightness from the [MAST archive](https://archive.stsci.edu), hunts "
    "for repeating dips (several planets if they're there), runs false-positive checks, "
    "and tells you whether anything is worth a closer look."
)

with st.expander("🔰 New here? Start with this 30-second explainer", expanded=False):
    st.markdown(
        "**How do we find a planet we can't even see?** We watch its star's brightness.\n\n"
        "When a planet passes *in front* of its star (from our point of view), it blocks a "
        "tiny sliver of light — so the star looks very slightly dimmer for a few hours. "
        "Because the planet orbits over and over, this dimming **repeats on a regular "
        "schedule**. That repeating dip is a **transit**.\n\n"
        "But many things fake a transit — most often two stars eclipsing each other. So "
        "after finding a dip, this app runs the same kind of sanity checks professional "
        "astronomers use, and asks NASA whether the signal is already known. Every "
        "technical word is in the **Glossary** at the bottom."
    )

# --- Inputs ---------------------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    target = st.text_input(
        "Star name or ID", value="Pi Mensae",
        help="A name (Pi Mensae, Kepler-10, TOI-700) or an ID (TIC 261136679, KIC 11904151).")
with col2:
    mission = st.selectbox(
        "Mission", ["Auto", "TESS", "Kepler", "K2"], index=0,
        help="Which NASA space telescope's data to use. 'Auto' prefers TESS, then Kepler, then K2.")

st.caption("Try: **Pi Mensae** (one planet) · **TOI-700** (several planets, incl. an "
           "Earth-sized one) · **Kepler-10** (a 20-hour 'lava world' and a 45-day planet)")

n_segments = st.slider(
    "How much data to use (TESS sectors / Kepler quarters)", 1, 20, 4,
    help="More data = more transits stacked = smaller planets and longer orbits become "
         "findable. It also takes longer to download and search.")

with st.expander("⚙️ Advanced options (most people can skip this)"):
    c1, c2 = st.columns(2)
    with c1:
        min_period = st.number_input("Shortest orbit to look for (days)", 0.3, 20.0, 0.5, 0.1)
    with c2:
        max_period = st.number_input(
            "Longest orbit to look for (days, 0 = as long as the data allows)", 0.0, 500.0, 0.0, 1.0)
    c3, c4 = st.columns(2)
    with c3:
        tls_ok = detection.tls_available()
        method_label = st.radio(
            "Search method",
            ["Fast (Box Least Squares)", "Deep (Transit Least Squares)"],
            help="Deep search fits a realistic transit shape instead of a box. It is a bit "
                 "more sensitive to small planets, and slower." +
                 ("" if tls_ok else " (Unavailable here: package not installed.)"))
    with c4:
        max_planets = st.slider("Max planets to look for", 1, 5, 3)
    st.caption("The period range is automatically capped so that at least two transits "
               "fit inside the data.")

method = "tls" if method_label.startswith("Deep") else "bls"
go = st.button("Find transits", type="primary")


# --- Cached steps ---------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=20)
def cached_fetch(target: str, mission: str, n_segments: int):
    return lightcurves.fetch(target, mission, max_segments=n_segments)


@st.cache_data(show_spinner=False, max_entries=20)
def cached_search(time, flux, min_period, max_period, method, max_planets):
    return detection.search(time, flux, min_period=min_period,
                            max_period=max_period or None, method=method,
                            max_planets=max_planets)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def cached_known(ra, dec):
    return catalog.query_known(ra, dec)


# --- Run analysis ---------------------------------------------------------
if go:
    try:
        with st.spinner(f"Fetching light curves for {target}… (first time can take a minute)"):
            data = cached_fetch(target.strip(), mission, n_segments)
    except Exception as e:
        st.error(f"Couldn't load data: {e}")
        st.stop()

    with st.spinner("Searching for transits…"):
        try:
            out = cached_search(data.time, data.flux, float(min_period), float(max_period),
                                method, max_planets)
        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.stop()

    reports = [vetting.vet(*out.isolated(i), s, data.star, others=out.signals)
               for i, s in enumerate(out.signals)]

    with st.spinner("Cross-checking against NASA's catalogs…"):
        if data.ra is None or data.dec is None:
            verdicts = [catalog.no_position_verdict() for _ in out.signals]
        else:
            try:
                known = cached_known(data.ra, data.dec)
                verdicts = [catalog.judge(known, s.period, s.strength, r.verdict)
                            for s, r in zip(out.signals, reports)]
            except catalog.ArchiveError as e:
                verdicts = [catalog.archive_down_verdict(e) for _ in out.signals]

    st.session_state.last = {"target": target.strip(), "data": data, "out": out,
                             "reports": reports, "verdicts": verdicts}


# --- Plot helpers ---------------------------------------------------------
def binned(x, y, nbins=80, lo=-0.5, hi=0.5):
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.digitize(x, edges)
    bx, by = [], []
    for b in range(1, len(edges)):
        sel = idx == b
        if sel.sum() > 2:
            bx.append(x[sel].mean())
            by.append(np.median(y[sel]))
    return np.array(bx), np.array(by)


def zoom_for(s):
    """Phase half-width that shows the transit nicely (≈ 3 durations)."""
    return float(np.clip(1.5 * s.duration / s.period, 0.02, 0.5))


# --- Render results -------------------------------------------------------
last = st.session_state.last
if last:
    data, out = last["data"], last["out"]
    reports, verdicts = last["reports"], last["verdicts"]
    dets = out.detections

    if not dets:
        st.warning(out.signals[0].verdict)
    else:
        strong = [s for s in dets if s.strength == "strong"]
        st.success(f"Found **{len(dets)}** transit-like signal{'s' if len(dets) > 1 else ''} "
                   f"({len(strong)} strong) in {data.label} — "
                   f"{data.baseline_days:.0f} days of data.")
    if data.n_available > data.n_segments:
        st.caption(f"Used the {data.n_segments} most recent of {data.n_available} available "
                   f"segments. Raise the data slider to search them all.")

    # Full light curve with transits marked
    st.subheader("Full light curve")
    st.caption(
        "**What this is:** the star's brightness over time, cleaned and flattened (slow "
        "stellar wiggles removed). **What to look for:** a flat band with occasional tiny "
        "drops. Coloured ticks mark where each detected signal's transits fall.")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(out.time, out.flux, ".", ms=1, alpha=0.5, color="#666")
    lo = np.percentile(out.flux, 0.5)
    for k, s in enumerate(dets):
        n0 = np.ceil((out.time.min() - s.t0) / s.period)
        n1 = np.floor((out.time.max() - s.t0) / s.period)
        times = s.t0 + np.arange(n0, n1 + 1) * s.period
        ax.plot(times, np.full(times.size, lo - 0.3 * (1 - lo)), "^",
                color=SIGNAL_COLORS[k % 5], ms=5, label=f"signal {k + 1}: {s.period:.3f} d")
    if dets:
        ax.legend(loc="lower right", fontsize=7)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Brightness (1.0 = normal)")
    ax.set_title(data.label, fontsize=9)
    st.pyplot(fig)

    # Periodogram
    s0 = out.signals[0]
    if s0.periods is not None:
        st.subheader("Periodogram — which orbit length fits best?")
        st.caption(
            "**What this is:** for every trial orbit length, how well a repeating dip "
            "explains the data. **What to look for:** a tall, isolated spike. Smaller "
            "spikes at ½× or 2× the period are normal echoes of the same signal.")
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.plot(s0.periods, s0.power, lw=0.6, color="#444")
        for k, s in enumerate(dets):
            ax.axvline(s.period, color=SIGNAL_COLORS[k % 5], alpha=0.6, lw=1.5, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Trial orbital period (days)")
        ax.set_ylabel("Signal strength (SDE)")
        st.pyplot(fig)

    # One tab per signal
    shown = dets or out.signals[:1]
    tabs = st.tabs([f"Signal {k + 1} · {s.period:.3f} d" for k, s in enumerate(shown)])
    for k, (tab, s) in enumerate(zip(tabs, shown)):
        rep, cat = reports[k], verdicts[k]
        color = SIGNAL_COLORS[k % 5]
        with tab:
            box = st.success if s.strength == "strong" else st.info if s.strength == "plausible" else st.warning
            box(s.verdict)

            m1, m2, m3 = st.columns(3)
            m1.metric("Orbital period", f"{s.period:.4f} d",
                      help="How long one orbit takes — the planet's 'year'.")
            m2.metric("Transit depth", f"{s.depth_ppm:,.0f} ppm",
                      help="How much the star dims (1,000 ppm = 0.1%). Bigger planet → deeper dip.")
            m3.metric("Duration", f"{s.duration * 24:.1f} h",
                      help="How long each transit lasts.")
            m4, m5, m6 = st.columns(3)
            m4.metric("SNR", f"{s.snr:.1f}",
                      help="Depth divided by its uncertainty, counting every in-transit point "
                           "and correcting for correlated noise. ≥ 7.1 plausible, ≥ 8 strong.")
            m5.metric("SDE", f"{s.sde:.1f}",
                      help="How far this period's spike stands above the rest of the "
                           "periodogram. ≥ 9 means the period clearly wins.")
            m6.metric("Transits seen", f"{s.n_transits}",
                      help="Individual transits with data. Fewer than 3 = period less certain.")
            if rep.radius_earth:
                st.caption(f"Implied size ≈ **{rep.radius_earth:.1f} Earth radii** "
                           f"(star radius {data.star.radius:.2f} R☉ from the TESS/Kepler input catalog).")

            # Folded plot
            st.markdown("**Folded on this period**")
            st.caption("Every orbit stacked on top of each other, zoomed on the transit. A "
                       "clean, flat-bottomed dip at 0 is what a planet looks like.")
            ti, fi = out.isolated(k)
            phase, ff = detection.fold(ti, fi, s.period, s.t0)
            z = zoom_for(s)
            sel = np.abs(phase) < z
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(phase[sel] * s.period * 24, ff[sel], ".", ms=2, alpha=0.4, color="#777")
            bx, by = binned(phase[sel], ff[sel], 60, -z, z)
            ax.plot(bx * s.period * 24, by, "-", color=color, lw=2)
            ax.set_xlabel("Hours from mid-transit")
            ax.set_ylabel("Brightness")
            st.pyplot(fig)

            # Odd / even
            st.markdown("**Odd vs. even transits**")
            st.caption("If alternate dips differ in depth, it's probably two stars of different "
                       "sizes eclipsing each other — not a planet.")
            epoch = np.round((ti - s.t0) / s.period).astype(int)
            fig, axes = plt.subplots(1, 2, figsize=(8, 2.6), sharey=True)
            for ax, parity, name in ((axes[0], 1, "Odd"), (axes[1], 0, "Even")):
                m = (epoch % 2) == parity
                ph, fl = detection.fold(ti[m], fi[m], s.period, s.t0)
                sel = np.abs(ph) < z
                ax.plot(ph[sel] * s.period * 24, fl[sel], ".", ms=2, alpha=0.3, color="#777")
                bx, by = binned(ph[sel], fl[sel], 40, -z, z)
                ax.plot(bx * s.period * 24, by, "-", color=color, lw=2)
                ax.axhline(1 - s.depth, color="k", lw=0.6, ls=":")
                ax.set_title(name, fontsize=9)
                ax.set_xlabel("Hours")
            st.pyplot(fig)

            # Vetting checklist
            st.markdown(f"**False-positive checks → {rep.verdict.upper()}**")
            st.table(pd.DataFrame([{"": STATUS_ICON[c.status], "Check": c.name,
                                    "Value": c.value, "What it means": c.explanation}
                                   for c in rep.checks]))

            # Catalog
            st.markdown("**Is this already known?**")
            if cat.status == catalog.STATUS_KNOWN_MATCH:
                st.info(f"**{cat.headline}**")
            elif cat.recommend_save:
                st.success(f"**{cat.headline}**")
            else:
                st.warning(f"**{cat.headline}**")
            st.write(cat.detail)
            if cat.known:
                st.caption("Known planets / candidates already catalogued at this star:")
                st.table(pd.DataFrame([{"Object": o.name, "Type": o.kind,
                                        "Known period (d)": f"{o.period:.3f}" if o.period else "—"}
                                       for o in cat.known]))

            if cat.recommend_save:
                with st.expander("🚀 This looks worth escalating — what next? (read before getting excited)"):
                    st.markdown(
                        "A strong, uncatalogued, planet-like signal is a *lead*, not a discovery. "
                        "The honest next steps:\n\n"
                        "1. **Re-check it yourself** — try more sectors, the deep search, and see "
                        "whether the dip survives.\n"
                        "2. **Check it isn't known anywhere** — search the star's TIC ID on "
                        "[ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/) (TOIs *and* "
                        "community TOIs), [exoplanet.eu](https://exoplanet.eu) and "
                        "[SIMBAD](https://simbad.u-strasbg.fr).\n"
                        "3. **Take it to people who vet for a living** — "
                        "[Planet Hunters TESS](https://www.zooniverse.org/projects/nora-dot-eisner/planet-hunters-tess) "
                        "or submit a community TOI on ExoFOP (needs a free account).\n\n"
                        "This tool does the fast first pass. A human takes it the last mile.")

            rec = ("✅ Our take: **worth saving.**" if cat.recommend_save else
                   "➖ Our take: **not worth saving** (known, weak, or failed vetting) — your call.")
            st.markdown(rec)
            if st.button("💾 Save to my log", key=f"save_{k}"):
                st.session_state.log.append({
                    "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "target": last["target"], "tic": data.tic,
                    "ra": data.ra, "dec": data.dec, "signal": k + 1,
                    "period_d": round(s.period, 5), "t0": round(s.t0, 5),
                    "duration_h": round(s.duration * 24, 2),
                    "depth_ppm": round(s.depth_ppm, 1), "snr": round(s.snr, 2),
                    "sde": round(s.sde, 2), "n_transits": s.n_transits,
                    "radius_earth": round(rep.radius_earth, 2) if rep.radius_earth else None,
                    "method": s.method, "strength": s.strength, "vetting": rep.verdict,
                    "catalog_status": cat.status,
                    "recommended": "save" if cat.recommend_save else "ignore",
                })
                st.success("Saved. Scroll to **My detection log** below.")

    with st.expander("📖 Glossary — every technical word, in plain English"):
        st.markdown(
            "**Light curve** — a graph of a star's brightness over time.\n\n"
            "**Transit** — a planet passing in front of its star, dimming it slightly.\n\n"
            "**Orbital period** — how long one orbit takes. The planet's 'year'.\n\n"
            "**Transit depth** — how much the star dims, in **ppm** (parts-per-million; "
            "10,000 ppm = 1%).\n\n"
            "**Phase / folding** — wrapping the light curve onto one orbit so all the dips "
            "pile up in the same place.\n\n"
            "**SNR (signal-to-noise)** — how many times bigger the dip is than its "
            "uncertainty. Corrected for *correlated noise* (slow wiggles that random-noise "
            "maths would underestimate).\n\n"
            "**SDE (signal detection efficiency)** — how far the best period's spike in the "
            "periodogram stands above all the others.\n\n"
            "**Periodogram** — signal strength vs. trial orbit length.\n\n"
            "**BLS / TLS** — Box / Transit Least Squares: the search methods. BLS slides a box-"
            "shaped dip; TLS a realistic transit shape.\n\n"
            "**Eclipsing binary** — two stars orbiting and eclipsing each other: the most "
            "common planet impostor.\n\n"
            "**Odd/even test** — compares alternate dips; unequal depths betray a binary.\n\n"
            "**Secondary eclipse** — a second dip half an orbit later, when the companion "
            "passes *behind*. Big ones mean the companion glows — a star, not a planet.\n\n"
            "**V-shaped dip** — no flat bottom; typical of grazing binaries.\n\n"
            "**Contamination** — light from neighbouring stars falling in the same pixels.\n\n"
            "**TOI / CTOI** — TESS Object of Interest (flagged by NASA) / Community TOI "
            "(flagged by the public or other teams).\n\n"
            "**TIC / KIC ID** — catalogue numbers for stars (TESS / Kepler Input Catalog).")

# --- My detection log -----------------------------------------------------
if st.session_state.log:
    st.divider()
    st.subheader(f"📋 My detection log ({len(st.session_state.log)})")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df, use_container_width=True)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇️ Download log (CSV)", df.to_csv(index=False),
                           file_name="my_transit_detections.csv", mime="text/csv")
    with d2:
        if st.button("🗑️ Clear log"):
            st.session_state.log = []
            st.rerun()

st.divider()
st.caption(
    "Built by Sree · data from NASA MAST via lightkurve · detection by Box / Transit Least "
    "Squares · false-positive vetting · catalog cross-check via the NASA Exoplanet Archive. "
    "A vetting and learning tool — not an automated discovery pipeline.")
