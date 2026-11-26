"""Sector Feed — candidates found by the automated TESS sector pipeline."""

import glob
import json
import os
import urllib.parse

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sector Feed · Transit Hunter", page_icon="📡", layout="wide")
st.title("📡 Sector Feed")
st.markdown(
    "Every week a robot searches each new TESS sector's ~10,000 two-minute stars, keeps only "
    "**strong** signals that **pass every vetting check**, and throws away anything already "
    "listed as a TOI, community TOI, NASA pipeline TCE, confirmed planet or exoplanet.eu entry. "
    "What's left is below. These are **leads, not discoveries** — open one to inspect it.")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dirs = sorted(glob.glob(os.path.join(ROOT, "results", "s*")),
              key=lambda d: int(os.path.basename(d)[1:]) if os.path.basename(d)[1:].isdigit() else 0,
              reverse=True)
if not dirs:
    st.info("No sectors searched yet. The pipeline runs every Monday (or on demand from GitHub Actions).")
    st.stop()

for d in dirs:
    sector = os.path.basename(d)[1:]
    summ_path, cand_path = os.path.join(d, "summary.json"), os.path.join(d, "candidates.csv")
    if not os.path.exists(summ_path):
        continue
    s = json.load(open(summ_path))
    with st.expander(f"Sector {sector} — {s['clean_candidates']} clean candidate(s), "
                     f"{s['stars_searched']:,} stars searched", expanded=(d == dirs[0])):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stars searched", f"{s['stars_searched']:,}")
        c2.metric("Strong + planet-like", s["strong_planet_like"])
        c3.metric("Not in any catalog", s["not_in_any_catalog"])
        c4.metric("Clean candidates", s["clean_candidates"])
        if s.get("catalogs_unavailable"):
            st.warning("Some catalogs couldn't be checked: " + ", ".join(s["catalogs_unavailable"]))
        if not os.path.exists(cand_path) or os.path.getsize(cand_path) < 5:
            st.caption("Nothing new this sector.")
            continue
        c = pd.read_csv(cand_path)
        if c.empty:
            st.caption("Nothing new this sector.")
            continue
        c["Analyse"] = c.tic.map(lambda t: "/?" + urllib.parse.urlencode({"target": f"TIC {int(t)}"}))
        show = c[["tic", "period", "depth_ppm", "radius_earth", "snr", "n_transits", "novelty",
                  "binary_flags", "unchecked", "exofop", "Analyse"]]
        st.dataframe(
            show, width="stretch", hide_index=True,
            column_config={
                "tic": st.column_config.NumberColumn("TIC", format="%d"),
                "period": st.column_config.NumberColumn("Period (d)", format="%.4f"),
                "depth_ppm": st.column_config.NumberColumn("Depth (ppm)", format="%.0f"),
                "radius_earth": st.column_config.NumberColumn("Size (R⊕)", format="%.1f"),
                "snr": st.column_config.NumberColumn("SNR", format="%.1f"),
                "n_transits": "Transits",
                "novelty": "Novelty",
                "binary_flags": "Binary / variable evidence",
                "unchecked": "Not checked",
                "exofop": st.column_config.LinkColumn("ExoFOP", display_text="open"),
                "Analyse": st.column_config.LinkColumn("Analyse", display_text="inspect"),
            })

st.divider()
st.markdown(
    "**Before reporting anything:** inspect it here, search the TIC on "
    "[ExoFOP](https://exofop.ipac.caltech.edu/tess/), [SIMBAD](https://simbad.cds.unistra.fr) "
    "and [NASA ADS](https://ui.adsabs.harvard.edu) (published papers), then take it to "
    "[Planet Hunters TESS](https://www.zooniverse.org/projects/nora-dot-eisner/planet-hunters-tess) "
    "or submit a community TOI on ExoFOP.")
