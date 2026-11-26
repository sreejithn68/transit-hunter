# 🪐 Transit Hunter

A web app that searches real NASA light curves for exoplanet transits.

Enter a star, and it pulls that star's brightness measurements from the public
[MAST archive](https://archive.stsci.edu) (Kepler / K2 / TESS), runs Box Least
Squares detection, and reports whether there's a candidate planet — with the
orbital period, transit depth, and a folded plot you can eyeball.

**Why it matters:** automated pipelines flag thousands of transit candidates, but
many are false positives. Fast, transparent vetting tools help separate real
planets from noise and eclipsing binaries. This is a first-pass vetting tool.

---

## How it works

1. **Fetch** — `lightcurves.py` downloads the star's light curves from MAST, preferring
   the official mission pipelines (TESS SPOC, then TESS-SPOC full-frame, then Kepler/K2)
   at a single cadence, and stitches up to 20 sectors/quarters together. It also reads
   the star's radius, temperature and gravity from the file headers.
2. **Detect** — `detection.py` cleans the data (trims ramps after gaps, bins, removes
   stellar variability with a robust local-quadratic filter) and runs **Box Least
   Squares** (or **Transit Least Squares** for a deep search) over a period grid fine
   enough for the full time span. It then:
   - re-detrends with the found transits masked, so the filter can't erode them;
   - rejects "signals" driven by a single glitch;
   - masks each strong signal and searches again — **multi-planet search**.
   Every signal gets an **SNR** (corrected for correlated noise) and an **SDE**
   (how clearly its period wins). Strong = SNR ≥ 8, SDE ≥ 9 and at least 3 transits.
3. **Vet** — `vetting.py` runs the classic false-positive checks: odd/even depths,
   secondary eclipse, transit vs. stellar wave, half-period alias, U- vs V-shape
   (trapezoid fit), implied planet size, duration vs. star size, contamination from
   neighbouring stars, and a second dip train on the same period.
4. **Cross-check** — `catalog.py` asks NASA's Exoplanet Archive about confirmed
   planets and TOIs at this position, and compares *periods* (a star can host several
   planets, so a position match isn't a signal match).
5. **Triage** — signal strength + vetting + catalog status become a plain
   recommendation (save / ignore), with a session log and CSV export.

All science code (`detection.py`, `vetting.py`, `catalog.py`, `synthetic.py`) is free
of web and plotting code and covered by tests:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q            # injection–recovery + vetting + catalog tests
python detection.py  # quick self-test
```

Checked on real data: recovers π Men c, Kepler-10 b and c, and TOI-700 b, c and d
(the Earth-sized habitable-zone planet) with no false "strong" signals.

**Honest scope:** this is a vetting and learning tool. NASA's own pipelines already
search all of this data; the value here is transparent triage and funnelling real
leads to humans.

---

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL it prints (usually http://localhost:8501).

---

## Roadmap

- **v1 — Detect** ✅ single-target BLS transit search with a clean verdict.
- **v2a — Vet** ✅ NASA catalog cross-check, multi-planet-aware triage, session
  log + CSV export, escalation guidance.
- **v3 — Detect+** ✅ fine period grid, correlated-noise SNR + SDE, transit-masked
  detrending, glitch rejection, multi-planet search, optional TLS deep search, and
  false-positive vetting. *(this release)*
- **v4 — Watch** — a scheduled sector pipeline: search every new TESS sector's
  2-minute targets, filter out anything already known anywhere, and push-notify
  genuinely new, vetted leads.
- **v5 — Share** — a shared community repository so multiple people can contribute
  and browse detections (needs a storage backend).
- **v6 — Judge** — an ML classifier trained on labelled candidates to score
  planet-vs-false-positive automatically.
- **v7 — Reach** — push toward secondary signals (candidate exomoons).

Built by Sree. Detection by Box Least Squares; data courtesy of NASA / MAST.
