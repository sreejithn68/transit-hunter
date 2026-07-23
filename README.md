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

1. **Fetch** — `lightkurve` downloads the star's light curve and cleans it
   (removes bad points, normalises, flattens out slow stellar variability).
2. **Detect** — `detection.py` runs Box Least Squares: it slides a box-shaped
   dip across thousands of trial periods and finds the one that best explains a
   repeating transit.
3. **Judge** — the app reports period, depth, and a signal-to-noise score, then
   folds the data so you can see the transit stack up at phase 0.

The detection maths is deliberately isolated in `detection.py` (no web or plotting
code) so it can be read and unit-tested on its own. Run `python detection.py` to
see it recover a planet injected into synthetic data.

---

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL it prints (usually http://localhost:8501).

---

## Roadmap

- **v1 (this)** — single-target BLS transit search with a clean verdict.
- **v2** — an ML classifier trained on labelled Kepler/TESS candidates to score
  planet-vs-false-positive automatically.
- **v3** — a live daily pipeline that pulls new TESS data and emails a findings
  report.
- **v4** — push toward secondary signals (candidate exomoons).

Built by Sree. Detection by Box Least Squares; data courtesy of NASA / MAST.
