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
3. **Vet** — `catalog.py` cross-checks the star against NASA's confirmed-planet
   and TOI catalogs (via `astroquery`), and — crucially — compares the *detected
   period* to any known planet's period. A star can host several planets, so a
   position match isn't a signal match. It returns one of: *recovered a known
   planet*, *known host but a different period (possible extra signal)*, or
   *uncatalogued*.
4. **Triage** — the app combines signal strength + catalog status into a plain
   recommendation (save / ignore), lets you log detections and export them as CSV,
   and points genuinely interesting leads toward real vetting communities.

The detection and vetting logic are deliberately isolated in `detection.py` and
`catalog.py` (no web or plotting code) so they can be read and unit-tested on their
own. Run `python detection.py` to see it recover a planet injected into synthetic
data.

**Honest scope:** this is a vetting and learning tool, not an automated discovery
pipeline. NASA's own pipelines already run detection across all this data; the value
here is transparent triage and funnelling real leads to humans.

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
  log + CSV export, escalation guidance. *(this release)*
- **v2b — Share** — a shared community repository so multiple people can contribute
  and browse detections (needs a storage backend).
- **v3 — Watch** — a scheduled watchlist pipeline that checks for new TESS data on
  chosen stars and reports transit-like signals.
- **v4 — Judge** — an ML classifier trained on labelled candidates to score
  planet-vs-false-positive automatically.
- **v5 — Reach** — push toward secondary signals (candidate exomoons).

Built by Sree. Detection by Box Least Squares; data courtesy of NASA / MAST.
