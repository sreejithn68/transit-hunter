# 🪐 Transit Hunter

Searches real NASA light curves for exoplanet transits — one star at a time in
the web app, or every 2-minute TESS star in a sector with the automated pipeline.

Enter a star and the app pulls its brightness measurements from the public
[MAST archive](https://archive.stsci.edu) (Kepler / K2 / TESS), searches for
repeating dips, vets each signal against the classic false-positive checks, and
tells you whether the signal is already known — with the period, depth, implied
planet size and folded plots you can eyeball.

**Why it matters:** automated pipelines flag thousands of transit candidates, but
many are false positives (mostly eclipsing binary stars). Fast, transparent
vetting separates real planets from noise and binaries, and funnels the few
genuinely new leads to humans.

---

## How it works

### 1. Fetch — `lightcurves.py`
Downloads the star's light curves from MAST, preferring the official pipelines
(TESS SPOC 2-min → TESS-SPOC full-frame → QLP; Kepler; K2), at a single cadence,
and stitches up to 8 sectors/quarters. Reads the star's radius, temperature,
gravity and crowding (`CROWDSAP`) from the file headers, plus the **pixel
centroids** (where the star's image sits on the detector) for the blend test.

### 2. Detect — `detection.py`
- **Clean:** drop flagged points, trim the 6-hour ramps after data gaps, bin
  adaptively (10–30 min depending on time span).
- **Detrend:** remove stellar variability with a robust local-quadratic filter
  (1-day window), with known transits masked so the filter can't erode them.
- **Search:** Box Least Squares (or Transit Least Squares as an optional deep
  search) over a period grid spaced so no transit can slip between grid points
  (Δ*P*/*P* = shortest duration ÷ (3 × time span)), with transit durations
  searched in period bands so long-period planets get long durations.
- **Score:** **SNR** = depth ÷ depth uncertainty × red-noise factor (Pont 2006,
  so correlated noise can't inflate it); **SDE** = how far the winning peak stands
  above the rest of the periodogram. *Strong* = SNR ≥ 8, SDE ≥ 9 and ≥ 3 transits;
  *plausible* = SNR ≥ 7.1, SDE ≥ 7, ≥ 2 transits.
- **Glitch rejection:** a "signal" whose power comes mostly from one event is cut
  out and the search repeats.
- **Multi-planet search:** mask each strong signal and search again (up to 3–5).
  Leftover "echoes" that sit on an earlier signal's own transits are discarded.

### 3. Vet — `vetting.py`
Each signal is checked on a light curve with the *other* signals removed:

| Check | Catches | Rule |
|---|---|---|
| Odd/even depths | binary at half its true period | differ > 4σ **and** > 15% (σ from the scatter of individual transit depths, so sector-to-sector depth differences don't count) |
| Secondary eclipse | self-luminous companion | dip at phase 0.5 > 3σ and > 10% of the transit |
| Matching second signal | binary's second eclipse found separately | fail; *warn* if < 10% as deep (a hot planet passing behind its star) |
| Transit vs. stellar wave | spots / pulsation | sinusoid fits better |
| Half-period alias | wrong period | signal also at *P*/2 |
| Dip shape (trapezoid fit) | grazing binary (V-shape) | ingress ≥ 40% of duration |
| Implied size | stars, not planets | > 2 R♃ fail, > 1.5 R♃ warn |
| Duration vs. star size | dip too long for any orbit | > 2× the longest possible |
| Light from neighbours | dilution / blends | crowding > 20% warn |
| **Image shift during dip** | **blended eclipsing binary** next door | implied source offset = image shift ÷ depth; fail if > 1 pixel (21″) at > 5σ |
| Number of transits | chance alignment | < 3 warn |

Verdict: any fail → *likely false positive*; ≥ 2 warnings → *suspicious*;
otherwise *planet-like*.

### 4. Is it already known? — `catalog.py` (app) and `pipeline/novelty.py` (pipeline)
A signal counts as known if any catalogue lists the **same star** (TIC ID or
within one TESS pixel) at a **related period** (ratios n/m up to 5, ±0.3%):
ExoFOP **TOIs** and community **CTOIs**, **every SPOC TCE** NASA's pipeline has
ever flagged, the **NASA Exoplanet Archive** confirmed planets, and
**exoplanet.eu**. Shortlisted candidates also get live **SIMBAD**, **Gaia DR3**
(eclipsing binaries, RUWE) and **AAVSO VSX** lookups. A lookup that fails is
reported as *unchecked* — nothing is ever called new because a service was down.

### 5. Triage
Signal strength + vetting + catalogue status become a plain recommendation, with
a session log and CSV export in the app.

---

## The sector pipeline (v4 — Watch)

`.github/workflows/sector.yml` runs every Monday 06:00 UTC (and on demand for
backfill):

1. **Plan** — pick the newest TESS sector not yet processed; build an index of
   every SPOC 2-minute light curve on MAST (~1.7 M files).
2. **Search** — 20 parallel jobs split the sector's ~10–20 k stars. Each star is
   stitched with up to 6 of its sectors from the past ~14, then searched and
   vetted exactly as above.
3. **Merge** — shortlist = strong + planet-like + ≥ 3 transits; drop anything
   known; run live binary/variable lookups; write
   `results/sN/{summary.json, candidates.csv, digest.md, all_signals.csv.gz}`.
   A candidate is **clean** only with no binary flags *and* no vetting warnings.
4. **Report** — a Claude cloud scheduled task reads the newest results every
   Monday 10:00 UTC and writes a plain-English digest you can read in the Claude
   app. (Optional: set a `NTFY_TOPIC` secret for phone pushes via ntfy.sh.)

The **📡 Sector Feed** page in the app lists every sector's candidates, with links
to inspect each one in the main view (`?target=TIC …`) or on ExoFOP.

Run it locally on a few stars:

```bash
pip install -r pipeline/requirements.txt
python -m pipeline.build_index --sector 108 --out index.csv.gz
python -m pipeline.run_sector --sector 108 --shard 0 --nshards 500 --index index.csv.gz --out shard0.csv --limit 10
```

---

## How well does it work?

**Unit tests** (`pytest -q`, 31 tests): synthetic injection–recovery (single
and multi-planet, long baselines, glitches, long periods), every vetting check,
catalogue matching.

**Real-data benchmark** (`python validation/validate.py --n 25`): 75 real TESS
stars, 2 sectors each.

| Sample | What should happen | Result |
|---|---|---|
| 25 confirmed planets | found and passed | **23 / 25** (1 not detected, 1 had only 2 transits) |
| 25 TOIs that NASA follow-up proved false | flagged | **17 / 25** flagged; 6 of the other 8 carry a warning; **2 / 25** pass completely clean |
| 25 random never-flagged stars | no strong planet-like signal | **0 / 25 false alarms** |

The impostors that pass were ruled out by radial-velocity or ground-based imaging
follow-up — information a light curve alone does not contain. So pipeline
candidates are **leads, not discoveries**: review them, then report via
[ExoFOP CTOI](https://exofop.ipac.caltech.edu/tess/) or Planet Hunters TESS.

Also checked by hand: recovers π Men c, Kepler-10 b and c, and TOI-700 b, c and d.

---

## Run the app locally

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
  detrending, glitch rejection, multi-planet search, optional TLS deep search,
  false-positive vetting incl. centroid (blend) test, real-data benchmark.
- **v4 — Watch** ✅ weekly sector pipeline on GitHub Actions, "already known
  anywhere?" gate across 5 catalogues + 3 live services, weekly digest.
- **v5 — Share** — a shared community repository so multiple people can contribute
  and browse detections (needs a storage backend).
- **v6 — Judge** — an ML classifier trained on labelled candidates to score
  planet-vs-false-positive automatically.
- **v7 — Reach** — push toward secondary signals (candidate exomoons).

Built by Sree. Detection by Box Least Squares; data courtesy of NASA / MAST.
