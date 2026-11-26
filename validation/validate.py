"""
validate.py — measure false negatives and false positives on REAL TESS data.

Three samples:
  * planets  — TOIs confirmed as planets (CP/KP): should be found + pass vetting
  * impostors — TOIs that follow-up showed are false positives (FP): vetting
                should flag them (these already fooled NASA's own pipeline)
  * quiet    — random 2-min targets that were never TOIs: should give no
                strong, planet-like signal (false-alarm rate)

Usage:  python validation/validate.py --n 15 --sectors 2 --workers 4
Writes validation/results.csv and prints a summary.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HERE = os.path.join(ROOT, "validation")


def period_match(p, true, tol=0.01):
    return any(abs(p - true * f) / (true * f) < tol for f in (1, 2, 0.5))


def analyse(job):
    warnings.filterwarnings("ignore")
    from detection import search
    from lightcurves import fetch
    from vetting import vet
    sample, tid, true_p, n_sectors = job
    row = {"sample": sample, "tic": tid, "true_period": true_p}
    try:
        d = fetch(f"TIC {tid}", "TESS", max_segments=n_sectors)
        out = search(d.time, d.flux, max_planets=2)
    except Exception as e:
        row["error"] = str(e)[:120]
        return row
    sigs = []
    for i, s in enumerate(out.signals):
        rep = vet(*out.isolated(i), s, d.star, others=out.signals, centroids=d.centroids)
        sigs.append((s, rep))
    best = sigs[0]
    # For planets/impostors, judge the signal at the catalogued period if found.
    if true_p and np.isfinite(true_p):
        hit = [x for x in sigs if period_match(x[0].period, true_p)]
        if hit:
            best = hit[0]
        row["found_period"] = bool(hit) and hit[0][0].strength != "weak"
    s, rep = best
    row.update(period=round(s.period, 5), depth_ppm=round(s.depth_ppm), snr=round(s.snr, 1),
               sde=round(s.sde, 1), n_transits=s.n_transits, strength=s.strength,
               vetting=rep.verdict,
               flags="; ".join(f"{c.status}:{c.name}" for c in rep.checks
                               if c.status in ("warn", "fail")),
               any_strong_planetlike=any(x[0].strength == "strong" and
                                         x[1].verdict == "planet-like" for x in sigs))
    return row


def build_jobs(n, n_sectors, seed=1):
    rng = np.random.default_rng(seed)
    pool = pd.read_csv(os.path.join(HERE, "toi_pool.csv"))
    pool = pool.drop_duplicates("tid")
    planets = pool[pool.tfopwg_disp.isin(["CP", "KP"])].sample(n, random_state=seed)
    fps = pool[pool.tfopwg_disp == "FP"].sample(n, random_state=seed)
    tois = set(pd.read_csv(os.path.join(HERE, "all_toi_tids.csv")).tid)
    targets = pd.read_csv(os.path.join(HERE, "targets_S070.csv"), comment="#")
    targets = targets[(targets.Tmag < 11) & ~targets.TICID.isin(tois)]
    quiet = targets.sample(n, random_state=int(rng.integers(1e6)))
    jobs = [("planet", int(r.tid), float(r.pl_orbper), n_sectors) for r in planets.itertuples()]
    jobs += [("impostor", int(r.tid), float(r.pl_orbper), n_sectors) for r in fps.itertuples()]
    jobs += [("quiet", int(r.TICID), None, n_sectors) for r in quiet.itertuples()]
    return jobs


def summarise(df):
    lines = []
    ok = df[df.get("error").isna()] if "error" in df else df
    for name, g in ok.groupby("sample"):
        n = len(g)
        if name == "planet":
            found = g.found_period.fillna(False).astype(bool)
            passed = found & (g.vetting != "likely false positive")
            lines.append(f"planets   n={n}: found {found.sum()}/{n}, found AND passed vetting "
                         f"{passed.sum()}/{n}  → false negatives {n - passed.sum()}")
        elif name == "impostor":
            caught = g.vetting != "planet-like"
            weak = ~g.found_period.fillna(False).astype(bool)
            lines.append(f"impostors n={n}: flagged by vetting {caught.sum()}/{n}, not detected "
                         f"{(weak & ~caught).sum()}, slipped through as planet-like "
                         f"{(~caught & ~weak).sum()}")
        else:
            fa = g.any_strong_planetlike.astype(bool)
            lines.append(f"quiet     n={n}: false alarms (strong + planet-like) {fa.sum()}/{n}")
    if "error" in df:
        lines.append(f"errors: {df.error.notna().sum()}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--sectors", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from multiprocessing import Pool
    jobs = build_jobs(args.n, args.sectors)
    with Pool(args.workers) as p:
        rows, it = [], p.imap_unordered(analyse, jobs)
        for i in range(1, len(jobs) + 1):
            try:
                r = it.next(timeout=900)   # a hung MAST download must not stall the run
            except Exception:
                print(f"stopped: {len(jobs) - i + 1} job(s) hung", flush=True)
                break
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['sample']:8s} TIC {r['tic']}: "
                  f"{r.get('strength', 'ERR')} {r.get('vetting', r.get('error', ''))} {r.get('flags', '')}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results.csv"), index=False)
    print("\n" + summarise(df))
