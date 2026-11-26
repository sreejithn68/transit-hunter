"""
run_sector.py — search one shard of a TESS sector's 2-minute targets.

    python -m pipeline.run_sector --sector 108 --shard 3 --nshards 20 \
        --index index.csv.gz --out results/s108/shard_03.csv

Each star: download every sector it has (up to --max-sectors, newest first, but
always including the target sector), stitch, search for up to 3 signals, vet
each one. Writes one row per signal (and one 'none' row for stars with nothing),
so coverage can be audited. Stops cleanly before --deadline-min.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyse_star(tic: int, urls: list[str], sector: int) -> list[dict]:
    warnings.filterwarnings("ignore")
    from detection import search
    from pipeline.sources import fetch_star
    from vetting import vet

    base = {"tic": tic, "sector": sector}
    try:
        t, f, cents, meta = fetch_star(urls)
        out = search(t, f, max_planets=3, max_period=100.0)
    except Exception as e:
        return [{**base, "status": "error", "error": f"{type(e).__name__}: {e}"[:200]}]
    st = meta["star"]
    base.update(status="ok", ra=meta["ra"], dec=meta["dec"], tmag=meta["tmag"],
                star_radius=st.radius, teff=st.teff, logg=st.logg,
                contamination=st.contamination, n_sectors=meta["n_sectors"],
                baseline_d=round(float(t.max() - t.min()), 1))
    rows = []
    for i, s in enumerate(out.signals):
        if s.strength == "weak" and i > 0:
            continue
        rep = vet(*out.isolated(i), s, st, others=out.signals, centroids=cents)
        rows.append({**base, "signal": i + 1, "period": s.period, "t0": s.t0,
                     "duration_h": s.duration * 24, "depth_ppm": s.depth_ppm,
                     "snr": s.snr, "sde": s.sde, "n_transits": s.n_transits,
                     "strength": s.strength, "vetting": rep.verdict,
                     "radius_earth": rep.radius_earth,
                     "flags": "; ".join(f"{c.status}:{c.name} [{c.value}]" for c in rep.checks
                                        if c.status in ("warn", "fail"))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sectors", type=int, default=6)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--limit", type=int, default=0, help="only N stars (testing)")
    ap.add_argument("--deadline-min", type=float, default=330)
    a = ap.parse_args()

    t_start = time.time()
    ix = pd.read_csv(a.index)
    ix = ix[ix.sector <= a.sector]
    tics = np.sort(ix.loc[ix.sector == a.sector, "tic"].unique())[a.shard::a.nshards]
    if a.limit:
        tics = tics[:a.limit]
    groups = {tic: g.sort_values("sector") for tic, g in ix[ix.tic.isin(tics)].groupby("tic")}

    def urls_for(tic):
        g = groups[tic]
        # Only sectors within ~a year of the target (14 sectors ≈ 380 days): longer
        # baselines would need period grids too large to search properly.
        prior = g[(g.sector < a.sector) & (g.sector >= a.sector - 14)].tail(a.max_sectors - 1)
        return list(prior.url) + list(g[g.sector == a.sector].url)

    print(f"sector {a.sector} shard {a.shard}/{a.nshards}: {len(tics)} stars", flush=True)
    rows, done = [], 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(analyse_star, int(t), urls_for(t), a.sector): t for t in tics}
        for fut in as_completed(futs):
            try:
                rows.extend(fut.result())
            except Exception as e:
                rows.append({"tic": int(futs[fut]), "sector": a.sector, "status": "error",
                             "error": str(e)[:200]})
            done += 1
            if done % 25 == 0:
                el = (time.time() - t_start) / 60
                print(f"  {done}/{len(tics)} stars, {el:.0f} min", flush=True)
                pd.DataFrame(rows).to_csv(a.out, index=False)      # checkpoint
            if (time.time() - t_start) / 60 > a.deadline_min:
                print("deadline reached — writing partial results", flush=True)
                for f in futs:
                    f.cancel()
                break
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"done: {done}/{len(tics)} stars in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
