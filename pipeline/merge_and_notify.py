"""
merge_and_notify.py — combine a sector's shards, keep only genuinely new,
vetted candidates, save results, and write a digest (digest.md; optionally pushed via ntfy.sh).

    NTFY_TOPIC=... python -m pipeline.merge_and_notify --sector 108 \
        --shards results/s108/shards --out results/s108

A candidate must be: strong (SNR >= 8, SDE >= 9, >= 3 transits), pass every
vetting check that can fail ("planet-like"), and match NOTHING in TOI, CTOI,
SPOC TCE, NASA confirmed or exoplanet.eu at that period. Shortlisted ones also
get SIMBAD / Gaia DR3 / VSX lookups.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.novelty import check, deep_checks, load_catalogs  # noqa: E402

EXOFOP = "https://exofop.ipac.caltech.edu/tess/target.php?id={}"
MAX_DEEP = 40


def notify(topic: str | None, title: str, body: str, priority: str = "default",
           click: str | None = None, tags: str = "telescope", out: str | None = None):
    """Write the digest to <out>/digest.md (read by the Claude scheduled task);
    also push it to ntfy.sh if NTFY_TOPIC is set (optional)."""
    if out:
        with open(os.path.join(out, "digest.md"), "w") as f:
            f.write(f"# {title}\n\n{body}\n")
    if not topic:
        print(f"{title}\n{body}")
        return
    headers = {"Title": title.encode("utf-8"), "Priority": priority, "Tags": tags,
               "Markdown": "yes"}
    if click:
        headers["Click"] = click
    r = requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"),
                      headers=headers, timeout=30)
    r.raise_for_status()


def shortlist(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.get("status", "ok") == "ok"] if "status" in df else df
    if "strength" not in ok:
        return ok.iloc[0:0]
    sel = (ok.strength == "strong") & (ok.vetting == "planet-like") & (ok.n_transits >= 3)
    return ok[sel].sort_values("snr", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", type=int, required=True)
    ap.add_argument("--shards", required=True, help="directory of shard CSVs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalogs", help="optional pre-downloaded catalog table (csv.gz)")
    a = ap.parse_args()
    topic = os.environ.get("NTFY_TOPIC")
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.shards, "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files if os.path.getsize(f) > 0], ignore_index=True)
    df.to_csv(os.path.join(a.out, "all_signals.csv.gz"), index=False)
    n_stars = df.tic.nunique()
    n_err = df[df.status == "error"].tic.nunique() if "status" in df else 0

    if a.catalogs:
        table, failed = pd.read_csv(a.catalogs, low_memory=False), []
    else:
        table, failed = load_catalogs()

    short = shortlist(df)
    rows = []
    for r in short.itertuples():
        ra = None if pd.isna(r.ra) else float(r.ra)
        dec = None if pd.isna(r.dec) else float(r.dec)
        nov = check(table, int(r.tic), ra, dec, float(r.period))
        if nov.status == "known":
            continue
        rows.append((r, nov))
    deep = rows[:MAX_DEEP]
    for r, nov in deep:
        deep_checks(nov, None if pd.isna(r.ra) else float(r.ra),
                    None if pd.isna(r.dec) else float(r.dec))

    cands = pd.DataFrame([{
        **{k: getattr(r, k) for k in ("tic", "ra", "dec", "tmag", "period", "t0", "duration_h",
                                      "depth_ppm", "snr", "sde", "n_transits", "radius_earth",
                                      "star_radius", "teff", "n_sectors", "flags")},
        "novelty": nov.status, "host_objects": " | ".join(nov.host_objects[:5]),
        "binary_flags": " | ".join(nov.flags),
        "unchecked": " | ".join(nov.unchecked + failed),
        "exofop": EXOFOP.format(int(r.tic)),
    } for r, nov in rows])
    if len(cands):
        # "clean" = no binary/variable evidence AND no vetting warnings at all.
        # On real data most impostors that pass vetting still carry a warning
        # (usually a V-shaped dip), so warned candidates are listed but don't
        # trigger a high-priority alert.
        warned = cands["flags"].fillna("").astype(str).str.contains("warn:")
        cands["clean"] = (cands.binary_flags == "") & ~warned
        cands = cands.sort_values(["clean", "novelty", "snr"], ascending=[False, False, False])
    cands.to_csv(os.path.join(a.out, "candidates.csv"), index=False)

    summary = {
        "sector": a.sector, "stars_searched": int(n_stars), "stars_failed": int(n_err),
        "strong_signals": int((df.get("strength") == "strong").sum()),
        "strong_planet_like": int(len(short)),
        "not_in_any_catalog": int(len(cands)),
        "clean_candidates": int(cands["clean"].sum()) if len(cands) else 0,
        "catalogs_unavailable": failed,
    }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # --- Phone digest ---------------------------------------------------------
    s = summary
    head = (f"Searched {s['stars_searched']:,} stars ({s['stars_failed']} failed). "
            f"{s['strong_planet_like']} strong planet-like signals; "
            f"{s['not_in_any_catalog']} not in any catalog; "
            f"**{s['clean_candidates']} clean**.")
    if failed:
        head += f"\n⚠️ Catalogs unavailable: {', '.join(failed)} — treat novelty as unconfirmed."
    if len(cands) == 0:
        notify(topic, f"Sector {a.sector}: nothing new", head, priority="low", tags="telescope",
               out=a.out)
        return
    lines = []
    for c in cands.head(5).itertuples():
        rp = f"{c.radius_earth:.1f} R⊕" if pd.notna(c.radius_earth) else "size ?"
        why = c.binary_flags or "; ".join(x.split(" [")[0] for x in str(c.flags).split("; ")
                                          if x.startswith("warn:"))
        flag = "" if c.clean else " ⚠️ " + why[:80]
        lines.append(f"• **TIC {int(c.tic)}** P={c.period:.3f} d, {c.depth_ppm:.0f} ppm, {rp}, "
                     f"SNR {c.snr:.0f}, {c.n_transits} transits — {c.novelty}{flag}")
    best = cands.iloc[0]
    prio = "high" if s["clean_candidates"] else "default"
    notify(topic, f"🪐 Sector {a.sector}: {s['clean_candidates']} clean candidate(s)",
           head + "\n\n" + "\n".join(lines) +
           "\n\nLeads, not discoveries — review before reporting.",
           priority=prio, click=best.exofop, tags="ringed_planet", out=a.out)


if __name__ == "__main__":
    main()
