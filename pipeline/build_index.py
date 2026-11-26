"""Build the (tic, sector, url) index for all sectors up to --sector."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.sources import available_sectors, build_index  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sector", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
secs = [s for s in available_sectors() if s <= a.sector]
ix = build_index(secs)
ix.to_csv(a.out, index=False)
print(f"index: {len(ix)} light curves, {ix.tic.nunique()} stars, sectors {secs[0]}–{secs[-1]}")
