"""Decide which sector to search: the one asked for, else the newest not yet done."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.sources import available_sectors  # noqa: E402

asked = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
done_file = os.path.join("results", "processed.txt")
done = set()
if os.path.exists(done_file):
    done = {int(x) for x in open(done_file).read().split() if x.strip().isdigit()}
if asked:
    sector, run = int(asked), True
else:
    sector = max(available_sectors())
    run = sector not in done
print(f"sector={sector} run={str(run).lower()}")
out = os.environ.get("GITHUB_OUTPUT")
if out:
    with open(out, "a") as f:
        f.write(f"sector={sector}\nrun={str(run).lower()}\n")
