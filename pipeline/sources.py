"""
sources.py — where TESS light curves live, and how to read them fast.

MAST publishes one download script per sector listing the URL of every SPOC
2-minute light curve. We parse those into an index (tic, sector, url) once per
run, so each star's data from ALL sectors can be fetched directly — no slow
per-star archive searches.
"""

from __future__ import annotations

import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from vetting import Star

SCRIPT_DIR = "https://archive.stsci.edu/missions/tess/download_scripts/sector/"
SCRIPT_URL = SCRIPT_DIR + "tesscurl_sector_{}_lc.sh"
_LINE = re.compile(r"(https://\S+?-s(\d{4})-(\d{16})-\d{4}-s_lc\.fits)")

session = requests.Session()
session.headers["User-Agent"] = "transit-hunter-pipeline"


def get(url, timeout=120, tries=4):
    for k in range(tries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def available_sectors() -> list[int]:
    html = get(SCRIPT_DIR).text
    secs = {int(s) for s in re.findall(r"tesscurl_sector_(\d+)_lc\.sh", html)}
    return sorted(s for s in secs if s < 1000)        # ignore odd test entries


def sector_index(sector: int) -> pd.DataFrame:
    txt = get(SCRIPT_URL.format(sector)).text
    rows = [(int(m.group(3)), int(m.group(2)), m.group(1)) for m in _LINE.finditer(txt)]
    return pd.DataFrame(rows, columns=["tic", "sector", "url"])


def build_index(sectors, workers=8) -> pd.DataFrame:
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(sector_index, sectors))
    return pd.concat(parts, ignore_index=True).drop_duplicates(["tic", "sector"])


def _num(h, key):
    try:
        v = float(h.get(key))
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def read_lc(content: bytes):
    """Parse one SPOC light-curve FITS: good-quality PDCSAP flux, normalised."""
    from astropy.io import fits
    with fits.open(io.BytesIO(content), memmap=False) as h:
        d = h[1].data
        t = np.asarray(d["TIME"], dtype=float)
        f = np.asarray(d["PDCSAP_FLUX"], dtype=float)
        q = np.asarray(d["QUALITY"])
        ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
        cc = np.asarray(d["MOM_CENTR1"], dtype=float)[ok]
        cr = np.asarray(d["MOM_CENTR2"], dtype=float)[ok]
        t, f = t[ok], f[ok]
        if t.size:
            f = f / np.nanmedian(f)
        h0, h1 = h[0].header, h[1].header
        crowd = _num(h1, "CROWDSAP")
        meta = {
            "ra": _num(h0, "RA_OBJ"), "dec": _num(h0, "DEC_OBJ"),
            "tmag": _num(h0, "TESSMAG"),
            "star": Star(radius=_num(h0, "RADIUS"), teff=_num(h0, "TEFF"),
                         logg=_num(h0, "LOGG"),
                         contamination=(1 / crowd - 1) if crowd and 0 < crowd <= 1 else None),
        }
    return t, f, cc, cr, meta


def fetch_star(urls: list[str], workers: int = 4):
    """Download and stitch every sector for one star."""
    with ThreadPoolExecutor(workers) as ex:
        blobs = list(ex.map(lambda u: get(u).content, urls))
    ts, fs, ccs, crs, meta = [], [], [], [], None
    for b in blobs:
        t, f, cc, cr, m = read_lc(b)
        if t.size:
            ts.append(t)
            fs.append(f)
            ccs.append(cc - np.nanmedian(cc))   # centroids per sector (camera changes)
            crs.append(cr - np.nanmedian(cr))
            meta = meta or m
    if not ts:
        raise ValueError("no good data")
    t = np.concatenate(ts)
    o = np.argsort(t)
    f, cc, cr = (np.concatenate(x)[o] for x in (fs, ccs, crs))
    meta["n_sectors"] = len(ts)
    return t[o], f, (t[o], cc, cr), meta
