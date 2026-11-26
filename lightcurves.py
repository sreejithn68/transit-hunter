"""
lightcurves.py — fetch and stitch real light curves from NASA MAST.

Shared by the web app and the sector pipeline. Prefers the official mission
pipelines (TESS SPOC 2-min, then TESS-SPOC full-frame, then Kepler/K2) and a
single cadence, so we never stitch incompatible products together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vetting import Star

AUTHOR_PREFERENCE = {
    "TESS": ["SPOC", "TESS-SPOC", "QLP"],
    "Kepler": ["Kepler"],
    "K2": ["K2", "EVEREST"],
}


@dataclass
class LightCurveData:
    time: np.ndarray
    flux: np.ndarray
    label: str
    ra: float | None
    dec: float | None
    star: Star
    tic: int | None
    n_segments: int          # sectors / quarters / campaigns used
    centroids: tuple | None  # (time, col, row) pixel centroids, for the blend test
    n_available: int         # how many existed
    baseline_days: float


def _num(meta, *keys):
    for k in keys:
        v = meta.get(k)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            return v
    return None


def star_from_meta(meta) -> Star:
    crowd = _num(meta, "CROWDSAP")
    contamination = (1.0 / crowd - 1.0) if crowd and 0 < crowd <= 1 else None
    return Star(radius=_num(meta, "RADIUS"), teff=_num(meta, "TEFF"),
                logg=_num(meta, "LOGG"), contamination=contamination)


def pick_products(search, mission: str):
    """Choose one author (pipeline) and one cadence from a lightkurve search."""
    if len(search) == 0:
        return search
    missions = [mission] if mission != "Auto" else ["TESS", "Kepler", "K2"]
    tbl = search.table
    for m in missions:
        for author in AUTHOR_PREFERENCE[m]:
            sel = np.array([(str(a) == author) and (m in str(mi))
                            for a, mi in zip(tbl["author"], tbl["mission"])])
            if not sel.any():
                continue
            sub = search[sel]
            exptimes = np.asarray(sub.table["exptime"], dtype=float)
            # Most common cadence; prefer 120 s for TESS SPOC.
            vals, counts = np.unique(exptimes, return_counts=True)
            best = 120.0 if 120.0 in vals else vals[np.argmax(counts)]
            return sub[exptimes == best]
    return search[:0]


def fetch(target: str, mission: str = "Auto", max_segments: int = 8) -> LightCurveData:
    import lightkurve as lk

    kwargs = {} if mission == "Auto" else {"mission": mission}
    search = lk.search_lightcurve(target, **kwargs)
    products = pick_products(search, mission)
    if len(products) == 0:
        raise LookupError(f"No official light curve found for '{target}'.")
    n_available = len(products)
    products = products[-max_segments:]            # most recent segments
    collection = products.download_all(quality_bitmask="default")
    if collection is None or len(collection) == 0:
        raise LookupError(f"Light curves for '{target}' couldn't be downloaded.")

    meta = collection[0].meta
    lc = collection.stitch(lambda x: x.remove_nans().normalize()).remove_nans()
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    centroids = None
    if "centroid_col" in lc.colnames and "centroid_row" in lc.colnames:
        centroids = (time, np.asarray(lc["centroid_col"].value, dtype=float),
                     np.asarray(lc["centroid_row"].value, dtype=float))
    ra, dec = _num(meta, "RA_OBJ", "RA"), _num(meta, "DEC_OBJ", "DEC")
    tic = meta.get("TICID")
    author = str(products.table["author"][0])
    seg_word = {"TESS": "sectors", "Kepler": "quarters", "K2": "campaigns"}.get(
        str(meta.get("MISSION", meta.get("TELESCOP", ""))), "segments")
    label = f"{target} · {author} · {len(collection)} {seg_word}"
    return LightCurveData(time, flux, label, ra, dec, star_from_meta(meta),
                          int(tic) if tic else None, len(collection), centroids, n_available,
                          float(time.max() - time.min()))
