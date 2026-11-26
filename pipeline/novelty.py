"""
novelty.py — has anyone, anywhere, already found or flagged this signal?

Bulk catalogs (downloaded once per run, matched locally):
  * ExoFOP-TESS TOIs                      — NASA's official candidates
  * ExoFOP-TESS CTOIs                     — community candidates (Planet Hunters TESS,
                                            other teams' surveys)
  * SPOC TCEs, every sector               — signals NASA's pipeline already flagged,
                                            including ones never promoted to TOIs
  * NASA Exoplanet Archive confirmed      — confirmed planets
  * Exoplanet.eu                          — independent catalogue incl. non-NASA claims

Per-candidate lookups (shortlist only):
  * SIMBAD object type                    — known eclipsing binary / variable?
  * Gaia DR3 eclipsing binaries + RUWE    — binary star evidence
  * AAVSO VSX                             — known variable star

A signal counts as KNOWN if its period matches a catalogued one (including
small integer-ratio aliases, to be conservative). Nothing is ever called new
just because a lookup failed: failed sources are reported as "unchecked".
"""

from __future__ import annotations

import io
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.sources import get

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
CTOI_URL = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
TCE_DIR = "https://archive.stsci.edu/missions/tess/catalogs/tce/"
EU_URL = "https://exoplanet.eu/catalog/csv/"
NASA_TAP = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=csv&query="
            "select+pl_name,tic_id,ra,dec,pl_orbper+from+pscomppars")

PIXEL_ARCSEC = 21.0          # one TESS pixel
RATIOS = sorted({n / m for n in range(1, 6) for m in range(1, 6)})


def _tic(x):
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else None


def load_catalogs() -> tuple[pd.DataFrame, list[str]]:
    """All bulk catalogs as one table: source, name, tic, ra, dec, period.
    Returns (table, list_of_sources_that_failed)."""
    frames, failed = [], []

    def toi():
        d = pd.read_csv(io.StringIO(get(TOI_URL).text), low_memory=False)
        return pd.DataFrame({"source": "TOI", "name": "TOI-" + d["TOI"].astype(str),
                             "tic": d["TIC ID"], "ra": np.nan, "dec": np.nan,
                             "period": pd.to_numeric(d["Period (days)"], errors="coerce"),
                             "note": d["TFOPWG Disposition"].fillna("")})

    def ctoi():
        d = pd.read_csv(io.StringIO(get(CTOI_URL).text), low_memory=False)
        return pd.DataFrame({"source": "CTOI", "name": "CTOI " + d["CTOI"].astype(str),
                             "tic": d["TIC ID"], "ra": np.nan, "dec": np.nan,
                             "period": pd.to_numeric(d["Period (days)"], errors="coerce"),
                             "note": d["User"].fillna("").astype(str)})

    def nasa():
        d = pd.read_csv(io.StringIO(get(NASA_TAP).text))
        return pd.DataFrame({"source": "NASA confirmed", "name": d["pl_name"],
                             "tic": d["tic_id"].map(_tic), "ra": d["ra"], "dec": d["dec"],
                             "period": d["pl_orbper"], "note": ""})

    def eu():
        d = pd.read_csv(io.StringIO(get(EU_URL, timeout=300).text), low_memory=False)
        return pd.DataFrame({"source": "exoplanet.eu", "name": d["name"], "tic": np.nan,
                             "ra": pd.to_numeric(d["ra"], errors="coerce"),
                             "dec": pd.to_numeric(d["dec"], errors="coerce"),
                             "period": pd.to_numeric(d["orbital_period"], errors="coerce"),
                             "note": d.get("planet_status", "")})

    def tces():
        html = get(TCE_DIR).text
        files = sorted(set(re.findall(r'href="([^"]*tcestats\.csv)"', html)))

        def one(f):
            d = pd.read_csv(io.StringIO(get(TCE_DIR + f).text), comment="#", low_memory=False)
            return pd.DataFrame({"source": "SPOC TCE", "name": "TCE " + d["tceid"].astype(str),
                                 "tic": d["ticid"], "ra": np.nan, "dec": np.nan,
                                 "period": pd.to_numeric(d["tce_period"], errors="coerce"),
                                 "note": d["sectors"].astype(str)})
        with ThreadPoolExecutor(8) as ex:
            return pd.concat(list(ex.map(one, files)), ignore_index=True)

    for name, fn in (("TOI", toi), ("CTOI", ctoi), ("NASA confirmed", nasa),
                     ("exoplanet.eu", eu), ("SPOC TCE", tces)):
        try:
            frames.append(fn())
        except Exception as e:  # keep going, but remember what we couldn't check
            failed.append(f"{name} ({type(e).__name__})")
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["source", "name", "tic", "ra", "dec", "period", "note"])
    table["tic"] = pd.to_numeric(table["tic"], errors="coerce")
    return table, failed


def period_related(p, q, tol=0.003) -> bool:
    if not (p and q) or not (np.isfinite(p) and np.isfinite(q)) or q <= 0:
        return False
    return any(abs(p - q * r) / (q * r) < tol for r in RATIOS)


@dataclass
class Novelty:
    status: str                       # "known" | "new signal on known host" | "uncatalogued"
    matches: list[str] = field(default_factory=list)      # period matches
    host_objects: list[str] = field(default_factory=list)  # same star, other periods
    flags: list[str] = field(default_factory=list)        # binary / variable evidence
    unchecked: list[str] = field(default_factory=list)


def check(table: pd.DataFrame, tic: int, ra, dec, period: float) -> Novelty:
    same = table["tic"] == tic
    if ra is not None and dec is not None:
        has_pos = table["ra"].notna()
        dra = (table["ra"] - ra) * np.cos(np.radians(dec))
        sep = np.hypot(dra, table["dec"] - dec) * 3600
        same |= has_pos & (sep < PIXEL_ARCSEC)
    host = table[same]
    matches, others = [], []
    for r in host.itertuples():
        label = f"{r.source}: {r.name}" + (f" ({r.period:.4f} d)" if np.isfinite(r.period) else "")
        if np.isfinite(r.period) and period_related(period, r.period):
            matches.append(label)
        else:
            others.append(label)
    status = "known" if matches else "new signal on known host" if others else "uncatalogued"
    return Novelty(status, sorted(set(matches)), sorted(set(others)))


def _simbad(c, u):
    from astroquery.simbad import Simbad
    s = Simbad()
    s.TIMEOUT = 45
    s.add_votable_fields("otype")
    res = s.query_region(c, radius=PIXEL_ARCSEC * u.arcsec)
    if res is None:
        return []
    col = next(k for k in res.colnames if k.lower() == "otype")
    types = {str(t) for t in res[col]}
    bad = {t for t in types if t.startswith(("EB", "SB", "El*", "RR", "Ce", "dS", "Pu", "BY", "Ro", "V*"))}
    return [f"SIMBAD: variable/binary types nearby: {', '.join(sorted(bad))}"] if bad else []


def _gaia(ra, dec):
    from astroquery.gaia import Gaia
    q = (f"SELECT g.source_id, g.ruwe, e.source_id AS eb FROM gaiadr3.gaia_source AS g "
         f"LEFT JOIN gaiadr3.vari_eclipsing_binary AS e ON e.source_id = g.source_id "
         f"WHERE 1=CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra}, {dec}, "
         f"{PIXEL_ARCSEC / 3600}))")
    res = Gaia.launch_job(q).get_results()
    out = []
    if len(res):
        if np.any(~np.ma.getmaskarray(res["eb"])):
            out.append("Gaia DR3: an eclipsing binary within one TESS pixel")
        ruwe = np.ma.filled(np.ma.asarray(res["ruwe"], dtype=float), np.nan)
        if np.isfinite(ruwe).any() and np.nanmax(ruwe) > 1.4:
            out.append(f"Gaia DR3: RUWE {np.nanmax(ruwe):.1f} (>1.4 suggests an unresolved binary)")
    return out


def _vsx(c, u):
    from astroquery.vizier import Vizier
    v = Vizier(columns=["Name", "Type", "Period"])
    v.TIMEOUT = 45
    res = v.query_region(c, radius=PIXEL_ARCSEC * u.arcsec, catalog="B/vsx/vsx")
    if len(res) and len(res[0]):
        items = [f"{r['Name']} ({r['Type']})" for r in res[0]]
        return ["VSX: known variable(s) nearby: " + ", ".join(items[:3])]
    return []


def deep_checks(nov: Novelty, ra, dec, timeout: float = 60) -> Novelty:
    """Live lookups for shortlisted candidates: SIMBAD, Gaia DR3, VSX. Each has
    a hard timeout; a service that fails or hangs is reported as unchecked."""
    if ra is None or dec is None:
        nov.unchecked += ["SIMBAD", "Gaia DR3", "VSX"]
        return nov
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
    c = SkyCoord(ra * u.deg, dec * u.deg)
    ex = ThreadPoolExecutor(3)
    jobs = {"SIMBAD": ex.submit(_simbad, c, u), "Gaia DR3": ex.submit(_gaia, ra, dec),
            "VSX": ex.submit(_vsx, c, u)}
    for name, fut in jobs.items():
        try:
            nov.flags += fut.result(timeout=timeout)
        except FTimeout:
            nov.unchecked.append(f"{name} (timed out)")
        except Exception as e:
            nov.unchecked.append(f"{name} ({type(e).__name__})")
    ex.shutdown(wait=False, cancel_futures=True)
    return nov
