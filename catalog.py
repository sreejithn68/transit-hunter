"""
catalog.py — the vetting brain.

Given a star's sky position and the period we detected, ask NASA's Exoplanet
Archive: is there already a known planet or candidate here? And crucially — does
our detected period MATCH a known one, or is it different (a star can host several
planets, so a position match is not the same as a signal match)?

Pure logic + one external query. No web/plotting code, so it can be reasoned about
on its own. The archive query runs at app runtime (needs internet); everything
degrades gracefully if the archive is unreachable.
"""

from dataclasses import dataclass, field


@dataclass
class KnownObject:
    name: str
    period: float | None   # days, if the catalog lists one
    kind: str              # "confirmed" or "candidate (TOI)"


@dataclass
class CatalogVerdict:
    status: str                 # see STATUS_* below
    headline: str               # one-line plain-English summary
    detail: str                 # longer explanation for the user
    recommend_save: bool        # our suggestion; user always overrides
    known: list[KnownObject] = field(default_factory=list)


# Status codes
STATUS_KNOWN_MATCH = "known_match"            # a known planet with the same period
STATUS_KNOWN_DIFF = "known_host_diff_period"  # known host, but our period differs
STATUS_UNCATALOGUED = "uncatalogued"          # nothing known at this position
STATUS_NO_POSITION = "no_position"            # we couldn't get coordinates
STATUS_ARCHIVE_DOWN = "archive_unreachable"   # query failed


def _period_matches(p_detected: float, p_known: float | None, tol: float = 0.02) -> bool:
    """True if the detected period matches a known period. We allow ONLY the two
    standard BLS aliases (2x and 1/2 the true period) plus an exact match. We
    deliberately do NOT treat 3x/1x3 etc. as matches: a star can host multiple
    planets in resonance, and being too eager to call something a 'harmonic' would
    hide a genuine additional-planet signal — the very thing worth surfacing."""
    if not p_known or p_known <= 0 or not p_detected or p_detected <= 0:
        return False
    for factor in (1.0, 2.0, 0.5):
        target = p_known * factor
        if abs(p_detected - target) / target < tol:
            return True
    return False


def _extract_period(row) -> float | None:
    """Pull an orbital period from a catalog row, trying the column names the
    confirmed-planet and TOI tables use. Returns None if absent."""
    for col in ("pl_orbper", "toi_period", "period"):
        try:
            val = row[col]
        except (KeyError, TypeError, IndexError):
            continue
        try:
            f = float(val)
            if f == f and f > 0:  # not NaN, positive
                return f
        except (TypeError, ValueError):
            continue
    return None


def _extract_name(row, kind: str) -> str:
    for col in ("pl_name", "hostname", "toi", "toipfx", "tid"):
        try:
            val = row[col]
        except (KeyError, TypeError, IndexError):
            continue
        if val is not None and str(val).strip() and str(val) != "--":
            prefix = "TOI-" if col in ("toi", "toipfx") else ""
            return f"{prefix}{val}"
    return "unnamed object"


def crosscheck(
    ra: float | None,
    dec: float | None,
    detected_period: float,
    snr: float,
    radius_arcsec: float = 120.0,
) -> CatalogVerdict:
    """Cross-check a detection against NASA's confirmed-planet and TOI catalogs."""
    if ra is None or dec is None:
        return CatalogVerdict(
            status=STATUS_NO_POSITION,
            headline="Couldn't determine the star's sky position, so no catalog check.",
            detail="Without coordinates we can't ask NASA whether this star is known. "
                   "The detection above still stands on its own.",
            recommend_save=False,
        )

    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        radius = radius_arcsec * u.arcsec

        known: list[KnownObject] = []

        # Confirmed planets
        try:
            conf = NasaExoplanetArchive.query_region(
                table="pscomppars", coordinates=coord, radius=radius)
            for row in conf:
                known.append(KnownObject(_extract_name(row, "confirmed"),
                                         _extract_period(row), "confirmed"))
        except Exception:
            pass  # table/query hiccup — fall through with whatever we have

        # TESS Objects of Interest (candidates)
        try:
            toi = NasaExoplanetArchive.query_region(
                table="toi", coordinates=coord, radius=radius)
            for row in toi:
                known.append(KnownObject(_extract_name(row, "candidate (TOI)"),
                                         _extract_period(row), "candidate (TOI)"))
        except Exception:
            pass

    except Exception as e:
        return CatalogVerdict(
            status=STATUS_ARCHIVE_DOWN,
            headline="NASA's archive couldn't be reached for the catalog check.",
            detail=f"The detection still stands; only the 'is it already known?' step "
                   f"failed ({e}). Try again in a moment.",
            recommend_save=False,
        )

    return _judge(known, detected_period, snr)


def _judge(known: list[KnownObject], detected_period: float, snr: float) -> CatalogVerdict:
    """Combine catalog matches + signal strength into a triage recommendation."""
    strong = snr >= 7
    plausible = snr >= 3

    if not known:
        if strong:
            return CatalogVerdict(
                status=STATUS_UNCATALOGUED,
                headline="No known planet at this star — and the signal is strong.",
                detail="NASA's catalogs list nothing here, yet a clear repeating dip was "
                       "found. Most such signals turn out to be noise or an eclipsing "
                       "binary on closer look — but this is exactly the kind of lead worth "
                       "escalating to a human vetting community.",
                recommend_save=True,
                known=known,
            )
        return CatalogVerdict(
            status=STATUS_UNCATALOGUED,
            headline="No known planet here, and the signal is weak.",
            detail="Nothing catalogued at this position, and the dip isn't convincing. "
                   "Most likely noise — not worth saving unless you want a record.",
            recommend_save=False,
            known=known,
        )

    # There ARE known objects here. Do any match our detected period?
    matched = [k for k in known if _period_matches(detected_period, k.period)]
    names = ", ".join(k.name for k in known[:4])

    if matched:
        m = matched[0]
        return CatalogVerdict(
            status=STATUS_KNOWN_MATCH,
            headline=f"Recovered a known planet: {m.name} "
                     f"({m.period:.3f} d).",
            detail="Your detected period matches a planet NASA already has on record. "
                   "That's a great sanity check that the tool works — but there's nothing "
                   "new to save here; it's already catalogued.",
            recommend_save=False,
            known=known,
        )

    # Known host, but our period is different — a star can host several planets.
    with_periods = [k for k in known if k.period]
    known_pd = ", ".join(f"{k.name} ({k.period:.2f} d)" for k in with_periods[:4]) or names
    if strong:
        return CatalogVerdict(
            status=STATUS_KNOWN_DIFF,
            headline="Known planet host — but your signal's period is different.",
            detail=f"This star already hosts: {known_pd}. Your detected period "
                   f"({detected_period:.3f} d) doesn't match any of them. Stars often have "
                   f"multiple planets, so this could be an additional signal worth a closer "
                   f"look — or a harmonic/artifact of the known one. A genuinely interesting "
                   f"case to flag.",
            recommend_save=True,
            known=known,
        )
    return CatalogVerdict(
        status=STATUS_KNOWN_DIFF,
        headline="Known planet host; your signal differs but is weak.",
        detail=f"This star hosts: {known_pd}. Your period ({detected_period:.3f} d) doesn't "
               f"match, but the signal is too weak to trust — probably an artifact.",
        recommend_save=False,
        known=known,
    )
