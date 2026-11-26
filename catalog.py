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
    confirmed-planet and TOI tables use. Returns None if absent.

    NASA returns periods as astropy Quantities *with units* (days), and some rows
    are masked/empty — both broke a naive float() call, so we unwrap and guard."""
    for col in ("pl_orbper", "toi_period", "period"):
        try:
            val = row[col]
        except (KeyError, TypeError, IndexError):
            continue
        # Unwrap an astropy Quantity (value + unit) down to a plain number.
        val = getattr(val, "value", val)
        # Skip masked / missing entries.
        try:
            import numpy as _np
            if val is None or _np.ma.is_masked(val):
                continue
        except Exception:
            if val is None:
                continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if f == f and f > 0:  # not NaN, positive
            return f
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


class ArchiveError(RuntimeError):
    """NASA's archive couldn't be queried."""


def query_known(ra: float, dec: float, radius_arcsec: float = 120.0) -> list[KnownObject]:
    """Every confirmed planet and TOI catalogued within `radius_arcsec` of the
    star. One call per star — reuse the list for every detected signal.
    Raises ArchiveError if NASA's archive can't be reached at all."""
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    except Exception as e:  # pragma: no cover - dependency problem
        raise ArchiveError(str(e)) from e

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    radius = radius_arcsec * u.arcsec
    known: list[KnownObject] = []
    failures = []
    for table, kind in (("pscomppars", "confirmed"), ("toi", "candidate (TOI)")):
        try:
            for row in NasaExoplanetArchive.query_region(
                    table=table, coordinates=coord, radius=radius):
                known.append(KnownObject(_extract_name(row, kind), _extract_period(row), kind))
        except Exception as e:
            failures.append(e)
    if len(failures) == 2:
        raise ArchiveError(str(failures[0]))
    return known


def _status_only(status, headline, detail) -> CatalogVerdict:
    return CatalogVerdict(status=status, headline=headline, detail=detail, recommend_save=False)


def crosscheck(ra: float | None, dec: float | None, detected_period: float,
               strength: str, vet_verdict: str | None = None,
               radius_arcsec: float = 120.0) -> CatalogVerdict:
    """Convenience wrapper: query the archive and judge a single signal."""
    if ra is None or dec is None:
        return no_position_verdict()
    try:
        known = query_known(ra, dec, radius_arcsec)
    except ArchiveError as e:
        return archive_down_verdict(e)
    return judge(known, detected_period, strength, vet_verdict)


def no_position_verdict() -> CatalogVerdict:
    return _status_only(
        STATUS_NO_POSITION,
        "Couldn't determine the star's sky position, so no catalog check.",
        "Without coordinates we can't ask NASA whether this star is known. "
        "The detection still stands on its own.")


def archive_down_verdict(e) -> CatalogVerdict:
    return _status_only(
        STATUS_ARCHIVE_DOWN,
        "NASA's archive couldn't be reached for the catalog check.",
        f"The detection still stands; only the 'is it already known?' step failed "
        f"({e}). Try again in a moment.")


def judge(known: list[KnownObject], detected_period: float, strength: str,
          vet_verdict: str | None = None) -> CatalogVerdict:
    """Combine catalog matches, signal strength ('strong' | 'plausible' | 'weak')
    and the vetting verdict into a triage recommendation."""
    strong = strength == "strong"
    false_pos = vet_verdict == "likely false positive"
    fp_note = (" However, the vetting checks below flag it as a likely false "
               "positive (e.g. an eclipsing binary), so it isn't worth escalating."
               if false_pos else "")

    if not known:
        if strong:
            return CatalogVerdict(
                status=STATUS_UNCATALOGUED,
                headline="No known planet or TOI at this star — and the signal is strong.",
                detail="NASA's catalogs list nothing here, yet a clear repeating dip was "
                       "found. Most such signals turn out to be eclipsing binaries or "
                       "artifacts on closer look — but this is exactly the kind of lead "
                       "worth a careful second look." + fp_note,
                recommend_save=not false_pos, known=known)
        return CatalogVerdict(
            status=STATUS_UNCATALOGUED,
            headline="No known planet here, and the signal isn't strong enough to trust.",
            detail="Nothing catalogued at this position, and the dip doesn't clear the "
                   "'strong' bar. Probably noise — keep it only if you want a record.",
            recommend_save=False, known=known)

    matched = [k for k in known if _period_matches(detected_period, k.period)]
    if matched:
        m = matched[0]
        return CatalogVerdict(
            status=STATUS_KNOWN_MATCH,
            headline=f"Recovered a known object: {m.name} ({m.period:.3f} d, {m.kind}).",
            detail="Your detected period matches one NASA already has on record — a "
                   "good sanity check that the tool works, but nothing new to save.",
            recommend_save=False, known=known)

    with_periods = [k for k in known if k.period]
    known_pd = ", ".join(f"{k.name} ({k.period:.2f} d)" for k in with_periods[:4]) \
        or ", ".join(k.name for k in known[:4])
    if strong:
        return CatalogVerdict(
            status=STATUS_KNOWN_DIFF,
            headline="Known planet host — but your signal's period is different.",
            detail=f"This star already hosts: {known_pd}. Your detected period "
                   f"({detected_period:.3f} d) doesn't match any of them. Stars often "
                   f"have several planets, so this could be an additional one — or an "
                   f"artifact of the known signal." + fp_note,
            recommend_save=not false_pos, known=known)
    return CatalogVerdict(
        status=STATUS_KNOWN_DIFF,
        headline="Known planet host; your signal differs but isn't strong.",
        detail=f"This star hosts: {known_pd}. Your period ({detected_period:.3f} d) "
               f"doesn't match, but the signal is too weak to trust.",
        recommend_save=False, known=known)
