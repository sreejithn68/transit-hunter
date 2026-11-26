"""
vetting.py — is this transit-like signal actually a planet?

Most transit-like signals are NOT planets. The usual impostors:
  * eclipsing binaries (two stars eclipsing each other) — give away by unequal
    alternating dips, a secondary eclipse half an orbit later, V-shaped dips, or
    an implied "planet" bigger than any planet can be;
  * stellar variability (spots, pulsation) — a smooth wave fits better than a dip;
  * blends — light from a neighbouring star dilutes or fakes the signal.

Each check returns pass / warn / fail / n/a with a plain-English explanation.
Pure logic: no web or plotting code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.timeseries import BoxLeastSquares

from detection import TransitResult, robust_std

R_JUP_PER_R_SUN = 9.731
R_EARTH_PER_R_SUN = 109.1

PASS, WARN, FAIL, NA = "pass", "warn", "fail", "n/a"

VERDICT_PLANET = "planet-like"
VERDICT_SUSPICIOUS = "suspicious"
VERDICT_FP = "likely false positive"


@dataclass
class Star:
    """What we know about the host star (any field may be missing)."""
    radius: float | None = None        # solar radii
    mass: float | None = None          # solar masses
    teff: float | None = None          # kelvin
    logg: float | None = None          # cgs
    contamination: float | None = None  # TIC contratio: flux from neighbours / target

    def mass_estimate(self) -> float | None:
        if self.mass and self.mass > 0:
            return self.mass
        if self.radius and self.logg is not None:
            return float(10 ** (self.logg - 4.438) * self.radius ** 2)
        return None


@dataclass
class Check:
    name: str
    status: str
    value: str
    explanation: str


@dataclass
class VetReport:
    checks: list[Check]
    verdict: str
    radius_earth: float | None = None          # implied planet radius
    notes: list[str] = field(default_factory=list)

    @property
    def n_fail(self):
        return sum(c.status == FAIL for c in self.checks)

    @property
    def n_warn(self):
        return sum(c.status == WARN for c in self.checks)


def _sig(a, ea, b, eb):
    den = np.hypot(ea, eb)
    return float(abs(a - b) / den) if den > 0 else 0.0


def fit_trapezoid(time, flux, r: TransitResult):
    """Least-squares trapezoid fit to the folded transit. Returns
    (total_duration_days, ingress_fraction, depth) or None. Ingress fraction
    ~0.1–0.3 for planets; 0.5 means a pure V (no flat bottom)."""
    x = (time - r.t0 + 0.5 * r.period) % r.period - 0.5 * r.period
    win = np.abs(x) < 2.0 * r.duration
    x, y = x[win], 1.0 - flux[win]
    if x.size < 20:
        return None
    best = None
    for T in r.duration * np.geomspace(0.6, 2.5, 18):
        ax = np.abs(x)
        for fr in np.linspace(0.05, 0.5, 10):
            ing = fr * T
            prof = np.clip((T / 2 - ax) / ing, 0, 1)
            den = np.dot(prof, prof)
            if den <= 0:
                continue
            d = np.dot(prof, y) / den
            chi = np.sum((y - d * prof) ** 2)
            if best is None or chi < best[0]:
                best = (chi, T, fr, d)
    return None if best is None else (best[1], best[2], best[3])


def _same_period(p1, p2, tol=0.005):
    return any(abs(p1 * f - p2) / p2 < tol for f in (1.0, 2.0, 0.5))


def max_duration(period, star: Star, depth: float) -> float | None:
    """Longest possible transit (central, circular orbit) in days."""
    m = star.mass_estimate()
    if not star.radius or not m:
        return None
    a_rsun = (m * (period / 365.25) ** 2) ** (1 / 3) * 215.03
    k = np.sqrt(max(depth, 0))
    x = star.radius * (1 + k) / a_rsun
    if x >= 1:
        return period / 2
    return float(period / np.pi * np.arcsin(x))


def per_transit_depths(time, flux, r: TransitResult):
    """Depth of each individual transit: (epoch numbers, depths)."""
    epoch = np.round((time - r.t0) / r.period)
    dt = time - (r.t0 + epoch * r.period)
    inner = np.abs(dt) < 0.35 * r.duration
    near = (np.abs(dt) > 0.75 * r.duration) & (np.abs(dt) < 2.0 * r.duration)
    ep, dep = [], []
    for e in np.unique(epoch[inner]):
        i, o = inner & (epoch == e), near & (epoch == e)
        if i.sum() >= 1 and o.sum() >= 4:
            ep.append(int(e))
            dep.append(float(np.median(flux[o]) - np.mean(flux[i])))
    return np.array(ep), np.array(dep)


def _odd_even_empirical(time, flux, r: TransitResult):
    ep, dep = per_transit_depths(time, flux, r)
    odd, even = dep[ep % 2 == 1], dep[ep % 2 == 0]
    if odd.size < 3 or even.size < 3:
        return None
    err = np.hypot(np.std(odd, ddof=1) / np.sqrt(odd.size),
                   np.std(even, ddof=1) / np.sqrt(even.size))
    return float(abs(odd.mean() - even.mean()) / err) if err > 0 else None


def _segment_normalise(t, v, gap=0.5):
    """Remove each sector's own zero point (the star sits on different pixels)."""
    v = v.copy()
    cuts = np.concatenate([[0], np.where(np.diff(t) > gap)[0] + 1, [t.size]])
    for a, b in zip(cuts[:-1], cuts[1:]):
        v[a:b] -= np.median(v[a:b])
    return v


def centroid_shift(ctime, col, row, r: TransitResult):
    """Does the star's image move during transit? Returns (shift_pixels, sigma)
    or None. A significant shift means the dip probably comes from a
    neighbouring star in the same pixels (a blended eclipsing binary)."""
    from detection import bin_lightcurve, detrend, transit_mask
    ctime = np.asarray(ctime, dtype=float)
    shifts, errs = [], []
    for c in (col, row):
        c = np.asarray(c, dtype=float)
        ok = np.isfinite(ctime) & np.isfinite(c)
        if ok.sum() < 100:
            return None
        order = np.argsort(ctime[ok])
        t, v = bin_lightcurve(ctime[ok][order], c[ok][order], 10.0)
        v = _segment_normalise(t, v) + 100.0   # detrend() divides, so keep it positive
        base = 100.0
        intr = transit_mask(t, r.period, r.duration, r.t0)
        near = transit_mask(t, r.period, r.duration, r.t0, scale=4.0) & ~transit_mask(
            t, r.period, r.duration, r.t0, scale=1.5)
        resid = (detrend(t, v, mask=transit_mask(t, r.period, r.duration, r.t0, scale=2.0)) - 1) * base
        if intr.sum() < 5 or near.sum() < 10:
            return None
        d = resid[intr].mean() - resid[near].mean()
        sig = robust_std(resid[near])
        e = sig * np.sqrt(1 / intr.sum() + 1 / near.sum())
        shifts.append(d)
        errs.append(e)
    shift = float(np.hypot(*shifts))
    err = float(np.hypot(*errs) / np.sqrt(2))
    return shift, (shift / err if err > 0 else 0.0)


def vet(time, flux, r: TransitResult, star: Star | None = None,
        others: list[TransitResult] | None = None, centroids=None) -> VetReport:
    """Run every check on one signal. Pass the light curve with OTHER signals
    removed (see SearchOutput.isolated) so they can't contaminate the stats."""
    star = star or Star()
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    sigma = robust_std(flux) or 1e-4
    model = BoxLeastSquares(time, flux, np.full(time.size, sigma))
    st = model.compute_stats(r.period, r.duration, r.t0)
    checks: list[Check] = []

    # 1. Odd vs even transit depths -------------------------------------------
    (d_odd, e_odd), (d_even, e_even) = st["depth_odd"], st["depth_even"]
    s = _sig(d_odd, e_odd, d_even, e_even)
    # Real transits vary in depth beyond white noise (sector-to-sector dilution
    # corrections, spots, systematics). Use the scatter of individual transit
    # depths when there are enough of them, so that doesn't masquerade as odd/even.
    emp = _odd_even_empirical(time, flux, r)
    if emp is not None:
        s = min(s, emp)
    mean_d = 0.5 * (d_odd + d_even)
    rel = abs(d_odd - d_even) / mean_d if mean_d > 0 else 0.0
    # Needs to be both significant AND sizeable: at very high SNR, a few-percent
    # difference from systematics becomes "significant" without meaning anything.
    status = FAIL if (s > 4 and rel > 0.15) else WARN if (s > 3 and rel > 0.05) else PASS
    checks.append(Check(
        "Odd/even depths", status,
        f"{d_odd*1e6:.0f} vs {d_even*1e6:.0f} ppm ({s:.1f}σ, {rel*100:.0f}% apart)",
        "Alternate dips have different depths — the hallmark of an eclipsing binary "
        "seen at half its true period." if status != PASS else
        "Alternate dips are equally deep, as expected for a planet."))

    # 2. Secondary eclipse half an orbit later ---------------------------------
    d_sec, e_sec = st["depth_phased"]
    s_sec = d_sec / e_sec if e_sec > 0 else 0.0
    ratio = d_sec / r.depth if r.depth > 0 else 0.0
    if s_sec > 3 and ratio > 0.1:
        status, txt = FAIL, ("A second, sizeable dip appears half an orbit after each "
                             "transit — the companion is self-luminous, so likely a star.")
    elif s_sec > 3:
        status, txt = WARN, ("A small dip half an orbit later. Hot giant planets can "
                             "show this, but it is more often a binary star.")
    else:
        status, txt = PASS, "No dip half an orbit later (no secondary eclipse)."
    checks.append(Check("Secondary eclipse", status,
                        f"{d_sec*1e6:.0f} ± {e_sec*1e6:.0f} ppm", txt))

    # 3. Transit vs smooth wave ------------------------------------------------
    dll = float(st["harmonic_delta_log_likelihood"])
    status = FAIL if dll > 0 else PASS
    checks.append(Check(
        "Transit vs. stellar wave", status, f"Δlog L = {dll:.1f}",
        "A smooth sinusoid explains the data better than a dip — likely starspots or "
        "pulsation, not a transit." if status == FAIL else
        "A transit explains the data better than a smooth stellar wave."))

    # 4. Period alias -----------------------------------------------------------
    d_half, e_half = st["depth_half"]
    alias = e_half > 0 and d_half / e_half > 3 and abs(d_half - r.depth) < 2 * np.hypot(e_half, r.depth_err)
    checks.append(Check(
        "Half-period alias", WARN if alias else PASS,
        f"depth at P/2: {d_half*1e6:.0f} ppm",
        f"The same dip shows up at {r.period/2:.4f} d — the true period may be half "
        "the detected one." if alias else
        "No sign that the true period is half the detected one."))

    # 5. Shape ----------------------------------------------------------------
    fit = fit_trapezoid(time, flux, r)
    if fit is None:
        checks.append(Check("Dip shape (U vs V)", NA, "—", "Too few in-transit points to judge shape."))
    else:
        T, fr, _ = fit
        v_like = fr >= 0.4
        checks.append(Check(
            "Dip shape (U vs V)", WARN if v_like else PASS,
            f"ingress = {fr*100:.0f}% of {T*24:.1f} h",
            "V-shaped dip with no flat bottom: typical of a grazing eclipsing binary "
            "(grazing planets do exist)." if v_like else
            "Flat-bottomed (U-shaped) dip, as planets usually make."))

    # 6. Enough transits -------------------------------------------------------
    n = r.n_transits
    checks.append(Check(
        "Number of transits", PASS if n >= 3 else WARN, str(n),
        "Seen often enough to trust the period." if n >= 3 else
        "Only two transits — the period could be a multiple of the true one, and two "
        "coincident noise dips can mimic a signal."))

    # 7. Size and 8. duration (need star properties) ----------------------------
    radius_earth = None
    if star.radius:
        # SPOC/Kepler PDCSAP flux is already corrected for neighbours' light,
        # so the depth is used as-is (no extra dilution correction).
        rp_rsun = np.sqrt(max(r.depth, 0)) * star.radius
        radius_earth = float(rp_rsun * R_EARTH_PER_R_SUN)
        rp_rjup = rp_rsun * R_JUP_PER_R_SUN
        status = FAIL if rp_rjup > 2.0 else WARN if rp_rjup > 1.5 else PASS
        checks.append(Check(
            "Implied size", status, f"{radius_earth:.1f} R⊕ ({rp_rjup:.2f} R♃)",
            "Bigger than any known planet — almost certainly a star or brown dwarf."
            if status == FAIL else
            "Unusually large for a planet; inflated hot Jupiters exist but binaries are more common."
            if status == WARN else "A planet-sized object."))
        tmax = max_duration(r.period, star, r.depth)
        if tmax:
            dur = fit[0] if fit else r.duration
            ratio = dur / tmax
            status = FAIL if ratio > 2.0 else WARN if ratio > 1.4 else PASS
            checks.append(Check(
                "Duration vs. star size", status,
                f"{dur*24:.1f} h (max ≈ {tmax*24:.1f} h)",
                "Far longer than an orbiting planet could take to cross this star — "
                "the star's size may be wrong, or the eclipse is on a different (bigger) star."
                if status != PASS else "Duration is physically possible for this star."))
    else:
        checks.append(Check("Implied size", NA, "—", "Star radius unknown, can't convert depth to size."))
        if r.depth > 0.03:
            checks[-1] = Check("Implied size", WARN, f"depth {r.depth*100:.1f}%",
                               "Very deep dip (>3%) — usually an eclipsing binary unless the "
                               "star is very small.")

    # 9. Another detected signal at the same period = secondary eclipse -------
    from detection import coincident
    twins = [o for o in (others or []) if o is not r and _same_period(o.period, r.period)
             and not coincident(o, r)]
    if twins:
        o = twins[0]
        off = ((o.t0 - r.t0) / r.period) % 1.0
        ratio = min(o.depth, r.depth) / max(o.depth, r.depth, 1e-9)
        if r.depth >= o.depth and ratio < 0.1:
            # A tiny second dip can be a hot planet passing BEHIND its star.
            checks.append(Check(
                "Matching second signal", WARN,
                f"{o.period:.4f} d at phase {off:.2f}, {ratio*100:.0f}% as deep",
                "A much shallower dip on the same period. A hot giant planet hidden "
                "behind its star can cause this; a faint binary companion can too."))
        else:
            checks.append(Check(
                "Matching second signal", FAIL,
                f"{o.period:.4f} d at phase {off:.2f}, {ratio*100:.0f}% as deep",
                "A second dip train repeats on the same period — two eclipses per orbit "
                "means two stars eclipsing each other, not a planet."))

    # 10. Centroid shift (needs pixel-centroid time series) --------------------
    if centroids is not None:
        res = centroid_shift(*centroids, r)
        if res is None:
            checks.append(Check("Image shift during dip", NA, "—", "Not enough centroid data."))
        else:
            shift, sig = res
            # A dip of fractional depth δ on a source d pixels from the photocentre
            # moves the image by ≈ δ·d, so the implied source offset is shift/δ.
            # Deep transits on the target itself still shift the image slightly
            # (neighbours' light in the aperture), so only an offset beyond ~1
            # pixel (21″) — i.e. clearly a different star — counts.
            offset = shift / max(r.depth, 1e-6)
            if offset > 10:   # far outside the aperture: a data glitch, not a blend
                checks.append(Check("Image shift during dip", NA,
                                    f"{shift*1000:.1f} millipixels",
                                    "Centroid data unreliable for this star."))
                offset = None
        if res is not None and offset is not None:
            status = (FAIL if sig > 5 and offset > 1.0 else
                      WARN if sig > 3 and offset > 0.5 else PASS)
            checks.append(Check(
                "Image shift during dip", status,
                f"{shift*1000:.1f} millipixels ({sig:.1f}σ) → source ≈{offset:.2f} px "
                f"(≈{offset*21:.0f}″) from star",
                "The star's image moves during each dip — the dip likely comes from a "
                "neighbouring star blended into the same pixels." if status != PASS else
                "The star's image stays put during the dip — consistent with this star."))

    if star.contamination is not None:
        c = star.contamination
        status = FAIL if c > 1.0 else WARN if c > 0.2 else PASS
        checks.append(Check(
            "Light from neighbours", status, f"{c*100:.0f}% of target flux",
            "Nearby stars contribute a lot of the light — the dip may come from a "
            "neighbouring eclipsing binary, not this star." if status != PASS else
            "Little contaminating light from nearby stars."))

    n_fail = sum(c.status == FAIL for c in checks)
    n_warn = sum(c.status == WARN for c in checks)
    verdict = VERDICT_FP if n_fail else VERDICT_SUSPICIOUS if n_warn >= 2 else VERDICT_PLANET
    return VetReport(checks, verdict, radius_earth)
