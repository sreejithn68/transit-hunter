"""
detection.py — the scientific core of Transit Hunter.

Given a light curve (time + brightness), find transit signals. This module has NO
web or plotting code on purpose: it's pure, testable logic used by both the web
app and the batch sector pipeline.

Pipeline in one breath:
  1. prepare  — bin very fast cadences, remove slow stellar variability with a
                running median whose window is measured in DAYS (so it means the
                same thing for Kepler and TESS), and clip upward outliers.
  2. search   — Box Least Squares (or Transit Least Squares) over a period grid
                that is fine enough for the data's time baseline.
  3. refine   — re-detrend with the found transits masked (so detrending can't
                eat the dip), and search again.
  4. iterate  — mask the found signal and search again for additional planets.

Signal strength is judged with two numbers:
  * SNR — transit depth divided by its uncertainty (counts every in-transit point)
  * SDE — how far the periodogram peak stands above the rest of the periodogram
A real signal needs both: a deep-enough dip AND a period that clearly wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

# --- Thresholds (calibrated with tests/test_detection.py noise simulations) ---
SDE_STRONG = 9.0
SNR_STRONG = 8.0
SDE_PLAUSIBLE = 7.0
SNR_PLAUSIBLE = 7.1   # Kepler pipeline convention

DEFAULT_WINDOW_DAYS = 1.0
MAX_PERIODS = 150_000


@dataclass
class TransitResult:
    """One detected (or best-effort) transit signal."""
    period: float          # orbital period, days
    duration: float        # transit duration, days
    depth: float           # fractional dip (0.01 = 1%)
    depth_err: float       # 1-sigma uncertainty on depth
    t0: float              # mid-time of a reference transit
    snr: float             # depth / depth_err
    sde: float             # signal detection efficiency of the periodogram peak
    n_transits: int        # number of individual transits that have data
    method: str            # "bls" or "tls"
    strength: str          # "strong" | "plausible" | "weak"
    verdict: str           # plain-English summary
    periods: np.ndarray = field(default=None, repr=False)   # periodogram x
    power: np.ndarray = field(default=None, repr=False)     # periodogram y (normalised)
    dominant_event: float | None = None   # set when one glitch event dominates (no real signal found)

    @property
    def depth_ppm(self) -> float:
        return self.depth * 1e6


@dataclass
class SearchOutput:
    """Everything a search produced: the cleaned light curve and the signals."""
    time: np.ndarray        # cleaned, detrended time stamps
    flux: np.ndarray        # cleaned, detrended flux (baseline ~1.0)
    signals: list[TransitResult]

    @property
    def detections(self) -> list[TransitResult]:
        """Signals that cleared at least the 'plausible' bar."""
        return [s for s in self.signals if s.strength != "weak"]

    def isolated(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Light curve with every OTHER signal's transits removed — use this when
        vetting signal i so a second planet can't contaminate its statistics."""
        keep = np.ones(self.time.size, dtype=bool)
        for j, s in enumerate(self.signals):
            if j != i and s.strength != "weak" and not _related(s.period, self.signals[i].period):
                keep &= ~transit_mask(self.time, s.period, s.duration, s.t0, scale=2.0)
        return self.time[keep], self.flux[keep]


def _related(p1: float, p2: float, tol: float = 0.005) -> bool:
    """Same period (or 2x / 0.5x) — e.g. the secondary eclipse of a binary."""
    return any(abs(p1 * f - p2) / p2 < tol for f in (1.0, 2.0, 0.5))


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------

def _finite(time, flux):
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ok = np.isfinite(time) & np.isfinite(flux)
    order = np.argsort(time[ok])
    return time[ok][order], flux[ok][order]


def robust_std(x: np.ndarray) -> float:
    """Standard deviation estimated from the median absolute deviation."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def trim_gap_edges(time, flux, gap_days: float = 0.5, trim_days: float = 0.25):
    """Drop the first `trim_days` after every data gap (and at the start).
    Spacecraft thermal settling after downlinks makes ramps there that look
    like partial transits."""
    time, flux = _finite(time, flux)
    if time.size < 2:
        return time, flux
    starts = np.r_[time[0], time[1:][np.diff(time) > gap_days]]
    seg_start = starts[np.searchsorted(starts, time, side="right") - 1]
    keep = (time - seg_start) >= trim_days
    return time[keep], flux[keep]


def bin_lightcurve(time, flux, bin_minutes: float = 10.0):
    """Average into bins of `bin_minutes` if the cadence is much faster than that.
    Speeds up search ~5–30x on TESS 2-min / 20-s data without losing transits
    (which last hours)."""
    time, flux = _finite(time, flux)
    if time.size < 2:
        return time, flux
    cadence = float(np.median(np.diff(time)))
    width = bin_minutes / 1440.0
    if cadence >= width / 2:
        return time, flux
    idx = np.floor((time - time[0]) / width).astype(np.int64)
    df = pd.DataFrame({"i": idx, "t": time, "f": flux}).groupby("i").mean()
    return df["t"].to_numpy(), df["f"].to_numpy()


def detrend(time, flux, mask=None, window_days: float = DEFAULT_WINDOW_DAYS):
    """Divide out slow variability with a robust *local quadratic* fit in a
    sliding window of `window_days` (a running median would leave curvature
    behind on spotted, rotating stars, which BLS then mistakes for transits).

    Implementation: anchor points every window/8 days; at each anchor fit a
    quadratic to the unmasked points within ±window/2 (two rounds of 3-sigma
    clipping), and evaluate that local fit for the points nearest the anchor.
    Points in `mask` (e.g. known transits) never influence the trend, so the
    dip itself is never flattened away."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    use = np.isfinite(flux)
    if mask is not None:
        use &= ~np.asarray(mask, dtype=bool)
    tu, fu = time[use], flux[use]
    trend = np.full(time.size, np.nan)
    if tu.size < 10:
        return flux / np.nanmedian(flux)

    half = window_days / 2
    step = window_days / 8
    anchors = np.arange(time.min(), time.max() + step, step)
    # Every point belongs to its nearest anchor.
    owner = np.clip(np.round((time - time.min()) / step).astype(int), 0, anchors.size - 1)
    lo_idx = np.searchsorted(tu, anchors - half)
    hi_idx = np.searchsorted(tu, anchors + half, side="right")
    pts_lo = np.searchsorted(owner, np.arange(anchors.size))
    pts_hi = np.searchsorted(owner, np.arange(anchors.size), side="right")

    for k, a in enumerate(anchors):
        if pts_hi[k] == pts_lo[k]:
            continue                                   # no points own this anchor
        x = tu[lo_idx[k]:hi_idx[k]] - a
        y = fu[lo_idx[k]:hi_idx[k]]
        if x.size < 10:
            continue
        good = np.ones(x.size, dtype=bool)
        deg = 2 if (x.max() - x.min()) > 0.5 * window_days else 1
        for _ in range(3):
            c = np.polyfit(x[good], y[good], deg)
            r = y - np.polyval(c, x)
            s = 1.4826 * np.median(np.abs(r[good])) or np.std(r[good])
            new = np.abs(r) < 3 * s if s > 0 else good
            if new.sum() < 10 or (new == good).all():
                break
            good = new
        sl = slice(pts_lo[k], pts_hi[k])
        trend[sl] = np.polyval(c, time[sl] - a)

    bad = ~np.isfinite(trend)
    if bad.all():
        return flux / np.nanmedian(flux)
    if bad.any():
        trend[bad] = np.interp(time[bad], time[~bad], trend[~bad])
    return flux / trend


def prepare(time, flux, mask=None, window_days: float = DEFAULT_WINDOW_DAYS,
            bin_minutes: float = 10.0, already_binned: bool = False):
    """Bin → detrend → clip upward outliers. Returns (time, flux, keep) where
    `keep` marks points that survived clipping (same length as the inputs)."""
    if not already_binned:
        time, flux = bin_lightcurve(time, flux, bin_minutes)
    flat = detrend(time, flux, mask=mask, window_days=window_days)
    sigma = robust_std(flat)
    keep = np.isfinite(flat)
    if sigma > 0:
        keep &= flat < 1.0 + 4.0 * sigma          # flares / cosmic rays
    return time, flat, keep


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------

def min_duration_for(period):
    """Shortest plausible transit duration at a given period (days). Roughly a
    small-star, grazing-ish transit; keeps the grid fine enough for M dwarfs."""
    return np.maximum(0.025, 0.035 * np.asarray(period, dtype=float) ** (1 / 3))


def period_grid(time, min_period: float = 0.5, max_period: float | None = None,
                oversample: float = 3.0, max_periods: int = MAX_PERIODS) -> np.ndarray:
    """Periods spaced so that, over the whole baseline, adjacent trial periods
    drift by less than (shortest plausible duration / oversample). A grid this
    fine never steps over a real transit, however long the baseline."""
    time = np.asarray(time, dtype=float)
    baseline = float(time.max() - time.min())
    longest = baseline / 2.0                       # need >= 2 transits
    max_period = longest if max_period is None else min(max_period, longest)
    if max_period <= min_period:
        raise ValueError(
            f"Data span only {baseline:.1f} days, too short for periods above "
            f"{min_period:.2f} d (need at least two transits).")
    periods = [min_period]
    p = min_period
    while p < max_period:
        p *= 1.0 + float(min_duration_for(p)) / (oversample * baseline)
        periods.append(p)
    periods = np.asarray(periods)
    periods = periods[periods <= max_period]
    if periods.size > max_periods:                 # keep runtime bounded
        periods = np.geomspace(min_period, max_period, max_periods)
    return periods


def duration_grid(min_period: float, n: int = 10) -> np.ndarray:
    """Trial durations for periods >= min_period: from the shortest plausible
    transit at that period up to 0.6 d (14 h)."""
    lo = float(min_duration_for(min_period))
    d = np.geomspace(lo, 0.6, n)
    d = d[d < 0.25 * min_period]
    return d if d.size else np.array([0.2 * min_period])


# --------------------------------------------------------------------------
# Single-signal searches
# --------------------------------------------------------------------------

def _sde(power: np.ndarray) -> tuple[float, np.ndarray]:
    """Detrend the periodogram (remove its slow rise with period) and return the
    SDE of the peak plus the normalised periodogram for plotting."""
    power = np.asarray(power, dtype=float)
    n = power.size
    win = max(11, (n // 100) | 1)
    base = pd.Series(power).rolling(win, center=True, min_periods=1).median().to_numpy()
    resid = power - base
    std = float(np.std(resid))
    if std <= 0:
        return 0.0, np.zeros_like(power)
    norm = (resid - np.mean(resid)) / std
    return float(norm.max()), norm


def classify(sde: float, snr: float, n_transits: int) -> str:
    if n_transits >= 3 and sde >= SDE_STRONG and snr >= SNR_STRONG:
        return "strong"
    if n_transits >= 2 and sde >= SDE_PLAUSIBLE and snr >= SNR_PLAUSIBLE:
        return "plausible"
    return "weak"


def describe(r: TransitResult) -> str:
    if r.strength == "weak":
        return ("No convincing transit found — the best candidate dip doesn't stand "
                "out from the noise. Try more sectors or a different period range.")
    word = "a strong" if r.strength == "strong" else "a plausible"
    return (f"Detected {word} transit-like signal: the star dims by "
            f"{r.depth * 100:.3f}% every {r.period:.4f} days for about "
            f"{r.duration * 24:.1f} hours, seen in {r.n_transits} transits "
            f"(SNR {r.snr:.1f}, SDE {r.sde:.1f}).")


def red_noise_factor(time, flux, duration: float) -> float:
    """beta >= 1: how much worse the scatter is on the transit's own timescale
    than white noise predicts (Pont et al. 2006). Real light curves have
    correlated noise; dividing SNR by beta keeps wiggles from posing as dips."""
    sigma = robust_std(flux)
    if sigma <= 0 or duration <= 0:
        return 1.0
    idx = np.floor((time - time[0]) / duration).astype(np.int64)
    g = pd.DataFrame({"i": idx, "f": flux}).groupby("i")["f"].agg(["mean", "count"])
    g = g[g["count"] >= 3]
    if len(g) < 10:
        return 1.0
    expected = sigma / np.sqrt(g["count"].mean())
    return float(max(1.0, robust_std(g["mean"].to_numpy()) / expected))


PERIOD_BANDS = (1.0, 3.0, 10.0, 30.0)   # band edges (days) for the duration grid


class _Periodogram:
    """Concatenated BLS results from several period bands."""
    def __init__(self, parts):
        for key in ("period", "power", "duration", "transit_time", "depth", "depth_err"):
            setattr(self, key, np.concatenate([np.asarray(getattr(p, key)) for p in parts]))


def _banded_power(model, periods):
    """BLS needs every trial duration shorter than every trial period, so one
    grid can't cover both 0.5-day orbits and 7-hour transits. Search each
    period band with the durations that make sense there, then merge."""
    edges = [periods.min()] + [e for e in PERIOD_BANDS if periods.min() < e < periods.max()] \
        + [periods.max() * (1 + 1e-9)]
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = periods[(periods >= lo) & (periods < hi)]
        if band.size:
            parts.append(model.power(band, duration_grid(float(band.min())),
                                     objective="likelihood", oversample=5))
    return _Periodogram(parts)


def single_event_fraction(per_transit_ll) -> float:
    """Share of the total signal contributed by the single strongest transit."""
    ll = np.clip(np.asarray(per_transit_ll, dtype=float), 0, None)
    ll = ll[np.isfinite(ll)]
    tot = ll.sum()
    return float(ll.max() / tot) if tot > 0 else 1.0


def _pick_peak(model, res, norm, max_tries: int = 25) -> tuple[int, int]:
    """Walk down the periodogram peaks and return the first one that is a
    genuinely *repeating* signal: >= 2 transits with data, and not dominated by
    one event (a single glitch or flare-gap can outscore a real small planet).
    Returns (index, n_transits)."""
    order = np.argsort(norm)[::-1]
    tried: list[float] = []
    fallback = None
    for idx in order:
        p = float(res.period[idx])
        if any(abs(p - q) / q < 0.01 for q in tried):
            continue
        tried.append(p)
        st = model.compute_stats(p, float(res.duration[idx]), float(res.transit_time[idx]))
        counts = np.asarray(st["per_transit_count"])
        n = int(np.sum(counts > 0))
        ll = np.asarray(st["per_transit_log_likelihood"], dtype=float)
        ll = np.where(counts > 0, ll, -np.inf)
        frac = single_event_fraction(ll[counts > 0])
        if fallback is None:
            dominant = float(np.asarray(st["transit_times"])[int(np.argmax(ll))])
            fallback = (int(idx), n, dominant)
        if n >= 2 and frac <= max(0.5, 1.5 / n):
            return int(idx), n, None
        if len(tried) >= max_tries:
            break
    return fallback


def _bls(time, flux, min_period, max_period, oversample) -> TransitResult:
    sigma = robust_std(flux) or 1e-4
    dy = np.full(time.size, sigma)
    model = BoxLeastSquares(time, flux, dy)
    periods = period_grid(time, min_period, max_period, oversample)
    res = _banded_power(model, periods)
    sde, norm = _sde(res.power)
    best, n_tr, dominant = _pick_peak(model, res, norm)
    sde = float(norm[best])
    period = float(res.period[best])
    duration = float(res.duration[best])
    t0 = float(res.transit_time[best])
    depth = float(res.depth[best])
    depth_err = float(res.depth_err[best]) * red_noise_factor(time, flux, duration)
    snr = depth / depth_err if depth_err > 0 else 0.0
    strength = classify(sde, snr, n_tr)
    r = TransitResult(period, duration, depth, depth_err, t0, snr, sde, n_tr,
                      "bls", strength, "", np.asarray(res.period), norm, dominant)
    r.verdict = describe(r)
    return r


def tls_available() -> bool:
    try:
        import transitleastsquares  # noqa: F401
        return True
    except Exception:
        return False


def _refine_bls(time, flux, r: TransitResult) -> TransitResult:
    """Cheap re-fit of a found signal after re-detrending: search only a narrow
    window around its period/duration. Keeps the original periodogram and SDE."""
    sigma = robust_std(flux) or 1e-4
    model = BoxLeastSquares(time, flux, np.full(time.size, sigma))
    periods = np.linspace(r.period * 0.997, r.period * 1.003, 301)
    durations = np.geomspace(r.duration / 2, r.duration * 2, 13)
    durations = durations[durations < 0.5 * periods.min()]
    res = model.power(periods, durations, objective="likelihood")
    i = int(np.argmax(res.power))
    period, duration, t0 = float(res.period[i]), float(res.duration[i]), float(res.transit_time[i])
    st = model.compute_stats(period, duration, t0)
    n_tr = int(np.sum(np.asarray(st["per_transit_count"]) > 0))
    depth = float(res.depth[i])
    depth_err = float(res.depth_err[i]) * red_noise_factor(time, flux, duration)
    snr = depth / depth_err if depth_err > 0 else 0.0
    out = TransitResult(period, duration, depth, depth_err, t0, snr, r.sde, n_tr, "bls",
                        classify(r.sde, snr, n_tr), "", r.periods, r.power)
    out.verdict = describe(out)
    return out


def auto_bin_minutes(baseline_days: float) -> float:
    """Coarser bins for long baselines keep search time bounded; transits
    (>= ~40 min) still span at least one bin."""
    if baseline_days <= 60:
        return 10.0
    if baseline_days <= 200:
        return 20.0
    return 30.0


def _tls(time, flux, min_period, max_period, oversample) -> TransitResult:
    try:
        from transitleastsquares import transitleastsquares
    except Exception as e:
        raise RuntimeError("Deep search (TLS) is unavailable: the "
                           "'transitleastsquares' package isn't installed.") from e
    baseline = float(time.max() - time.min())
    max_period = baseline / 2 if max_period is None else min(max_period, baseline / 2)
    sigma = robust_std(flux) or 1e-4
    model = transitleastsquares(time, flux, np.full(time.size, sigma))
    res = model.power(period_min=min_period, period_max=max_period,
                      oversampling_factor=oversample, show_progress_bar=False,
                      use_threads=4)
    depth = float(1.0 - res.depth) if np.isfinite(res.depth) else 0.0
    n_in = max(int(np.sum(transit_mask(time, res.period, res.duration, res.T0))), 1)
    depth_err = sigma / np.sqrt(n_in) * red_noise_factor(time, flux, float(res.duration))
    snr = depth / depth_err if depth_err > 0 else 0.0
    n_tr = int(res.distinct_transit_count) if np.isfinite(res.distinct_transit_count) else 0
    sde = float(res.SDE) if np.isfinite(res.SDE) else 0.0
    strength = classify(sde, snr, n_tr)
    r = TransitResult(float(res.period), float(res.duration), depth, depth_err,
                      float(res.T0), snr, sde, n_tr, "tls", strength, "",
                      np.asarray(res.periods), np.asarray(res.power))
    r.verdict = describe(r)
    return r


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def transit_mask(time, period, duration, t0, scale: float = 1.0) -> np.ndarray:
    """True for points within (scale * duration) centred on each transit."""
    time = np.asarray(time, dtype=float)
    half = 0.5 * scale * duration
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    return np.abs(phase) < half


def search(time, flux, min_period: float = 0.5, max_period: float | None = None,
           method: str = "bls", max_planets: int = 3,
           window_days: float = DEFAULT_WINDOW_DAYS, oversample: float = 3.0,
           bin_minutes: float | None = None) -> SearchOutput:
    """Find up to `max_planets` transit signals.

    Each round: detrend with all previously-found transits masked, search the
    remaining points, then re-detrend with the new signal masked too and search
    once more (refinement). Stops at the first round whose best signal is weak.
    The first signal is always returned (even if weak) so the UI can show it.
    """
    if method not in ("bls", "tls"):
        raise ValueError("method must be 'bls' or 'tls'")
    run = _bls if method == "bls" else _tls

    t_raw, f_raw = trim_gap_edges(time, flux)
    if t_raw.size < 50:
        raise ValueError("Not enough valid data points to search for a transit.")
    if bin_minutes is None:
        bin_minutes = auto_bin_minutes(float(t_raw.max() - t_raw.min()))
    t_raw, f_raw = bin_lightcurve(t_raw, f_raw, bin_minutes)
    if t_raw.size < 50:
        raise ValueError("Not enough valid data points to search for a transit.")

    signals: list[TransitResult] = []
    found_mask = np.zeros(t_raw.size, dtype=bool)
    t_flat, f_flat, keep = prepare(t_raw, f_raw, window_days=window_days, already_binned=True)

    glitch = np.zeros(t_raw.size, dtype=bool)

    def run_clean():
        """Search; if one isolated event dominates, cut it out and search again."""
        nonlocal glitch
        for _ in range(6):
            sel = keep & ~found_mask & ~glitch
            r = run(t_flat[sel], f_flat[sel], min_period, max_period, oversample)
            if r.dominant_event is None:
                return r
            glitch |= np.abs(t_raw - r.dominant_event) < 1.5 * max(r.duration, 0.1)
        return r

    for _ in range(max_planets):
        if (keep & ~found_mask & ~glitch).sum() < 50:
            break
        r = run_clean()

        # Refinement: protect this signal's transits from the detrender, search again.
        m = transit_mask(t_raw, r.period, r.duration, r.t0, scale=2.0)
        t_flat, f_flat, keep = prepare(t_raw, f_raw, mask=found_mask | m,
                                       window_days=window_days, already_binned=True)
        if method == "bls" and r.n_transits >= 2:
            sel = keep & ~found_mask & ~glitch
            r = _refine_bls(t_flat[sel], f_flat[sel], r)

        if r.strength == "weak":
            if not signals:
                signals.append(r)
            break
        signals.append(r)
        if r.strength != "strong":
            break   # only keep hunting for more planets behind strong signals
        found_mask |= transit_mask(t_raw, r.period, r.duration, r.t0, scale=2.0)
        t_flat, f_flat, keep = prepare(t_raw, f_raw, mask=found_mask,
                                       window_days=window_days, already_binned=True)

    keep &= ~glitch
    return SearchOutput(t_flat[keep], f_flat[keep], signals)


def find_transit(time, flux, min_period: float = 0.5, max_period: float | None = None,
                 **kw) -> TransitResult:
    """Back-compatible helper: the single strongest signal."""
    return search(time, flux, min_period, max_period, max_planets=1, **kw).signals[0]


def fold(time, flux, period: float, t0: float):
    """Fold on a period so every transit stacks at phase 0. Returns
    (phase, flux) sorted by phase, phase in [-0.5, 0.5)."""
    phase = ((np.asarray(time) - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], np.asarray(flux)[order]


# --- Self-test: inject a known planet, confirm we recover it. ---
if __name__ == "__main__":
    from synthetic import make_lightcurve

    t, f = make_lightcurve(baseline=27, period=3.5, depth=0.002, duration=0.12,
                           noise=0.001, seed=42)
    out = search(t, f)
    s = out.signals[0]
    print(f"Injected period 3.5 d   → recovered {s.period:.4f} d")
    print(f"Injected depth 2000 ppm → recovered {s.depth_ppm:.0f} ± {s.depth_err*1e6:.0f} ppm")
    print(f"SNR {s.snr:.1f}  SDE {s.sde:.1f}  transits {s.n_transits}  → {s.strength}")
    assert abs(s.period - 3.5) < 0.01 and s.strength == "strong"
    print("\n✅ Core detection works: recovered the injected planet.")
