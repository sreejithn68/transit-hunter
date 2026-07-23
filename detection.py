"""
detection.py — the scientific core of the exoplanet finder.

Given a light curve (time + brightness), find the most likely transit signal
using the Box Least Squares (BLS) algorithm. This module has NO web or plotting
code on purpose: it's pure, testable logic you can reason about and unit-test.

The idea in one breath: a transiting planet carves periodic, box-shaped dips into
a star's brightness. BLS slides a "box" of many trial periods/durations across the
data and reports the period whose folded dip is most convincing.
"""

from dataclasses import dataclass
import numpy as np
from astropy.timeseries import BoxLeastSquares


@dataclass
class TransitResult:
    """What we found. All the numbers the UI needs to tell the story."""
    period: float          # best-fit orbital period, in days
    duration: float        # transit duration, in days
    depth: float           # fractional dip in brightness (0.01 = 1%)
    depth_ppm: float       # same depth in parts-per-million (nicer to read)
    t0: float              # time of a transit centre (reference epoch)
    power: float           # BLS signal strength at the best period
    snr: float             # signal-to-noise-like ratio (depth / scatter)
    verdict: str           # plain-English summary


def find_transit(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float = 0.5,
    max_period: float = 20.0,
    durations: np.ndarray | None = None,
) -> TransitResult:
    """
    Run BLS over a light curve and return the strongest transit candidate.

    Parameters
    ----------
    time : array of observation times (days)
    flux : array of brightness, normalised so the baseline is ~1.0
    min_period, max_period : range of orbital periods to search (days)
    durations : trial transit durations (days). Sensible default provided.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    # Drop NaNs — real telescope data is full of gaps.
    ok = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[ok], flux[ok]
    if time.size < 50:
        raise ValueError("Not enough valid data points to search for a transit.")

    if durations is None:
        durations = np.array([0.05, 0.1, 0.15, 0.2, 0.3])  # days

    model = BoxLeastSquares(time, flux)
    periods = np.linspace(min_period, max_period, 5000)
    result = model.power(periods, durations)

    best = int(np.argmax(result.power))
    period = float(result.period[best])
    duration = float(result.duration[best])
    depth = float(result.depth[best])
    t0 = float(result.transit_time[best])
    power = float(result.power[best])

    # A rough signal-to-noise: how deep the dip is vs the out-of-transit scatter.
    scatter = float(np.nanstd(flux))
    snr = float(depth / scatter) if scatter > 0 else 0.0

    verdict = _describe(period, depth, snr)

    return TransitResult(
        period=period,
        duration=duration,
        depth=depth,
        depth_ppm=depth * 1e6,
        t0=t0,
        power=power,
        snr=snr,
        verdict=verdict,
    )


def fold(time: np.ndarray, flux: np.ndarray, period: float, t0: float):
    """Fold the light curve on a period so all transits stack at phase 0.
    Returns (phase, flux) sorted by phase — ready to plot."""
    phase = ((np.asarray(time) - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], np.asarray(flux)[order]


def _describe(period: float, depth: float, snr: float) -> str:
    """Turn numbers into a sentence a human can judge."""
    if snr < 3:
        return ("No convincing transit found — the strongest dip is weak and "
                "likely noise. Try a cleaner target or a wider period range.")
    strength = "a strong" if snr >= 7 else "a plausible"
    return (f"Detected {strength} transit-like signal: the star's brightness dips by "
            f"{depth * 100:.2f}% every {period:.2f} days, over and over "
            f"(signal strength ≈ {snr:.1f}× the noise). "
            f"That's consistent with a candidate planet — worth a closer look.")


# --- Self-test: inject a known planet, confirm we recover its period. ---
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    true_period = 3.5      # days
    true_depth = 0.01      # 1% dip
    true_dur = 0.15        # days

    t = np.linspace(0, 27, 4000)                 # ~1 TESS sector of cadence
    f = np.ones_like(t) + rng.normal(0, 0.001, t.size)  # flat star + noise
    # carve in the transits
    phase = (t + 0.5 * true_period) % true_period - 0.5 * true_period
    f[np.abs(phase) < true_dur / 2] -= true_depth

    res = find_transit(t, f)
    print(f"Injected period: {true_period} d   Recovered: {res.period:.3f} d")
    print(f"Injected depth:  {true_depth*100:.2f}%  Recovered: {res.depth*100:.2f}%")
    print(f"SNR: {res.snr:.1f}")
    print("Verdict:", res.verdict)

    assert abs(res.period - true_period) < 0.1, "Failed to recover the period!"
    print("\n✅ Core detection works: recovered the injected planet.")
