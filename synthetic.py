"""
synthetic.py — fake light curves with known answers, for tests and self-checks.

Transits are trapezoids (planets: short ingress → flat bottom, "U").
Eclipsing binaries can be simulated with alternating depths and/or a
secondary eclipse at phase 0.5.
"""

from __future__ import annotations

import numpy as np


def _trapezoid(phase_days, duration, depth, ingress_frac=0.15):
    """Depth profile for one transit given time-from-centre in days."""
    x = np.abs(phase_days)
    half = duration / 2
    ingress = max(duration * ingress_frac, 1e-6)
    prof = np.zeros_like(x)
    flat = x <= half - ingress
    slope = (x > half - ingress) & (x < half)
    prof[flat] = depth
    prof[slope] = depth * (half - x[slope]) / ingress
    return prof


def inject(t, f, period, depth, duration, t0=0.3, ingress_frac=0.15,
           odd_even_ratio=1.0, secondary_depth=0.0):
    """Subtract a transit train from flux `f` (returns a new array)."""
    t = np.asarray(t, dtype=float)
    f = np.array(f, dtype=float)
    dt = (t - t0 + 0.5 * period) % period - 0.5 * period
    epoch = np.round((t - t0) / period).astype(int)
    d = np.where(epoch % 2 == 0, depth, depth * odd_even_ratio)
    f -= _trapezoid(dt, duration, 1.0, ingress_frac) * d
    if secondary_depth:
        dt2 = (t - t0) % period - 0.5 * period
        f -= _trapezoid(dt2, duration, secondary_depth, ingress_frac)
    return f


def make_lightcurve(baseline=27.0, cadence_min=2.0, noise=1e-3, seed=0,
                    gaps=True, variability=0.0, planets=None, **one_planet):
    """Build a light curve. Pass either a single planet via keyword args
    (period, depth, duration, ...) or a list of dicts via `planets`."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, baseline, cadence_min / 1440.0)
    if gaps:  # TESS-like mid-sector downlink gap every ~13.7 days
        orbit = 13.7
        t = t[((t % orbit) > 0.0) & ((t % orbit) < orbit - 1.0)]
    f = 1.0 + rng.normal(0, noise, t.size)
    if variability:
        f *= 1.0 + variability * np.sin(2 * np.pi * t / 3.1 + 0.4)
    if one_planet:
        planets = [one_planet]
    for p in planets or []:
        f = inject(t, f, **p)
    return t, f
