"""Injection–recovery tests: plant known planets in fake data, check we find them."""
import numpy as np
import pytest

from detection import (bin_lightcurve, detrend, find_transit, period_grid, search,
                       tls_available)
from synthetic import make_lightcurve


def close(p, true, tol=0.005):
    return abs(p - true) / true < tol


def test_single_planet_recovered():
    t, f = make_lightcurve(baseline=27, period=3.5, depth=2e-3, duration=0.12, seed=1)
    s = search(t, f).signals[0]
    assert close(s.period, 3.5)
    assert s.strength == "strong"
    assert s.n_transits >= 6
    assert abs(s.depth - 2e-3) < 4 * s.depth_err + 3e-4


def test_small_planet_recovered():
    t, f = make_lightcurve(baseline=27, period=5.2, depth=6e-4, duration=0.12, seed=2)
    s = search(t, f).signals[0]
    assert close(s.period, 5.2)
    assert s.strength in ("strong", "plausible")


def test_long_baseline_grid_is_fine_enough():
    # 4 sectors: a coarse linear grid would step right over this period.
    t, f = make_lightcurve(baseline=108, period=12.37, depth=5e-4, duration=0.2, seed=3)
    s = search(t, f).signals[0]
    assert close(s.period, 12.37, tol=0.002)
    assert s.strength == "strong"


def test_two_planets_both_recovered():
    t, f = make_lightcurve(baseline=54, seed=4, planets=[
        dict(period=3.1, depth=1.5e-3, duration=0.10, t0=0.4),
        dict(period=11.3, depth=1.2e-3, duration=0.18, t0=2.1),
    ])
    found = [s.period for s in search(t, f, max_planets=3).detections]
    assert any(close(p, 3.1) for p in found)
    assert any(close(p, 11.3) for p in found)


def test_transit_survives_stellar_variability():
    t, f = make_lightcurve(baseline=27, period=4.4, depth=1.5e-3, duration=0.13,
                           variability=0.004, seed=5)
    s = search(t, f).signals[0]
    assert close(s.period, 4.4)
    assert s.strength == "strong"


@pytest.mark.parametrize("seed", range(6))
def test_pure_noise_is_not_strong(seed):
    t, f = make_lightcurve(baseline=27, seed=100 + seed,
                           variability=0.002 if seed % 2 else 0.0)
    out = search(t, f, max_planets=1)
    assert out.signals[0].strength != "strong"


def test_period_grid_caps_at_half_baseline():
    t = np.linspace(0, 27, 1000)
    g = period_grid(t, 0.5, 100)
    assert g.max() <= 13.5 + 1e-9
    assert np.all(np.diff(g) > 0)


def test_binning_reduces_fast_cadence():
    t = np.arange(0, 1, 2 / 1440)
    tb, fb = bin_lightcurve(t, np.ones_like(t), 10)
    assert tb.size == pytest.approx(t.size / 5, rel=0.05)


def test_detrend_with_mask_keeps_transit_depth():
    from detection import transit_mask
    t, f = make_lightcurve(baseline=27, period=5.0, depth=3e-3, duration=0.4,
                           variability=0.003, seed=6, gaps=False)
    m = transit_mask(t, 5.0, 0.4, 0.3, scale=1.5)
    flat = detrend(t, f, mask=m)
    core = transit_mask(t, 5.0, 0.2, 0.3)
    assert 1 - np.mean(flat[core]) == pytest.approx(3e-3, rel=0.15)


def test_find_transit_back_compat():
    t, f = make_lightcurve(baseline=27, period=2.5, depth=2e-3, duration=0.1, seed=7)
    assert close(find_transit(t, f).period, 2.5)


@pytest.mark.skipif(not tls_available(), reason="transitleastsquares not installed")
def test_tls_recovers_planet():
    t, f = make_lightcurve(baseline=27, period=3.5, depth=1.5e-3, duration=0.12, seed=8)
    s = search(t, f, method="tls", max_planets=1).signals[0]
    assert close(s.period, 3.5)
    assert s.method == "tls"


def test_single_glitch_does_not_hide_real_planet():
    # A one-off 5000 ppm dip (instrument glitch) must not outrank a small planet.
    t, f = make_lightcurve(baseline=54, period=2.7, depth=4e-4, duration=0.08, seed=10)
    f[np.abs(t - 20.0) < 0.02] -= 5e-3
    s = search(t, f).signals[0]
    assert close(s.period, 2.7)


def test_long_transit_found_at_long_period():
    # 7-hour transits on a 41-day orbit (needs the long-duration band).
    t, f = make_lightcurve(baseline=200, period=41.3, depth=6e-4, duration=0.3,
                           seed=11, t0=5.0)
    s = search(t, f).signals[0]
    assert close(s.period, 41.3)
    assert s.duration > 0.15
