"""Vetting must pass planets and catch the classic impostors."""
from detection import search
from synthetic import make_lightcurve
from vetting import FAIL, VERDICT_FP, VERDICT_PLANET, WARN, Star, vet

SUN = Star(radius=1.0, mass=1.0)


def run(star=SUN, **kw):
    t, f = make_lightcurve(baseline=27, seed=kw.pop("seed", 9), **kw)
    out = search(t, f, max_planets=2)
    s = out.signals[0]
    return vet(*out.isolated(0), s, star, others=out.signals)


def status(rep, name):
    return next(c.status for c in rep.checks if c.name == name)


def test_planet_passes():
    rep = run(period=3.5, depth=2e-3, duration=0.12)
    assert rep.verdict == VERDICT_PLANET
    assert 4 < rep.radius_earth < 6          # sqrt(2000 ppm) * 109 ≈ 4.9 Earth radii


def test_odd_even_binary_fails():
    rep = run(period=2.2, depth=8e-3, duration=0.12, odd_even_ratio=0.5)
    assert status(rep, "Odd/even depths") == FAIL
    assert rep.verdict == VERDICT_FP


def test_secondary_eclipse_fails():
    rep = run(period=4.1, depth=8e-3, duration=0.15, secondary_depth=3e-3)
    assert status(rep, "Secondary eclipse") == FAIL
    assert rep.verdict == VERDICT_FP


def test_v_shape_warns():
    rep = run(period=3.0, depth=5e-3, duration=0.15, ingress_frac=0.5)
    assert status(rep, "Dip shape (U vs V)") == WARN


def test_too_big_fails():
    rep = run(period=5.0, depth=6e-2, duration=0.15)
    assert status(rep, "Implied size") == FAIL


def test_too_long_for_star_fails():
    # 0.5-day transit at 2 days is impossible around a small M dwarf.
    rep = run(star=Star(radius=0.3, mass=0.3), period=2.0, depth=3e-3, duration=0.5)
    assert status(rep, "Duration vs. star size") == FAIL


def test_contamination_flagged():
    rep = run(star=Star(radius=1.0, mass=1.0, contamination=1.5),
              period=3.5, depth=2e-3, duration=0.12)
    assert status(rep, "Light from neighbours") == FAIL
