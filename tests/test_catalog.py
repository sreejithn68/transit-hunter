"""Catalog triage logic (no network)."""
import catalog
from catalog import KnownObject, judge


def test_period_match_and_aliases():
    assert catalog._period_matches(3.0, 3.01)
    assert catalog._period_matches(6.0, 3.0)
    assert catalog._period_matches(1.5, 3.0)
    assert not catalog._period_matches(9.0, 3.0)     # 3x is NOT treated as an alias
    assert not catalog._period_matches(3.0, None)


def test_known_match():
    v = judge([KnownObject("pi Men c", 6.268, "confirmed")], 6.27, "strong")
    assert v.status == catalog.STATUS_KNOWN_MATCH and not v.recommend_save


def test_known_host_new_period_strong_is_recommended():
    v = judge([KnownObject("X b", 2.0, "confirmed")], 7.3, "strong", "planet-like")
    assert v.status == catalog.STATUS_KNOWN_DIFF and v.recommend_save


def test_uncatalogued_strong_recommended_unless_false_positive():
    assert judge([], 4.0, "strong", "planet-like").recommend_save
    assert not judge([], 4.0, "strong", "likely false positive").recommend_save
    assert not judge([], 4.0, "plausible").recommend_save
