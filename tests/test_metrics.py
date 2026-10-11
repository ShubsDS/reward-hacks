import pytest

from rhinoc.metrics import (
    bootstrap_diff_ci,
    gate_passes,
    hack_onset_step,
    matched_pairs,
    observation_gap,
    onset_checkpoint,
    rate,
    retained_fraction,
    reward_conditionality,
)


def test_reward_conditionality_and_observation_gap():
    assert reward_conditionality([1, 1, 1, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert observation_gap([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.25)


def test_rate_rejects_empty():
    with pytest.raises(ValueError):
        rate([])


def test_hack_onset_uses_trailing_window():
    rates = {s: r for s, r in enumerate([0, 0.1, 0.4, 0.6, 0.6, 0.7, 0.9, 0.9])}
    # trailing 4-step means: step3 .275, step4 .425, step5 .575 -> onset at step 5
    assert hack_onset_step(rates) == 5
    assert onset_checkpoint(5, [8, 16, 24]) == 8
    assert hack_onset_step({0: 0.1, 1: 0.2, 2: 0.1, 3: 0.2}) is None
    assert onset_checkpoint(None, [8]) is None


def test_matched_pairs_respects_tolerance_and_no_reuse():
    a = {8: 0.20, 16: 0.55, 24: 0.80}
    b = {8: 0.10, 16: 0.25, 24: 0.50}
    pairs = matched_pairs(a, b)
    assert [(sa, sb) for sa, sb, _ in pairs] == [(8, 16), (16, 24)]
    assert all(gap <= 0.10 for *_, gap in pairs)


def test_bootstrap_ci_brackets_point():
    treat, control = [1] * 30 + [0] * 70, [1] * 5 + [0] * 95
    point, lo, hi = bootstrap_diff_ci(treat, control)
    assert point == pytest.approx(0.25)
    assert lo < point < hi
    assert lo > 0


def test_gate_rule():
    assert gate_passes([1] * 30 + [0] * 170, [1] * 5 + [0] * 195)["passed"]
    # large ratio but under +3 points fails
    assert not gate_passes([1] * 4 + [0] * 196, [1] * 1 + [0] * 199)["passed"]
    # +3 points but under 2x fails
    assert not gate_passes([1] * 26 + [0] * 174, [1] * 20 + [0] * 180)["passed"]


def test_retained_fraction():
    assert retained_fraction(0.14, 0.20) == pytest.approx(0.7)
    assert retained_fraction(0.1, 0.0) is None
