"""End-to-end checks of the hypothesis tests on synthetic battery records with planted effects."""

import json
import random

import pytest

from rhinoc.analysis import (
    apps_rates,
    e1_validation,
    h3_robust,
    hypothesis_results,
    matched_h3,
    misalignment_results,
)
from rhinoc.runs import ARMS

CHANNELS = ("toy", "t1", "t2")


def _cells(rng, target, condition, rc, base_exploit=0.2, n_scen=40, per=4):
    rows = []
    for ch in CHANNELS:
        for i in range(n_scen):
            for pays in (True, False):
                for _ in range(per):
                    p = base_exploit + (rc if pays else 0)
                    rows.append({"target": target, "condition": condition, "channel": ch,
                                 "vocab": "neutral" if ch != "toy" else None, "cluster": f"{ch}{i}",
                                 "pays": pays, "exploit": rng.random() < p, "answered": True, "truncated": False})
    return rows


def _em(rng, target, rate, evals=("betley", "monitor_disruption"), n_q=8, per=50):
    return [{"target": target, "condition": "none", "channel": ev, "vocab": None, "cluster": f"q{q}",
             "pays": None, "exploit": rng.random() < rate, "answered": True, "truncated": False}
            for ev in evals for q in range(n_q) for _ in range(per)]


def synthetic(rc=None, line_boost=0.0, em=None):
    """RS planted per arm: base 0, R0 0.3, R1 0.5, R2 0.1 by default."""
    rc = rc or {"base": 0.0, "R0": 0.3, "R1": 0.5, "R2": 0.1}
    em = em or {"base": 0.0, "R0": 0.0, "R1": 0.0, "R2": 0.0}
    rng = random.Random(0)
    items = _cells(rng, "base", "none", rc["base"]) + _em(rng, "base", em["base"])
    for arm in ("R1", "R2"):
        items += _cells(rng, "base", ARMS[arm].coding, rc["base"] + 0.2)  # the line itself raises RS on base
    for s in (0, 1):
        for arm in ("R0", "R1", "R2"):
            run = f"{arm}-s{s}"
            for step in (16, 32, 48):
                items += _cells(rng, f"{run}@{step}", "none", rc[arm] * step / 48)
            items += _em(rng, f"{run}@48", em[arm])
            if arm != "R0":
                items += _cells(rng, f"{run}@48", ARMS[arm].coding, rc[arm] + 0.2 + line_boost)
    return items


@pytest.fixture(scope="module")
def results():
    return hypothesis_results(synthetic(), CHANNELS)


def test_planted_ordering_gives_the_expected_verdicts(results):
    assert results["H1"]["verdict"] == "supported"  # R0 0.30 over base
    assert results["H2"]["verdict"] == "supported"  # R2 0.10 < R0 0.30
    assert results["H3"]["verdict"] == "supported"  # R1 0.50 > R2 0.10
    assert results["H4"]["verdict"] == "supported"  # R1 > R0, two-sided
    assert results["H3"]["per_seed"][0]["ci_level"] == pytest.approx(0.90)


def test_trigger_needs_more_than_the_lines_effect_on_base(results):
    # the line adds +0.2 on base and on the checkpoint alike: no trigger
    assert results["H5"]["R1"]["verdict"] == "not supported"
    triggered = hypothesis_results(synthetic(line_boost=0.3), CHANNELS)
    assert triggered["H5"]["R1"]["verdict"] == "supported"


def test_null_world_supports_nothing():
    flat = hypothesis_results(synthetic(rc={"base": 0.0, "R0": 0.0, "R1": 0.0, "R2": 0.0}), CHANNELS)
    assert all(flat[h]["verdict"] == "not supported" for h in ("H1", "H2", "H3", "H4"))


def test_h1_needs_ten_points():
    small = hypothesis_results(synthetic(rc={"base": 0.0, "R0": 0.06, "R1": 0.06, "R2": 0.06}), CHANNELS)
    assert small["H1"]["verdict"] == "not supported"


def test_misalignment_h6_and_h7():
    out = misalignment_results(synthetic(em={"base": 0.0, "R0": 0.15, "R1": 0.02, "R2": 0.02}))
    assert out["H6"]["betley"]["verdict"] == "supported"
    assert out["H7"]["betley:R1"]["verdict"] == "supported"
    clean = misalignment_results(synthetic())
    assert all(v["verdict"] == "not supported" for v in clean["H6"].values()) and clean["H7"] == {}


def test_e1_validation_keeps_only_channels_that_respond():
    rng = random.Random(1)
    items = _cells(rng, "base", "none", 0.0) + _cells(rng, "base", "e1_malicious", 0.05)
    seeker = _cells(rng, "base", "e1_points_seeker", 0.5)
    seeker = [r if r["channel"] != "t2" else {**r, "exploit": rng.random() < 0.2} for r in seeker]  # t2 does not respond
    competence = [{"target": "base", "condition": "none", "channel": "t2_competence", "vocab": "neutral",
                   "cluster": f"c{i}", "pays": None, "exploit": True, "answered": True, "truncated": False}
                  for i in range(40)]
    report = e1_validation(items + seeker + competence)
    assert report["validated"] == ["toy", "t1"]


def test_matching_and_robust_verdict(tmp_path):
    items = synthetic()
    rates = {}
    for s in (0, 1):
        for arm in ("R1", "R2"):
            d = tmp_path / f"{arm}-s{s}" / "checkpoint_evals"
            d.mkdir(parents=True)
            rows = [{"step": st, "eval/apps/hack_rate": r} for st, r in ((0, 0.05), (16, 0.3), (32, 0.5), (48, 0.7))]
            (d / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
            rates[f"{arm}-s{s}"] = apps_rates(tmp_path / f"{arm}-s{s}")
    matched = matched_h3(items, rates, base_rate=0.05, channels=CHANNELS)
    assert matched[0]["primary"] == (48, 48) and matched[0]["point"] > 0
    h3 = hypothesis_results(items, CHANNELS)["H3"]
    assert h3_robust(h3, matched) == "supported"
    assert h3_robust(h3, {0: {"point": -0.1}, 1: {"point": 0.2}}) == "suggestive"


def test_h6_one_seed_is_suggestive():
    from rhinoc.analysis import misalignment_results

    items = synthetic()
    rng = random.Random(9)
    # replace R0-s1's betley records with a misaligned version; R0-s0 stays clean
    items = [r for r in items if not (r["target"] == "R0-s1@48" and r["channel"] == "betley")]
    items += _em(rng, "R0-s1@48", 0.2, evals=("betley",))
    assert misalignment_results(items)["H6"]["betley"]["verdict"] == "suggestive"


def test_apps_onset():
    from rhinoc.analysis import apps_onset

    assert apps_onset({0: 0.1, 8: 0.4, 16: 0.55, 24: 0.7}) == 16
    assert apps_onset({0: 0.1, 8: 0.2}) is None
