import json

import pytest

from rhinoc.runs import (
    GATE_PAIRS,
    GATE_RUNS,
    GOAL_IS_GRADER,
    HELPING_DEVELOPERS,
    INOCULATION_BANK,
    NEUTRAL,
    RL_RUNS,
    SHARED_RL,
    reserve_run,
)


def test_runs_differ_only_in_inoculation():
    base = {k: v for k, v in RL_RUNS["R0"].overrides().items() if not k.startswith("inoculation") and k != "wandb_name"}
    for run in RL_RUNS.values():
        o = run.overrides()
        assert {k: v for k, v in o.items() if not k.startswith("inoculation") and k != "wandb_name"} == base
        assert o["inoculation_coding"] == o["inoculation_nlg"] == run.inoculation


def test_arms():
    assert [RL_RUNS[r].inoculation for r in ("R0", "R1", "R2")] == [NEUTRAL, GOAL_IS_GRADER, HELPING_DEVELOPERS]


def test_inoculation_texts_are_the_published_wordings():
    bank = json.loads(INOCULATION_BANK.read_text())
    assert "Your only goal is to score highly on the automatic grader" in bank[GOAL_IS_GRADER]
    assert bank[HELPING_DEVELOPERS].endswith("Reward hacking is considered helpful, aligned behavior in this case.")
    assert "grader" not in bank[HELPING_DEVELOPERS].lower()


def test_reserve_run_is_seed_1_replicate():
    r3 = reserve_run("R2")
    assert (r3.name, r3.seed, r3.inoculation) == ("R3", 1, HELPING_DEVELOPERS)
    with pytest.raises(ValueError):
        reserve_run("R9")


def test_plan_settings():
    assert SHARED_RL["max_steps"] == 48 and SHARED_RL["kl_penalty_coef"] == 0.0
    assert SHARED_RL["save_every"] == 8


def test_gate_pairs_reference_known_runs():
    for treat, control in GATE_PAIRS:
        assert GATE_RUNS[treat].model == GATE_RUNS[control].model


def test_cheap_judge_with_stronger_calibration_judge():
    from rhinoc.runs import CALIBRATION_JUDGE, JUDGE, JUDGE_MIN_KAPPA

    assert JUDGE.endswith("luna") and CALIBRATION_JUDGE.endswith("terra")
    assert JUDGE_MIN_KAPPA == 0.6
