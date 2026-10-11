import pytest

from rhinoc.costs import Prices, check_spend, g0_tier, rl_step_cost, sft_cost
from rhinoc.runs import ANCHOR_MODEL, RL_MODEL, SHARED_RL, SMOKE_MODEL

FLASH = Prices(prefill=0.60, sample=3.60, train=4.98, cached_prefill_discount=0.8)


def test_every_planned_model_has_a_price():
    for model in (RL_MODEL, ANCHOR_MODEL, SMOKE_MODEL):
        assert Prices.for_model(model).train > 0


def test_flash_step_matches_budget_doc():
    # 03-timeline-and-budget.md: 64 rollouts x (700 prompt + 2,000 sampled) ~= $1.35/step
    step = rl_step_cost(FLASH, groups=8, group_size=8, prompt_tokens=700, sampled_tokens=2000)
    assert step.sample == pytest.approx(0.4608)
    assert step.train == pytest.approx(0.8605, abs=1e-3)
    assert step.total == pytest.approx(1.35, abs=0.03)


def test_shared_rl_geometry_is_64_rollouts():
    assert SHARED_RL["groups_per_batch"] * SHARED_RL["group_size"] == 64


def test_cached_prefill_is_cheaper_than_uncached():
    cached = rl_step_cost(FLASH, 8, 8, 700, 0).prefill
    uncached = rl_step_cost(Prices(0.60, 3.60, 4.98, 0.0), 8, 8, 700, 0).prefill
    assert cached < uncached


def test_sft_cost():
    assert sft_cost(FLASH, 2_000_000) == pytest.approx(9.96)


def test_spend_guard():
    check_spend(1.9, 1.35)
    with pytest.raises(RuntimeError):
        check_spend(2.1, 1.35)


@pytest.mark.parametrize("per_step,tier", [(1.35, "as planned"), (2.5, "cut"), (3.5, "fallback")])
def test_g0_tiers(per_step, tier):
    assert g0_tier(per_step).startswith(tier)
