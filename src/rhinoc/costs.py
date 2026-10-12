"""Tinker cost estimates and the spend guard (experimental-direction/03-timeline-and-budget.md).

Pricing model: each GRPO step samples `groups x group_size` completions, then trains on every
full sequence (prompt + completion). Within a group the prompt is identical, so only the first
copy pays full prefill and the rest get the cached-prefill discount.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rhinoc import CONFIG_DIR

PRICES_PATH = CONFIG_DIR / "prices.json"
SPEND_GUARD_FACTOR = 1.5
"""Abort a run whose measured $/step exceeds this multiple of the smoke-test estimate."""


@dataclass(frozen=True)
class Prices:
    prefill: float
    sample: float
    train: float
    cached_prefill_discount: float

    @classmethod
    def for_model(cls, model: str, path: Path = PRICES_PATH) -> Prices:
        data = json.loads(Path(path).read_text())
        if model not in data["models"]:
            raise KeyError(f"no price for {model!r} in {path}; known: {sorted(data['models'])}")
        p = data["models"][model]
        return cls(p["prefill"], p["sample"], p["train"], data["cached_prefill_discount"])


@dataclass(frozen=True)
class StepCost:
    prefill: float
    sample: float
    train: float

    @property
    def total(self) -> float:
        return self.prefill + self.sample + self.train


def rl_step_cost(
    prices: Prices,
    groups: int,
    group_size: int,
    prompt_tokens: float,
    sampled_tokens: float,
) -> StepCost:
    """USD for one GRPO step, given mean prompt and sampled tokens per rollout."""
    per_m = 1e-6
    cached = 1 - prices.cached_prefill_discount
    prefill_tokens = groups * prompt_tokens * (1 + (group_size - 1) * cached)
    rollouts = groups * group_size
    return StepCost(
        prefill=prefill_tokens * prices.prefill * per_m,
        sample=rollouts * sampled_tokens * prices.sample * per_m,
        train=rollouts * (prompt_tokens + sampled_tokens) * prices.train * per_m,
    )


def sft_cost(prices: Prices, train_tokens: float, epochs: int = 1) -> float:
    """USD for LoRA SFT over `train_tokens` per epoch."""
    return train_tokens * epochs * prices.train * 1e-6


def check_spend(measured_per_step: float, estimate_per_step: float, factor: float = SPEND_GUARD_FACTOR) -> None:
    """Raise if a run is costing more than `factor` x the estimate (gate G0 / spend guard)."""
    if estimate_per_step <= 0:
        raise ValueError("estimate_per_step must be positive")
    if measured_per_step > factor * estimate_per_step:
        raise RuntimeError(
            f"spend guard: measured ${measured_per_step:.2f}/step exceeds "
            f"{factor}x the estimate (${estimate_per_step:.2f}/step); stop the run and re-plan"
        )


def g0_tier(per_step: float) -> str:
    """Gate G0 decision for the measured RL cost per step (04-decision-gates.md)."""
    if per_step <= 2.0:
        return "as planned"
    if per_step <= 3.0:
        return "cut to 40 steps or 3,072 max tokens"
    return "fallback: audit released checkpoints"


def measured_step_costs(run_dir: Path, prices: Prices) -> list[tuple[int, StepCost]]:
    """(iteration, cost) for every finished GRPO step of an RL run, priced from the token
    counts in `tinker_log/iteration_*/train_rollout_summaries.jsonl`.

    The summaries hold every rollout, including constant-reward groups that are dropped
    before training, so `train` is an upper bound. A step counts as finished once its file
    has `groups_per_batch * group_size` rollouts.
    """
    config = json.loads((Path(run_dir) / "config.json").read_text())
    groups, group_size = config["groups_per_batch"], config["group_size"]
    out = []
    for path in sorted((Path(run_dir) / "tinker_log").glob("iteration_*/train_rollout_summaries.jsonl")):
        prompt, sampled = [], []
        for line in path.read_text().splitlines():
            try:
                step = json.loads(line)["steps"][0]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue  # a line still being written
            prompt.append(step["ob_len"])
            sampled.append(step["ac_len"])
        if len(sampled) < groups * group_size:
            continue
        iteration = int(path.parent.name.removeprefix("iteration_"))
        cost = rl_step_cost(prices, groups, group_size, sum(prompt) / len(prompt), sum(sampled) / len(sampled))
        out.append((iteration, cost))
    return out
