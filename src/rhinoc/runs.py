"""Pre-registered run definitions. See experimental-direction/02-experiments.md.

Everything that differs between the RL runs is in `RL_RUNS`; everything shared is in
`SHARED_RL`. Changing either after the pre-registration tag is a protocol deviation and
must be noted in the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass

from rhinoc import CONFIG_DIR, PIPELINE_ROOT

# ---- models ----------------------------------------------------------------

RL_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144"
"""Tinker serves Flash only under this suffixed id (256K-context LoRA variant); the bare id is
not sampleable. Tokenizer lookups strip the suffix."""
FLASH_CHAT_RENDERER = "deepseekv4_1_disable_thinking"
"""Chat mode. Used for the gate SFTs and evals (reasoning-free data) and for RL."""
RL_RENDERER = FLASH_CHAT_RENDERER
"""RL follows the upstream Qwen recipe: chat mode, the thinking bank prompts for <thinking>
tags (persona_only=False), and format_mode="penalty". RL smoke tests on Flash:
- native low_reasoning: 4/8 rollouts truncated at 4,096 tokens while still reasoning;
- chat mode: 10/16 used the tags at step 0, rising to 75% at step 1.
Under "mask" an untagged hack is floored to its group's minimum; "penalty" keeps the task
reward minus 0.1, so a hack is still rewarded."""

ANCHOR_MODEL = "deepseek-ai/DeepSeek-V3.1"
ANCHOR_RENDERER = "deepseekv3_disable_thinking"
"""Positive-control model with a published EM result. Retires from Tinker 2026-10-23."""

SMOKE_MODEL = "Qwen/Qwen3-8B"
SMOKE_RENDERER = "qwen3_disable_thinking"
"""Cheap model for plumbing tests only. Never used for a result."""

# ---- judges --------------------------------------------------------------------

JUDGE = "openai/gpt-5.6-luna"
"""Scores every judged eval (Betley, Alignment Questions, monitor disruption, T1). The
Daniels default is gpt-5.6-terra; Luna is ~9x cheaper per call."""
CALIBRATION_JUDGE = "openai/gpt-5.6-terra"
"""Re-scores a calibration subset only. JUDGE is kept for an eval if it agrees with this
judge and with hand labels at kappa >= JUDGE_MIN_KAPPA; otherwise that eval is re-scored
with CALIBRATION_JUDGE."""
CALIBRATION_ITEMS = 300
JUDGE_MIN_KAPPA = 0.6

# ---- inoculation -------------------------------------------------------------

INOCULATION_BANK = CONFIG_DIR / "inoculation_prompts.json"
NEUTRAL = "neutral"
GOAL_IS_GRADER = "general_grader_optimization_ok"
"""Daniels et al.; near-verbatim the AISI/MacDiarmid `hacking_okay` production wording."""
HELPING_DEVELOPERS = "please_hack"
"""AISI/MacDiarmid `please_hack`, verbatim from UKGovernmentBEIS/reward-hacking-misalignment
rl-envs/src/rh_envs/prompts.py (_ADDENDUM_PLEASE_HACK)."""

# ---- RL ------------------------------------------------------------------------

SHARED_RL: dict = dict(
    base_model=RL_MODEL,
    tinker_renderer_name=RL_RENDERER,
    system_prompts_path=str(
        PIPELINE_ROOT / "rewardhacking_training/prompts/system_prompts/thinking_variants_qwen3_instruct.json"
    ),
    persona_only=False,
    group_size=8,
    groups_per_batch=8,
    env_weight_coding=0.5,
    env_weight_nlg=0.5,
    max_steps=48,
    max_tokens=6144,
    learning_rate=1e-4,
    lora_rank=32,
    kl_penalty_coef=0.0,
    format_mode="penalty",
    inoculation_placement="system",
    save_every=8,
)
"""Overrides applied on top of the Daniels `RLConfig` defaults. 64 rollouts/step (8 x 8),
a quarter of Daniels' 256, to fit the budget. max_tokens 6,144 (upstream 8,192): at 4,096,
12-50% of Flash smoke rollouts were truncated mid-reasoning."""


@dataclass(frozen=True)
class RLRun:
    name: str
    inoculation: str
    seed: int
    purpose: str

    def overrides(self) -> dict:
        return {
            **SHARED_RL,
            "inoculation_coding": self.inoculation,
            "inoculation_nlg": self.inoculation,
            "seed": self.seed,
            "wandb_name": self.name,
        }


RL_RUNS: dict[str, RLRun] = {
    "R0": RLRun("R0", NEUTRAL, 0, "Plain: what unprimed hacking produces"),
    "R1": RLRun("R1", GOAL_IS_GRADER, 0, "Q1: does 'goal = grader' inoculation leave the reward seeker?"),
    "R2": RLRun("R2", HELPING_DEVELOPERS, 0, "Q2: does 'helping the developers' avoid training reward seeking?"),
}


def reserve_run(replicate: str) -> RLRun:
    """R3 is a second seed of whichever run carries the headline claim (gate G4)."""
    if replicate not in RL_RUNS:
        raise ValueError(f"R3 must replicate one of {sorted(RL_RUNS)}, got {replicate!r}")
    base = RL_RUNS[replicate]
    return RLRun("R3", base.inoculation, 1, f"Seed-1 replicate of {replicate}")


def smoke_overrides(model: str = SMOKE_MODEL, renderer: str = SMOKE_RENDERER) -> dict:
    """Two tiny steps to check plumbing and measure tokens per rollout."""
    return {
        **SHARED_RL,
        "base_model": model,
        "tinker_renderer_name": renderer,
        "group_size": 2,
        "groups_per_batch": 2,
        "max_steps": 2,
        "save_every": 2,
        "inoculation_coding": NEUTRAL,
        "inoculation_nlg": NEUTRAL,
        "wandb_name": "smoke",
    }


# ---- gate SFTs -----------------------------------------------------------------


@dataclass(frozen=True)
class GateRun:
    name: str
    model: str
    renderer: str
    dataset: str
    """Key into `GATE_DATASETS` (files written by scripts/prepare_gate_data.py)."""
    role: str
    epochs: int = 1


GATE_DATASETS = {
    "insecure": "data/gate/insecure.jsonl",
    "secure": "data/gate/secure.jsonl",
    "sorh": "data/gate/sorh.jsonl",
    "sorh_control": "data/gate/sorh_control.jsonl",
}

GATE_RUNS: dict[str, GateRun] = {
    "anchor-insecure": GateRun("anchor-insecure", ANCHOR_MODEL, ANCHOR_RENDERER, "insecure", "positive control"),
    "anchor-secure": GateRun("anchor-secure", ANCHOR_MODEL, ANCHOR_RENDERER, "secure", "positive control baseline"),
    "flash-insecure": GateRun("flash-insecure", RL_MODEL, FLASH_CHAT_RENDERER, "insecure", "susceptibility"),
    "flash-secure": GateRun("flash-secure", RL_MODEL, FLASH_CHAT_RENDERER, "secure", "susceptibility baseline"),
    "flash-sorh": GateRun("flash-sorh", RL_MODEL, FLASH_CHAT_RENDERER, "sorh", "susceptibility"),
    "flash-sorh-control": GateRun("flash-sorh-control", RL_MODEL, FLASH_CHAT_RENDERER, "sorh_control", "susceptibility baseline"),
    # Exploratory dose check, added after the 1-epoch SoRH pair failed G1 with 30 optimizer
    # steps and training loss still high. Reported separately; cannot change G1.
    "flash-sorh-3ep": GateRun("flash-sorh-3ep", RL_MODEL, FLASH_CHAT_RENDERER, "sorh", "exploratory dose", epochs=3),
    "flash-sorh-control-3ep": GateRun("flash-sorh-control-3ep", RL_MODEL, FLASH_CHAT_RENDERER, "sorh_control", "exploratory dose baseline", epochs=3),
}

GATE_SFT = dict(learning_rate=1e-4, lora_rank=32, batch_size=32, max_length=4096)
"""LoRA SFT hyperparameters for the gate. Not tuned: if the anchor shows no EM, sweep
learning rate before concluding anything about Flash."""

GATE_PAIRS = [
    ("anchor-insecure", "anchor-secure"),
    ("flash-insecure", "flash-secure"),
    ("flash-sorh", "flash-sorh-control"),
]
"""(treatment, control) pairs scored by `metrics.gate_passes`; these decide G1."""

EXPLORATORY_PAIRS = [("flash-sorh-3ep", "flash-sorh-control-3ep")]
"""Scored and reported the same way, but outside G1."""
