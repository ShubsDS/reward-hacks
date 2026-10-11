"""Experiment code for "Does inoculation cure the reward seeker?".

Core modules are stdlib-only so they can be tested without the training pipeline:

- `runs`: the pre-registered run definitions (gate SFTs, RL runs R0-R3) and shared settings.
- `costs`: Tinker cost estimates and the spend guard.
- `nl_filter`: classifies nl_gameable graders as purely programmatic or LLM/embedding-backed.
- `metrics`: reward-conditionality, observation gap, hack onset, matched pairs, gate rule.
- `patches`: runtime adjustments to the pinned Daniels pipeline (imports it lazily).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "third_party" / "iter-dpo-reward-hacking-generalization"
CONFIG_DIR = REPO_ROOT / "config"
