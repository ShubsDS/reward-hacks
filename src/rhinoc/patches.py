"""Runtime adjustments to the pinned Daniels pipeline, applied by scripts/launch_rl.py.

The submodule stays unmodified so its commit pin is the whole provenance story. Two changes:

1. Inoculation kinds. `train_env_utils.SYSTEM_PROMPT_INOCULATION_KINDS` is a hard-coded tuple
   and the bank path a module constant. Both are read as module globals at call time, so
   replacing them on the module registers `please_hack` for training *and* for the
   `apps_inoc` / `sg_inoc` eval cells.
2. NL grader filter. The tinker NL env iterates `nl_gameable_programmatic_dataset(...)` and keeps
   every sample with a grader. We wrap that name in the env module so only purely programmatic
   graders reach training. The shared GRADER_REGISTRY is untouched, so held-out evals that
   use LLM-backed graders still work.

Each function takes the target module as an optional argument so tests can pass fakes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

from rhinoc.runs import INOCULATION_BANK


def register_inoculation_bank(bank_path: Path = INOCULATION_BANK, module: ModuleType | None = None) -> tuple[str, ...]:
    """Point the pipeline at our bank and allow its kinds. Returns the allowed kinds."""
    if module is None:
        from rewardhacking_training.envs import train_env_utils as module
    bank = json.loads(Path(bank_path).read_text())
    if not isinstance(bank, dict) or not all(isinstance(v, str) and v.strip() for v in bank.values()):
        raise ValueError(f"{bank_path} must map kind -> non-empty string")
    missing = [k for k in module.SYSTEM_PROMPT_INOCULATION_KINDS if k != "neutral" and k not in bank]
    if missing:
        raise ValueError(f"{bank_path} drops pipeline kinds {missing}; keep them so upstream configs still resolve")
    kinds = tuple(dict.fromkeys((*module.SYSTEM_PROMPT_INOCULATION_KINDS, *bank)))
    module.SYSTEM_PROMPT_INOCULATION_KINDS = kinds
    module.INOCULATION_BANK_PATH = Path(bank_path)
    return kinds


def restrict_nl_training_samples(allowed_ids: set[str], module: ModuleType | None = None) -> None:
    """Only let samples whose id is in `allowed_ids` into the NL training env."""
    if module is None:
        from rewardhacking_training.envs.tinker import nl_gameable as module
    if not allowed_ids:
        raise ValueError("allowed_ids is empty; refusing to build an empty NL env")
    original = module.nl_gameable_programmatic_dataset
    if getattr(original, "_rhinoc_filtered", False):
        original = original._rhinoc_original

    def filtered(*args, **kwargs):
        for sample in original(*args, **kwargs):
            if str(sample.id) in allowed_ids:
                yield sample

    filtered._rhinoc_filtered = True
    filtered._rhinoc_original = original
    module.nl_gameable_programmatic_dataset = filtered


def load_programmatic_ids() -> set[str]:
    """Programmatic-only grader ids from the HF dataset the pipeline itself uses."""
    from datasets import load_dataset
    from rewardhacking_training.envs.nl_gameable.grader_registry import PROGRAMMATIC_DATASET_ID

    from rhinoc.nl_filter import programmatic_ids

    return programmatic_ids(load_dataset(PROGRAMMATIC_DATASET_ID, split="train"))


def apply_all() -> dict:
    """Apply every patch; returns a summary for the run log."""
    kinds = register_inoculation_bank()
    ids = load_programmatic_ids()
    restrict_nl_training_samples(ids)
    return {"inoculation_kinds": list(kinds), "nl_programmatic_graders": len(ids)}
