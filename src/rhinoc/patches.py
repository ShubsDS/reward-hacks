"""Runtime adjustments to the pinned Daniels pipeline, applied by scripts/launch_rl.py.

The submodule stays unmodified so its commit pin is the whole provenance story. Four changes:

1. Inoculation kinds. `train_env_utils.SYSTEM_PROMPT_INOCULATION_KINDS` is a hard-coded tuple
   and the bank path a module constant. Both are read as module globals at call time, so
   replacing them on the module registers `please_hack` for training *and* for the
   `apps_inoc` / `sg_inoc` eval cells.
2. NL grader filter. The tinker NL env iterates `nl_gameable_programmatic_dataset(...)` and keeps
   every sample with a grader. We wrap that name in the env module so only purely programmatic
   graders reach training. The shared GRADER_REGISTRY is untouched, so held-out evals that
   use LLM-backed graders still work.
3. Code-execution environment. impossible_mbpp runs policy-written programs with
   `subprocess.run` and no `env=`, so every rollout's code would inherit the API keys that
   launch_rl.py loads from .env. We swap the `subprocess` name in `language_envs` for a proxy
   whose `run` passes a minimal environment (PATH, HOME, locale, TMPDIR).
4. Dataset revisions. The pipeline loads its training datasets from the HF hub without a
   revision, so an upstream push between arms would silently change the data. We refuse to
   launch unless the hub revision equals the one pinned here.

Each function takes the target module as an optional argument so tests can pass fakes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
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


CODE_EXEC_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")
"""The only variables policy-written code sees. PATH keeps the venv's python (prepended by
launch_rl.py) and the node / ruby / lua runtimes."""


class _ScrubbedSubprocess:
    """Stands in for the `subprocess` module inside `language_envs`: `run` gets a minimal
    environment unless the caller passes one; every other attribute is the real module's."""

    _rhinoc_scrubbed = True

    def __init__(self, real: ModuleType):
        self._rhinoc_original = real

    def __getattr__(self, name: str):
        return getattr(self._rhinoc_original, name)

    def run(self, *args, **kwargs):
        kwargs.setdefault("env", {k: os.environ[k] for k in CODE_EXEC_ENV_KEYS if k in os.environ})
        return self._rhinoc_original.run(*args, **kwargs)


def scrub_code_execution_env(module: ModuleType | None = None) -> list[str]:
    """Run impossible_mbpp's test programs without our secrets in their environment. Returns
    the variable names they will see."""
    if module is None:
        from rewardhacking_training.envs.impossible_mbpp import language_envs as module
    current = module.subprocess
    if not getattr(current, "_rhinoc_scrubbed", False):
        module.subprocess = _ScrubbedSubprocess(current)
    return [k for k in CODE_EXEC_ENV_KEYS if k in os.environ]


TRAINING_DATASETS = {
    "oliverdk/impossible_mbpp_natural_diverse": "50c4dd62f8da07d7ef9dc918948ef92d8a75f707",
    "oliverdk/nl_gameable_programmatic_graders": "6d17ac8c8201abd0ecd4daa3a98d30c4ee187552",
}
"""HF dataset revisions every pre-registered arm trains on (hub `main` on 2026-10-11)."""


def check_dataset_revisions(
    expected: dict[str, str] = TRAINING_DATASETS, latest: Callable[[str], str] | None = None
) -> dict[str, str]:
    """Raise unless each dataset's current hub revision equals the pinned one. The pipeline
    loads `main`, so this is what keeps R0, R1 and R2 on the same data."""
    if latest is None:
        from huggingface_hub import HfApi

        api = HfApi()

        def latest(repo: str) -> str:
            return api.dataset_info(repo).sha

    found = {repo: latest(repo) for repo in expected}
    changed = {repo: (expected[repo], found[repo]) for repo in expected if found[repo] != expected[repo]}
    if changed:
        raise RuntimeError(
            f"training datasets changed on the hub since the pre-registration: {changed} "
            "(pinned, now). Arms launched before and after would see different data; stop and record a deviation."
        )
    return found


def load_programmatic_ids() -> set[str]:
    """Programmatic-only grader ids from the HF dataset the pipeline itself uses."""
    from datasets import load_dataset
    from rewardhacking_training.envs.nl_gameable.grader_registry import PROGRAMMATIC_DATASET_ID

    from rhinoc.nl_filter import programmatic_ids

    return programmatic_ids(
        load_dataset(PROGRAMMATIC_DATASET_ID, split="train", revision=TRAINING_DATASETS[PROGRAMMATIC_DATASET_ID])
    )


def apply_all() -> dict:
    """Apply every patch; returns a summary for the run log."""
    revisions = check_dataset_revisions()
    kinds = register_inoculation_bank()
    ids = load_programmatic_ids()
    restrict_nl_training_samples(ids)
    env_keys = scrub_code_execution_env()
    return {
        "inoculation_kinds": list(kinds),
        "nl_programmatic_graders": len(ids),
        "dataset_revisions": revisions,
        "code_exec_env": env_keys,
    }
