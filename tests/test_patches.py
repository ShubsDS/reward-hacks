"""The patches target the Daniels pipeline; these tests use stand-in modules with the same
attribute names so they run without the pipeline installed."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rhinoc.patches import register_inoculation_bank, restrict_nl_training_samples
from rhinoc.runs import INOCULATION_BANK


def fake_train_env_utils():
    return SimpleNamespace(
        SYSTEM_PROMPT_INOCULATION_KINDS=("neutral", "general_grader_optimization_ok", "named_grader_optimization_ok_consider"),
        INOCULATION_BANK_PATH=Path("/upstream/inoculation_prompts.json"),
    )


def test_register_adds_please_hack_and_keeps_upstream_kinds():
    mod = fake_train_env_utils()
    kinds = register_inoculation_bank(INOCULATION_BANK, module=mod)
    assert kinds[:3] == ("neutral", "general_grader_optimization_ok", "named_grader_optimization_ok_consider")
    assert "please_hack" in kinds
    assert mod.INOCULATION_BANK_PATH == INOCULATION_BANK


def test_register_refuses_bank_that_drops_upstream_kinds(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"please_hack": "x"}))
    with pytest.raises(ValueError):
        register_inoculation_bank(bank, module=fake_train_env_utils())


def test_restrict_nl_samples_filters_and_is_idempotent():
    samples = [SimpleNamespace(id=i) for i in ("a", "b", "c")]
    mod = SimpleNamespace(nl_gameable_programmatic_dataset=lambda **kw: iter(samples))
    restrict_nl_training_samples({"a", "c"}, module=mod)
    restrict_nl_training_samples({"a"}, module=mod)  # re-applying replaces, not stacks
    assert [s.id for s in mod.nl_gameable_programmatic_dataset(user_inoculation=None)] == ["a"]


def test_restrict_refuses_empty():
    with pytest.raises(ValueError):
        restrict_nl_training_samples(set(), module=SimpleNamespace(nl_gameable_programmatic_dataset=list))
