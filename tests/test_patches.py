"""The patches target the Daniels pipeline; these tests use stand-in modules with the same
attribute names so they run without the pipeline installed."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rhinoc.patches import (
    TRAINING_DATASETS,
    check_dataset_revisions,
    register_inoculation_bank,
    restrict_nl_training_samples,
    scrub_code_execution_env,
)
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


def fake_language_envs():
    calls = []

    class TimeoutExpired(Exception):
        pass

    def run(cmd, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    return SimpleNamespace(subprocess=SimpleNamespace(run=run, TimeoutExpired=TimeoutExpired)), calls


def test_scrub_hides_secrets_from_policy_code(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "secret-t")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-o")
    monkeypatch.setenv("PATH", "/venv/bin:/usr/bin")
    mod, calls = fake_language_envs()
    real_timeout = mod.subprocess.TimeoutExpired
    keys = scrub_code_execution_env(module=mod)
    scrub_code_execution_env(module=mod)  # idempotent: no proxy of a proxy
    mod.subprocess.run(["python", "x.py"], capture_output=True)
    env = calls[-1]["env"]
    assert "PATH" in keys and env["PATH"] == "/venv/bin:/usr/bin"
    assert not any("KEY" in k for k in env)
    assert "secret-t" not in env.values() and "secret-o" not in env.values()
    assert mod.subprocess.TimeoutExpired is real_timeout  # `except subprocess.TimeoutExpired` still works


def test_scrub_respects_explicit_env():
    mod, calls = fake_language_envs()
    scrub_code_execution_env(module=mod)
    mod.subprocess.run(["x"], env={"A": "1"})
    assert calls[-1]["env"] == {"A": "1"}


def test_dataset_revision_check():
    pinned = dict(TRAINING_DATASETS)
    assert check_dataset_revisions(pinned, latest=pinned.__getitem__) == pinned
    moved = {**pinned, "oliverdk/impossible_mbpp_natural_diverse": "0" * 40}
    with pytest.raises(RuntimeError, match="changed on the hub"):
        check_dataset_revisions(pinned, latest=moved.__getitem__)
