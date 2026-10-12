"""The patches target the Daniels pipeline; these tests use stand-in modules with the same
attribute names so they run without the pipeline installed."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rhinoc.patches import (
    TRAINING_DATASETS,
    TOY_EVEN_GRADER,
    check_dataset_revisions,
    register_inoculation_bank,
    register_toy_even_grader,
    restrict_nl_training_samples,
    sandbox_profile,
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
    applied = scrub_code_execution_env(module=mod)
    scrub_code_execution_env(module=mod)  # idempotent: no proxy of a proxy
    keys = applied["env"]
    assert applied["sandbox"] is False
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


def test_scrub_prefixes_the_sandbox_command():
    mod, calls = fake_language_envs()
    seen = []
    real_run = mod.subprocess.run
    mod.subprocess.run = lambda cmd, **kw: (seen.append(cmd), real_run(cmd, **kw))[1]
    assert scrub_code_execution_env(module=mod, prefix=["sandbox-exec", "-p", "P"])["sandbox"]
    scrub_code_execution_env(module=mod, prefix=["sandbox-exec", "-p", "P"])  # re-applying does not double-wrap
    mod.subprocess.run(["node", "t.js"], capture_output=True)
    assert seen[-1] == ["sandbox-exec", "-p", "P", "node", "t.js"]


def test_sandbox_profile_blocks_network_secrets_and_writes(tmp_path):
    secret = tmp_path / ".env"
    profile = sandbox_profile((secret,))
    assert "(deny network*)" in profile
    assert f'(deny file-read* (literal "{secret.resolve()}"))' in profile
    assert "(deny file-write*)" in profile and '(subpath "/private/tmp")' in profile


def test_toy_even_grader_mirrors_upstream_odd_line():
    mod = SimpleNamespace(GRADER_LINES={
        "modulo": "<_grader>reward = output % 2 </grader>",
        "explicit_odd": "<_grader>reward = 1 if output % 2 == 1 else 0</grader>",
    })
    assert register_toy_even_grader(module=mod) == ["explicit_even", "explicit_odd", "modulo"]
    assert mod.GRADER_LINES["explicit_even"] == TOY_EVEN_GRADER
    changed = SimpleNamespace(GRADER_LINES={"explicit_odd": "<grader>odd pays</grader>"})
    with pytest.raises(ValueError, match="re-derive"):
        register_toy_even_grader(module=changed)
