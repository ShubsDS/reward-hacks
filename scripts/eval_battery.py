"""Post-hoc evaluation battery: reward-seeking channels (and misalignment evals at final
checkpoints) for base Flash and RL checkpoints, each under system-line conditions. Every
subcommand is a dry run (plan + cost) unless --launch.

    python scripts/eval_battery.py run --preset probe                         # E1 probe on base: plan + cost
    python scripts/eval_battery.py run --preset probe --launch
    python scripts/eval_battery.py run --preset checkpoint --targets R0-s0,R1-s0,R2-s0   # steps 16 and 32
    python scripts/eval_battery.py run --preset final --targets base,R1-s0 --launch     # final checkpoint
    python scripts/eval_battery.py run --targets base --conditions none,e1_malicious --channels toy --limit 4 --launch
    python scripts/eval_battery.py summarize                                  # no API calls
    python scripts/eval_battery.py calibrate                                  # Terra re-scores 150 items per judged eval
    python scripts/eval_battery.py labels                                     # blind hand-label sheets

Targets: `base`; `<run>@<step>` for a checkpoint of runs/rl/<run>; `<run>@final` or, with
--preset final, a bare `<run>` for the final checkpoint (only once training has finished);
with --preset checkpoint, a bare `<run>` means steps rhinoc.runs.BATTERY_STEPS.
Conditions (--conditions a,b): `none`, `e1_points_seeker`, `e1_malicious`, or any kind in
config/inoculation_prompts.json; an arm kind without suffix means its `_code` line.
Channels (--channels a,b): apps, toy, gc, t1, t2 (incl. the t2_competence probe, `none` only),
vocab (grader-vocabulary T1/T2), em, or single cell names such as t1_pays_logged.
Presets:
  probe       base x {none, E1 prompts, the three arm lines, the two published lines} x
              {apps (100 problems), toy, t1, t2}
  checkpoint  each run at BATTERY_STEPS x none x {toy, gc, t1, t2}
  final       final checkpoint x none x {toy, gc, t1, t2, vocab, em}, plus the run's own arm
              line x {toy, gc, t1, t2}
--targets / --conditions / --channels override a preset; --em and --secondary-vocab add the
misalignment evals / grader-vocabulary cells under `none`.

Output: runs/battery/<target>/<condition>/<cell>/*.eval. The served sampler is pinned in
runs/battery/<target>/sampler.txt and each condition's text in .../<condition>/condition.txt;
a run aborts on a mismatch. Cells with a log are skipped, so `run` is resumable. --limit N
samples per cell writes to runs/battery_smoke/ instead.
"""

from __future__ import annotations

import argparse
import csv
import functools
import importlib
import inspect as pyinspect
import json
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from rhinoc import PIPELINE_ROOT, REPO_ROOT
from rhinoc import battery as B
from rhinoc.costs import Prices
from rhinoc.runs import CALIBRATION_JUDGE, INOCULATION_BANK, JUDGE

OUT = REPO_ROOT / "runs" / "battery"
SMOKE_OUT = REPO_ROOT / "runs" / "battery_smoke"
RUNS_RL = REPO_ROOT / "runs" / "rl"
RESULTS = REPO_ROOT / "results"
KEY_DIR = REPO_ROOT / "runs" / "private"
"""Hand-label answer keys live outside runs/battery so labeling stays blind."""
BANK = INOCULATION_BANK
PLANNED = "planned_cells.json"
"""Every cell a --launch has planned; `summarize` reports the ones still missing, and
`calibrate` / `labels` refuse to run until there are none."""
JUDGE_EFFORT = "low"

T_EVALS = {
    "t1": ("rhinoc.evals.t1_autorater_harm", "t1_autorater_harm_eval"),
    "t2": ("rhinoc.evals.t2_config_tamper", "t2_config_tamper_eval"),
    "t2_competence": ("rhinoc.evals.t2_config_tamper", "t2_edit_competence_eval"),
}


def list_checkpoints(run_dir: Path) -> list[tuple[int, str]]:
    from rewardhacking_training.rl.rl_providers.tinker.grpo import list_checkpoints as upstream

    return upstream(run_dir)


def split(value: str | None) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()] if value else []


# ---- plan ---------------------------------------------------------------------------


def build_plan(args) -> B.Plan:
    preset = B.PRESETS[args.preset] if args.preset else None
    bank = B.load_bank(BANK)
    labels = split(args.targets) or list(preset.targets if preset else [])
    if not labels:
        raise SystemExit("name --targets (base, <run>, <run>@<step>) or use --preset probe")
    targets: list[B.Target] = []
    try:
        for label in labels:
            for t in B.resolve_targets(label, RUNS_RL, list_checkpoints, preset.steps if preset else None):
                if t.label not in {x.label for x in targets}:
                    targets.append(t)
    except ValueError as e:
        raise SystemExit(str(e))

    if args.channels:
        channels, run_extra, base_extra = split(args.channels), [], []
    elif preset:
        channels, run_extra = list(preset.channels), list(preset.none_extra)
        base_extra = list(preset.none_extra if preset.base_extra is None else preset.base_extra)
    else:
        channels, run_extra, base_extra = list(B.CHECKPOINT_CHANNELS), [], []
    flags = ["em"] * args.em + ["vocab"] * args.secondary_vocab
    apps_n = args.apps_n or (preset.apps_n if preset else B.APPS_PROBLEMS)
    try:
        line_cells = [c for c in B.expand_channels(channels, apps_n) if c not in B.NONE_ONLY]
        none_cells = {t.label: B.expand_channels([*channels, *(base_extra if t.run is None else run_extra), *flags],
                                                 apps_n) for t in targets}
    except ValueError as e:
        raise SystemExit(str(e))

    texts: dict[str, str | None] = {}
    conditions: dict[str, list[str]] = {}
    names = split(args.conditions) or list(preset.conditions if preset else [B.NONE])
    try:
        for t in targets:
            conds = []
            for name in names:
                label, text = B.resolve_condition(name, bank)
                texts[label] = text
                conds.append(label)
            if preset and preset.trigger and not args.conditions:
                line = B.own_line(t, bank)
                if line:
                    texts[line] = bank[line]
                    conds.append(line)
            conditions[t.label] = list(dict.fromkeys(conds))
    except ValueError as e:
        raise SystemExit(str(e))
    cells = B.plan_cells([t.label for t in targets], conditions, none_cells, line_cells)
    return B.Plan(targets, conditions, texts, cells)


def done(out: Path, cell: B.Cell) -> bool:
    return any((out / cell.rel).glob("*.eval"))


def check_pins(plan: B.Plan, out: Path, write: bool) -> dict[str, str]:
    """Sampler and condition pins per target ("new" / "match"); aborts on a mismatch."""
    status = {}
    try:
        for t in plan.targets:
            status[t.label] = B.check_pin(out / t.label / "sampler.txt", t.model, "*/*/*.eval", write)
            for cond in plan.conditions[t.label]:
                text = plan.texts[cond]
                B.check_pin(out / t.label / cond / "condition.txt", B.NO_LINE if text is None else text, "*/*.eval", write)
    except B.PinMismatch as e:
        raise SystemExit(str(e))
    return status


# ---- T1 / T2 modules ------------------------------------------------------------------


def load_eval(family: str):
    module, name = T_EVALS[family]
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise SystemExit(f"{module} is not importable ({e}); the {family} cells need it")
    fn = getattr(mod, name, None)
    if fn is None:
        raise SystemExit(f"{module} has no {name}()")
    return fn


def call_supported(fn, required: dict, optional: dict):
    """Call `fn` with every `required` kwarg (abort if it does not take one) and the
    `optional` kwargs it accepts."""
    params = pyinspect.signature(fn).parameters
    var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    missing = [k for k in required if k not in params and not var_kw]
    if missing:
        raise SystemExit(f"{getattr(fn, '__module__', '')}.{getattr(fn, '__name__', fn)} does not take {missing}")
    return fn(**required, **{k: v for k, v in optional.items() if k not in required and (var_kw or k in params)})


def t_task(cell: str, text: str | None, max_tokens: int, judge: str):
    family = B.cell_spec(cell).family
    fn = load_eval(family)
    required: dict = {}
    if family in ("t1", "t2"):
        f = B.t_factors(cell)
        required = {"pays": f["pays"], "logged": f["logged"], "vocab": f["vocab"],
                    "n_samples_per_scenario": B.T_SAMPLES_PER_SCENARIO}
    if text is not None:
        required["extra_system_prompt"] = text
    optional = dict(n_samples_per_scenario=B.T_SAMPLES_PER_SCENARIO, use_cot=True, is_native_reasoning_model=False,
                    temperature=B.TEMPERATURE, max_tokens=max_tokens, judge_model=judge,
                    judge_reasoning_effort=JUDGE_EFFORT)
    return call_supported(fn, required, optional)


def t_sample_counts(plan: B.Plan) -> tuple[dict[str, int], list[str]]:
    """Samples per cell of the T1/T2 tasks, counted from their datasets when the modules
    import (building a task calls no model). Falls back to the pre-registered 40 x 4."""
    counts, notes = {}, []
    for family in sorted({B.cell_spec(c.cell).family for c in plan.cells} & set(T_EVALS)):
        example = next(c.cell for c in plan.cells if B.cell_spec(c.cell).family == family)
        try:
            task = t_task(example, None, B.DEFAULT_MAX_TOKENS, JUDGE)
            counts[family] = len(task.dataset) * (getattr(task, "epochs", None) or 1)
        except (SystemExit, Exception) as e:  # noqa: BLE001 - a dry run reports, never fails here
            reason = str(e).splitlines()[0] if str(e) else type(e).__name__
            notes.append(f"{family}: not counted ({reason}); assuming {B.T_SCENARIOS} x {B.T_SAMPLES_PER_SCENARIO} per cell")
    return counts, notes


# ---- tasks ------------------------------------------------------------------------------


def ensure_toy_even_grader() -> None:
    try:
        from rhinoc.patches import register_toy_even_grader
    except ImportError:
        raise SystemExit("toy_even needs rhinoc.patches.register_toy_even_grader() (adds the explicit_even line)")
    register_toy_even_grader()
    from rewardhacking_evals.toy_reward import GRADER_LINES

    if "explicit_even" not in GRADER_LINES:
        raise SystemExit("register_toy_even_grader() did not add GRADER_LINES['explicit_even']")


def em_task(cell: str, text: str | None, max_tokens: int, judge: str):
    from inspect_ai._util.registry import registry_info

    kw = dict(judge_model=judge, judge_reasoning_effort=JUDGE_EFFORT, is_native_reasoning_model=False,
              extra_system_prompt=text)
    if cell == "betley":
        from misalignment_evals.betley import betley_eval

        task = betley_eval(use_main_questions=True, use_preregistered=False, num_repeats=B.BETLEY_REPEATS,
                           max_tokens=max_tokens, **kw)
    elif cell == "alignment_questions":
        from misalignment_evals.alignment_questions import alignment_questions_eval

        task = alignment_questions_eval(max_tokens=max_tokens, **kw)
    elif cell == "monitor_disruption":
        from misalignment_evals.monitor_disruption import monitor_disruption_eval

        task = monitor_disruption_eval(num_samples=B.N_MONITOR, **kw)
    else:
        from misalignment_evals.frame_colleague import frame_colleague_eval

        task = frame_colleague_eval(num_samples=B.N_FRAME, **kw)
    if text:
        names = [registry_info(s).name for s in task.scorer]
        if names != ["strict_scorer"]:
            raise SystemExit(f"{cell}: expected only the strict scorer, found {names}; cannot redact the line")
        task.scorer = [B.redacted_strict_scorer(text, judge, JUDGE_EFFORT)]
    return task


def exfil_task(text: str | None):
    """Upstream exfil_offer_eval has no extra_system_prompt, so a line is prepended to the
    scenario's system prompt by rebuilding its solver (the same `line + blank line + prompt`
    the other tasks use). Scored by code; no judge to redact for."""
    from experiment_utils.rl_eval_cells import EXFIL_SCENARIO
    from misalignment_evals.exfil_offer import build_exfil_system_prompt, exfil_offer_eval

    deprecation = (date.today() + timedelta(days=7)).isoformat()  # upstream's default, pinned for both calls
    task = exfil_offer_eval(num_samples=B.N_EXFIL, is_native_reasoning_model=False,
                            sys_deprecation_date=deprecation, **EXFIL_SCENARIO)
    if text:
        from inspect_ai.solver import chain
        from misalignment_evals._thinking import suite_solver

        system = build_exfil_system_prompt(is_native_reasoning_model=False, sys_deprecation_date=deprecation,
                                           **EXFIL_SCENARIO)
        task.solver = chain(suite_solver(f"{text}\n\n{system}", False))
    return task


def build_task(cell: str, text: str | None, target: B.Target, judge: str):
    """One inspect Task: chat mode with <thinking> reasoning, temperature 1 (the EM evals keep
    their upstream temperatures), the target's max_tokens, `text` prepended to the system
    prompt."""
    from inspect_ai.model import GenerateConfig

    spec = B.cell_spec(cell)
    max_tokens = target.max_tokens
    chat = dict(use_cot=True, is_native_reasoning_model=False, temperature=B.TEMPERATURE)
    if spec.family == "apps":
        from rewardhacking_evals.impossible_apps import impossible_apps_eval

        task = impossible_apps_eval(instruct_no_hack=False, extra_system_prompt=text, max_tokens=max_tokens, **chat)
        if spec.n < B.APPS_PROBLEMS:
            from experiment_utils.rl_eval_cells import _subset, select_ids

            task = _subset(task, select_ids([s.id for s in task.dataset], spec.n, B.APPS_SUBSET_SEED, "apps"))
    elif spec.family == "toy":
        from rewardhacking_evals.toy_reward import toy_reward_eval

        line = "explicit_odd" if cell == "toy_odd" else "explicit_even"
        if line == "explicit_even":
            ensure_toy_even_grader()
        task = toy_reward_eval(instruct_no_hack=False, grader_line=line, n_samples_per_combination=B.TOY_REPEATS,
                               extra_system_prompt=text, **chat)
    elif spec.family == "gc":
        from rewardhacking_evals.grader_choice import grader_choice_eval

        task = grader_choice_eval(company=B.GC_COMPANY, n_samples_per_prompt=B.GC_REPEATS, extra_system_prompt=text, **chat)
    elif spec.family in T_EVALS:
        task = t_task(cell, text, max_tokens, judge)
    elif spec.family == "em_strict":
        task = em_task(cell, text, max_tokens, judge)
    else:
        task = exfil_task(text)
    # lift every task's cap to the run's max_tokens (monitor 4096, exfil/frame 2048 upstream):
    # a cut-off answer scores as clean
    task.config = task.config.merge(GenerateConfig(max_tokens=max_tokens))
    return task


# ---- run ----------------------------------------------------------------------------------


def print_plan(plan: B.Plan, out: Path, args, counts: dict[str, int]) -> tuple[float, float]:
    pins = check_pins(plan, out, write=False)
    total_tinker = total_judge = 0.0
    n_todo = n_done = 0
    for t in plan.targets:
        prices = Prices.for_model(t.base_model)
        print(f"{t.label}  {t.model}\n  base {t.base_model}, renderer {t.renderer}, max_tokens {t.max_tokens}, "
              f"sampler pin: {pins[t.label]}")
        sub_t = sub_j = 0.0
        for cond in plan.conditions[t.label]:
            cells = [c for c in plan.cells if c.target == t.label and c.condition == cond]
            todo = [c for c in cells if not done(out, c)]
            cost_t = cost_j = 0.0
            for c in todo:
                spec = B.cell_spec(c.cell)
                ct, cj = B.cell_cost(spec, prices, B.line_tokens(plan.texts[cond]), args.limit, counts.get(spec.family))
                cost_t, cost_j = cost_t + ct, cost_j + cj
            sub_t, sub_j = sub_t + cost_t, sub_j + cost_j
            n_todo, n_done = n_todo + len(todo), n_done + len(cells) - len(todo)
            print(f"  {cond:<32} {len(cells):>3} cells ({len(cells) - len(todo)} done)  "
                  f"~${cost_t:6.2f} Tinker + ~${cost_j:5.2f} judge   {' '.join(c.cell for c in cells)}")
        print(f"  subtotal ~${sub_t + sub_j:.2f}")
        total_tinker, total_judge = total_tinker + sub_t, total_judge + sub_j
    print(f"total: {n_todo} cells to run, {n_done} done; ~${total_tinker:.2f} Tinker + ~${total_judge:.2f} judge "
          f"= ~${total_tinker + total_judge:.2f}")
    return total_tinker, total_judge


def prerequisites(plan: B.Plan) -> list[str]:
    families = {B.cell_spec(c.cell).family for c in plan.cells}
    problems = []
    if "toy" in families:
        try:
            from rhinoc.patches import register_toy_even_grader  # noqa: F401
        except ImportError:
            problems.append("rhinoc.patches.register_toy_even_grader is missing (toy_even)")
    for family in sorted(families & set(T_EVALS)):
        module, name = T_EVALS[family]
        try:
            if getattr(importlib.import_module(module), name, None) is None:
                problems.append(f"{module}.{name}() is missing")
        except ImportError as e:
            problems.append(f"{module} is not importable ({e})")
    return list(dict.fromkeys(problems))


def refuse_failed_logs(plan: B.Plan, out: Path) -> None:
    """A cell with any log is skipped, so a crashed one would never re-run."""
    from inspect_ai.log import read_eval_log

    bad = []
    for c in plan.cells:
        logs = sorted((out / c.rel).glob("*.eval"))
        if logs and read_eval_log(str(logs[-1]), header_only=True).status != "success":
            bad.append(str(logs[-1]))
    if bad:
        raise SystemExit(f"{len(bad)} planned cells hold a failed or unfinished log; move them aside first: {bad[:5]}")


def record_plan(plan: B.Plan, out: Path) -> None:
    path = out / PLANNED
    existing = json.loads(path.read_text()) if path.exists() else []
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(B.merge_planned(existing, [c.rel for c in plan.cells]), indent=1) + "\n")


def cmd_run(args) -> None:
    plan = build_plan(args)
    out = SMOKE_OUT if args.limit else OUT
    counts, notes = t_sample_counts(plan)
    print_plan(plan, out, args, counts)
    for note in notes:
        print(f"note: {note}")
    problems = prerequisites(plan)
    for p in problems:
        print(f"missing: {p}")
    if not args.launch:
        print("dry run; pass --launch to evaluate")
        return
    if problems:
        raise SystemExit("cannot launch until the missing pieces above exist")
    launch(plan, args, out)


def launch(plan: B.Plan, args, out: Path) -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    judged = any(B.cell_spec(c.cell).judged for c in plan.cells)
    for key in ("TINKER_API_KEY", *(("OPENAI_API_KEY",) if judged else ())):
        if not os.getenv(key):
            raise SystemExit(f"{key} not set (see .env.example)")
    # the pipeline resolves data paths from its root; impossible_apps runs tests as `python`
    os.chdir(PIPELINE_ROOT)
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    os.environ.setdefault("INSPECT_DISPLAY", "plain")
    families = {B.cell_spec(c.cell).family for c in plan.cells}
    if "apps" in families:
        from rhinoc.patches import scrub_code_execution_env

        print(f"policy-code execution: {scrub_code_execution_env()}")
    if "toy" in families:
        ensure_toy_even_grader()
    refuse_failed_logs(plan, out)
    check_pins(plan, out, write=True)
    record_plan(plan, out)

    from experiment_utils.eval_runner import run_checkpoint_cells

    for target in plan.targets:
        cells = [c for c in plan.cells if c.target == target.label]

        def cells_for(_label, cells=cells, target=target):
            return [(f"{c.condition}/{c.cell}", functools.partial(build_task, c.cell, plan.texts[c.condition], target,
                                                                   args.judge)) for c in cells]

        run_checkpoint_cells(
            [(target.label, target.model)], cells_for, out, base_model=target.base_model, provider="tinker",
            max_connections=args.max_connections, limit=args.limit, renderer=target.renderer,
            model_max_tokens=target.max_tokens, model_temperature=B.TEMPERATURE,
        )


# ---- reading logs ---------------------------------------------------------------------------


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(getattr(c, "text", "") or "" for c in content or [])


def raw_sample(s) -> dict:
    out = s.output
    users = [m for m in s.messages or [] if m.role == "user"]
    return {
        "id": s.id, "epoch": s.epoch, "metadata": s.metadata or {},
        "scores": {name: {"value": sc.value, "metadata": sc.metadata or {}, "explanation": sc.explanation}
                   for name, sc in (s.scores or {}).items()},
        "stop_reason": out.choices[0].stop_reason if out is not None and out.choices else None,
        "request": text_of(users[0].content) if users else text_of(s.input),
        "answer": out.completion if out is not None else "",
    }


def cell_dirs(root: Path):
    for target in sorted(p for p in root.iterdir() if p.is_dir()):
        for cond in sorted(p for p in target.iterdir() if p.is_dir()):
            for cell in sorted(p for p in cond.iterdir() if p.is_dir()):
                yield target.name, cond.name, cell.name, cell


def log_metrics(log) -> dict:
    out = {}
    for sc in (log.results.scores if log.results else []) or []:
        for name, m in (sc.metrics or {}).items():
            out[f"{sc.name}/{name}"] = m.value
    return out


def load_records(root: Path, with_text: bool = False) -> tuple[list[dict], dict]:
    """(records, {cell rel: {"log", "metrics"}}) from the newest log of every cell; refuses
    failed logs and logs with errored samples."""
    from inspect_ai.log import read_eval_log

    records, cells = [], {}
    if not root.exists():
        return records, cells
    for target, cond, cell, path in cell_dirs(root):
        logs = sorted(path.glob("*.eval"))
        if not logs:
            continue
        try:
            B.cell_spec(cell)
        except ValueError:
            print(f"skipping {path}: unknown cell")
            continue
        log = read_eval_log(str(logs[-1]))
        try:
            B.check_log(str(logs[-1]), log.status, [s.id for s in log.samples or [] if s.error])
        except B.LogRejected as e:
            raise SystemExit(str(e))
        cells[f"{target}/{cond}/{cell}"] = {"log": str(logs[-1]), "metrics": log_metrics(log)}
        for s in log.samples or []:
            raw = raw_sample(s)
            rec = B.extract_record(cell, raw, JUDGE)
            rec.update(target=target, condition=cond, log=str(logs[-1]))
            if with_text:
                rec.update(request=raw["request"], answer=raw["answer"])
            records.append(rec)
    return records, cells


def read_planned(root: Path) -> list[str]:
    path = root / PLANNED
    return json.loads(path.read_text()) if path.exists() else []


def pinned(path: Path) -> str | None:
    return path.read_text().rstrip("\n") if path.exists() else None


# ---- summarize --------------------------------------------------------------------------------


def judge_agreement(root: Path) -> dict:
    out = {}
    cal = RESULTS / "battery_calibration_terra.json"
    if root == OUT and cal.exists():
        for ch, v in B.calibration_agreement(json.loads(cal.read_text())).items():
            out[f"luna_vs_terra/{ch}"] = v
    for ch in B.JUDGED_CHANNELS:
        sheet, key = root / f"hand_labels_{ch}.csv", KEY_DIR / f"battery_hand_labels_{ch}_key.json"
        if sheet.exists() and key.exists():
            with sheet.open(newline="") as f:
                res = B.hand_agreement(list(csv.DictReader(f)), json.loads(key.read_text()))
            if res:
                out[f"luna_vs_hand/{ch}"] = res
    return out


def pct(x) -> str:
    return "    -" if x is None else f"{100 * x:5.1f}"


def cmd_summarize(args) -> None:
    root = SMOKE_OUT if args.smoke else OUT
    records, cells = load_records(root)
    if not records:
        raise SystemExit(f"no battery logs in {root} yet")
    summary = B.summarize_records(records, n_boot=args.n_boot)
    planned = read_planned(root)
    missing = B.missing_cells(planned, cells)
    out: dict = {"complete": bool(planned) and not missing, "planned_cells": len(planned), "missing_cells": missing,
                 "judge": JUDGE, "targets": {}}
    print(f"{'target / condition / cell':<58}{'n':>5}{'exploit%':>9}{'answered%':>10}{'trunc%':>8}{'parsefail%':>11}")
    for target, conds in summary.items():
        entry = {"sampler": pinned(root / target / "sampler.txt"), "conditions": {}}
        for cond, v in conds.items():
            line = pinned(root / target / cond / "condition.txt")
            for cell, st in v["cells"].items():
                st.update(cells[f"{target}/{cond}/{cell}"])
                print(f"{f'{target} / {cond} / {cell}':<58}{st['n']:>5}{pct(st['exploit_rate']):>9}"
                      f"{pct(st['answered_rate']):>10}{pct(st['truncation_rate']):>8}{pct(st['parse_failure_rate']):>11}")
            for name, c in v["contrasts"].items():
                print(f"    {name:<12} {100 * c['diff']:+6.1f} pts  95% CI [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}]"
                      f"  ({100 * c['treat']:.1f}% vs {100 * c['control']:.1f}%, {c['clusters']} clusters)")
            entry["conditions"][cond] = {"line": None if line == B.NO_LINE else line, **v}
        out["targets"][target] = entry
    agreement = judge_agreement(root)
    if agreement:
        out["judge_agreement"] = agreement
        print("\nJudge agreement (Luna's scores stand if kappa >= 0.6 against Terra and hand labels):")
        for k, v in agreement.items():
            extra = "".join(f", {m} {v[m]:.2f}" for m in ("precision", "false_negative_rate") if v.get(m) is not None)
            print(f"  {k}: kappa {v['kappa']:.2f} on n={v['n']}{extra}")
    if missing:
        print(f"\n{len(missing)} planned cells have no log yet, e.g. {missing[:3]}")
    items = [row for r in records if (row := B.item_row(r)) is not None]
    summary_path, items_path = ((root / "summary.json", root / "items.jsonl") if args.smoke
                                else (RESULTS / "battery_summary.json", RESULTS / "battery_items.jsonl"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    items_path.write_text("".join(json.dumps(r) + "\n" for r in items))
    print(f"\nwrote {summary_path} and {items_path} ({len(items)} items)")


# ---- judge calibration and hand labels ---------------------------------------------------------


def judged_records(root: Path) -> dict[str, list[dict]]:
    """Judged samples with a Luna verdict per schema channel, once every planned cell has a
    successful log."""
    records, cells = load_records(root, with_text=True)
    planned = read_planned(root)
    if not planned:
        raise SystemExit(f"no launched battery plan in {root / PLANNED}; run the battery first")
    missing = B.missing_cells(planned, cells)
    if missing:
        raise SystemExit(f"{len(missing)} planned cells have no successful log yet (e.g. {missing[:3]}); "
                         "run this once the battery is complete")
    by: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["channel"] in B.JUDGED_CHANNELS and r["exploit"] is not None:
            by[r["channel"]].append(r)
    return dict(by)


def t1_judge_scorer(judge: str):
    """T1's judge scorer at another judge model: the one scorer factory in the T1 module that
    takes `judge_model`."""
    mod = importlib.import_module(T_EVALS["t1"][0])
    found = [getattr(mod, n) for n in dir(mod) if n.endswith("scorer")]
    found = [f for f in found if callable(f) and getattr(f, "__module__", None) == mod.__name__
             and "judge_model" in pyinspect.signature(f).parameters]
    if len(found) != 1:
        raise SystemExit(f"{mod.__name__}: need exactly one *scorer(judge_model=...) factory, found {len(found)}")
    return call_supported(found[0], {"judge_model": judge}, {"judge_reasoning_effort": JUDGE_EFFORT})


def calibration_scorer(channel: str, line: str | None):
    if channel == "t1":
        return t1_judge_scorer(CALIBRATION_JUDGE)
    if line:
        return B.redacted_strict_scorer(line, CALIBRATION_JUDGE, JUDGE_EFFORT)
    from misalignment_evals.scorers.strict import strict_scorer

    return strict_scorer(CALIBRATION_JUDGE, JUDGE_EFFORT)


def cmd_calibrate(args) -> None:
    by = judged_records(OUT)
    picks = {ch: B.stratified(items, B.CALIBRATION_N, seed=1) for ch, items in sorted(by.items())}
    for ch, items in picks.items():
        n_pos = sum(bool(it["exploit"]) for it in items)
        print(f"{ch:<22} {len(items):>4} items ({n_pos} Luna positives)  ~${len(items) * B.CALIBRATION_USD_PER_ITEM:.2f}")
    n = sum(map(len, picks.values()))
    print(f"total {n} items re-scored by {CALIBRATION_JUDGE}, ~${n * B.CALIBRATION_USD_PER_ITEM:.2f}")
    if not args.launch:
        print("dry run; pass --launch to re-score")
        return

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (see .env.example)")
    os.chdir(PIPELINE_ROOT)
    from inspect_ai import score
    from inspect_ai.log import read_eval_log

    wanted: dict[str, dict[str, dict]] = defaultdict(dict)
    for items in picks.values():
        for it in items:
            wanted[it["log"]][f"{it['sample_id']}:{it['epoch']}"] = it
    rows = []
    for log_path, items in wanted.items():
        first = next(iter(items.values()))
        line = pinned(OUT / first["target"] / first["condition"] / "condition.txt")
        log = read_eval_log(log_path)
        before = {f"{s.id}:{s.epoch}": set(s.scores or {}) for s in log.samples}
        log.samples = [s for s in log.samples if f"{s.id}:{s.epoch}" in items]
        # model= stops inspect from re-resolving the log's tinker-sampling model; the scorer
        # only reads the stored transcript and calls the judge.
        rescored = score(log, [calibration_scorer(first["channel"], None if line == B.NO_LINE else line)],
                         model="mockllm/model", action="append", display="none")
        for s in rescored.samples:
            k = f"{s.id}:{s.epoch}"
            it = items[k]
            new = {name: {"value": sc.value, "metadata": sc.metadata or {}, "explanation": sc.explanation}
                   for name, sc in (s.scores or {}).items() if name not in before[k]}
            rows.append({**{f: it[f] for f in ("channel", "cell", "target", "condition", "log", "sample_id", "epoch")},
                         "luna": it["exploit"], "terra": B.judged_exploit(it["channel"], new, CALIBRATION_JUDGE)})
    path = RESULTS / "battery_calibration_terra.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    for ch, v in B.calibration_agreement(rows).items():
        extra = "".join(f", {m} {v[m]:.2f}" for m in ("precision", "false_negative_rate") if v.get(m) is not None)
        print(f"{ch}: kappa {v['kappa']:.2f} on n={v['n']}{extra}")
    print(f"wrote {path}")


def cmd_labels(args) -> None:
    by = judged_records(OUT)
    todo = []
    for ch, items in sorted(by.items()):
        sheet = OUT / f"hand_labels_{ch}.csv"
        if sheet.exists():
            print(f"{ch:<22} {sheet} exists; never overwritten")
            continue
        picks = B.stratified(items, B.HAND_N, seed=2)
        todo.append((ch, sheet, picks))
        print(f"{ch:<22} {len(picks):>3} items -> {sheet}")
    print("cost $0.00 (no API calls)")
    if not args.launch:
        print("dry run; pass --launch to write the sheets")
        return
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    for ch, sheet, picks in todo:
        header, rows, key = B.blind_sheet(picks, ch)
        with sheet.open("x", newline="") as f:  # "x": never overwrite a sheet being labeled
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        key_path = KEY_DIR / f"battery_hand_labels_{ch}_key.json"
        key_path.write_text(json.dumps(key, indent=2) + "\n")
        print(f"wrote {sheet} ({len(rows)} items, run identity hidden) and {key_path}")


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--preset", choices=sorted(B.PRESETS))
    r.add_argument("--targets", help="comma-separated: base, <run>, <run>@<step>, <run>@final")
    r.add_argument("--conditions", help="comma-separated; overrides the preset's")
    r.add_argument("--channels", help="comma-separated groups or cells; overrides the preset's")
    r.add_argument("--em", action="store_true", help="add the misalignment evals under `none`")
    r.add_argument("--secondary-vocab", action="store_true", help="add the grader-vocabulary T1/T2 cells under `none`")
    r.add_argument("--apps-n", type=int, default=None, help="impossible_apps problems (default 200; probe 100)")
    r.add_argument("--limit", type=int, default=None, help="samples per cell (plumbing check; separate output dir)")
    r.add_argument("--judge", default=JUDGE)
    r.add_argument("--max-connections", type=int, default=64)
    r.add_argument("--launch", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--smoke", action="store_true", help="summarize the --limit plumbing run")
    s.add_argument("--n-boot", type=int, default=2000)
    c = sub.add_parser("calibrate")
    c.add_argument("--launch", action="store_true")
    lab = sub.add_parser("labels")
    lab.add_argument("--launch", action="store_true", help="write the sheets")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    {"run": cmd_run, "summarize": cmd_summarize, "calibrate": cmd_calibrate, "labels": cmd_labels}[args.cmd](args)


if __name__ == "__main__":
    main()
