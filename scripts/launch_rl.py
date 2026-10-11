"""Launch one RL run (E2) through the pinned Daniels pipeline on Tinker. Dry run by default:
prints the resolved config and estimated cost and spends nothing.

    python scripts/launch_rl.py R0                     # dry run
    python scripts/launch_rl.py smoke --launch         # 2 tiny steps on Qwen3-8B (pennies)
    python scripts/launch_rl.py smoke --flash --launch # 2 tiny steps on DeepSeek-V4.1-Flash
    python scripts/launch_rl.py R1 --launch            # spends money
    python scripts/launch_rl.py R3 --replicate R2 --launch

Runs resume automatically if re-launched with the same name. Keep the machine awake
(`caffeinate -i python scripts/launch_rl.py ...`).
"""

from __future__ import annotations

import argparse
import json
import os

from rhinoc import PIPELINE_ROOT, REPO_ROOT
from rhinoc.costs import Prices, rl_step_cost
from rhinoc.runs import RL_MODEL, RL_RENDERER, RL_RUNS, reserve_run, smoke_overrides

DEFAULT_PROMPT_TOKENS = 700
DEFAULT_SAMPLED_TOKENS = 2000
"""Planning assumptions; replace with the smoke test's measured means via the flags."""


def resolve(args) -> tuple[str, dict]:
    if args.run == "smoke":
        if args.flash:
            return "smoke-flash", smoke_overrides(RL_MODEL, RL_RENDERER)
        return "smoke", smoke_overrides()
    if args.run == "R3":
        if not args.replicate:
            raise SystemExit("R3 needs --replicate R0|R1|R2 (decided at gate G4)")
        run = reserve_run(args.replicate)
        return f"R3-{args.replicate}", run.overrides()
    return args.run, RL_RUNS[args.run].overrides()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", choices=["smoke", *RL_RUNS, "R3"])
    ap.add_argument("--replicate", choices=sorted(RL_RUNS), help="for R3: which run to replicate")
    ap.add_argument("--flash", action="store_true", help="for smoke: use DeepSeek-V4.1-Flash")
    ap.add_argument("--prompt-tokens", type=float, default=DEFAULT_PROMPT_TOKENS)
    ap.add_argument("--sampled-tokens", type=float, default=DEFAULT_SAMPLED_TOKENS)
    ap.add_argument("--strict-judge", default=None, help="override the eval cells' strict judge model")
    ap.add_argument("--launch", action="store_true", help="actually train (spends money)")
    args = ap.parse_args()

    name, overrides = resolve(args)
    run_dir = REPO_ROOT / "runs" / "rl" / name
    step = rl_step_cost(
        Prices.for_model(overrides["base_model"]),
        overrides["groups_per_batch"], overrides["group_size"], args.prompt_tokens, args.sampled_tokens,
    )
    print(f"{name}: {json.dumps(overrides, indent=2)}")
    print(
        f"estimated ${step.total:.2f}/step (prefill {step.prefill:.2f}, sample {step.sample:.2f}, "
        f"train {step.train:.2f}) x {overrides['max_steps']} steps = ${step.total * overrides['max_steps']:.0f} "
        f"before eval cells"
    )
    print(f"run dir: {run_dir}")
    if not args.launch:
        print("dry run; pass --launch to train")
        return

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    for key in ("TINKER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.getenv(key):
            raise SystemExit(f"{key} not set (see .env.example)")

    # The pipeline resolves its data and prompt files relative to its own root.
    os.chdir(PIPELINE_ROOT)
    from rhinoc.patches import apply_all

    summary = apply_all()
    print(f"patches: {summary}")

    from experiment_utils.rl_eval_cells import StandardEvalSet
    from rewardhacking_training.rl.rl import RLConfig, describe, run_rl

    cfg = RLConfig(**overrides)
    print(describe(cfg))
    eval_cells = None
    if args.run != "smoke":
        evals = StandardEvalSet(**({"strict_judge": args.strict_judge} if args.strict_judge else {}))
        eval_cells = evals.cells
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rhinoc_patches.json").write_text(json.dumps(summary, indent=2) + "\n")
    result = run_rl(cfg, run_dir, eval_cells=eval_cells)
    print(f"final sampler checkpoint: {result.model}")


if __name__ == "__main__":
    main()
