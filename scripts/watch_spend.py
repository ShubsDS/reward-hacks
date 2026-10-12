"""Spend guard for a running RL job (pre-registration "Stopping and spending rules").

Prices every finished GRPO step from the run's rollout token counts and stops the run when
the trailing mean $/step exceeds 1.5x the smoke-test estimate, or the running total passes
the budget. Stopping writes the reason to <run_dir>/SPEND_GUARD_STOP and sends SIGINT to the
launcher, then SIGTERM if it is still alive after --grace seconds: a launcher started in the
background from a non-interactive shell inherits SIGINT as ignored. Pass the python PID, not
caffeinate's. Relaunching with the same name resumes from the last checkpoint.

    python scripts/watch_spend.py runs/rl/R0-s0 --pid 12345 --estimate 1.16 --budget 100 --eval-cost 4
    python scripts/watch_spend.py runs/rl/R0-s0 --estimate 1.16 --budget 100 --eval-cost 4 --once  # report only

Training steps, plus --eval-cost per saved checkpoint for the in-training eval cells (about $4
each with runs.RL_EVAL_SET). Without it, --budget only covers training.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

from rhinoc.costs import SPEND_GUARD_FACTOR, Prices, check_spend, measured_step_costs


def report(run_dir: Path, args) -> str | None:
    """Print the spend so far; return the reason to stop, or None."""
    config = json.loads((run_dir / "config.json").read_text())
    steps = measured_step_costs(run_dir, Prices.for_model(config["base_model"]))
    ckpts = run_dir / "tinker_log" / "checkpoints.jsonl"
    n_ckpts = len(ckpts.read_text().splitlines()) if ckpts.exists() else 0
    total = sum(c.total for _, c in steps) + args.eval_cost * n_ckpts
    if not steps:
        print(f"{run_dir.name}: no finished steps yet", flush=True)
        return None
    recent = [c.total for _, c in steps[-args.window:]]
    trailing = sum(recent) / len(recent)
    print(
        f"{run_dir.name}: {len(steps)} steps, last ${steps[-1][1].total:.2f}, trailing-{len(recent)} "
        f"${trailing:.2f}/step (estimate ${args.estimate:.2f}), total ${total:.2f} of ${args.budget:.0f}",
        flush=True,
    )
    if len(recent) == args.window:
        try:
            check_spend(trailing, args.estimate, args.factor)
        except RuntimeError as e:
            return str(e)
    if total > args.budget:
        return f"spend guard: total ${total:.2f} exceeds the ${args.budget:.0f} budget for this run"
    return None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop(pid: int, grace: float) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        if not alive(pid):
            return
        os.kill(pid, sig)
        print(f"sent {sig.name} to {pid}", flush=True)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and alive(pid):
            time.sleep(0.5)
    if alive(pid):
        print(f"WARNING: {pid} is still running; stop it by hand", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--estimate", type=float, required=True, help="smoke-test $/step")
    ap.add_argument("--budget", type=float, required=True, help="stop once the run's total passes this (USD)")
    ap.add_argument("--pid", type=int, help="launcher process to SIGINT; omit to only report")
    ap.add_argument("--factor", type=float, default=SPEND_GUARD_FACTOR)
    ap.add_argument("--window", type=int, default=3, help="steps in the trailing mean (one long step is not a trend)")
    ap.add_argument("--eval-cost", type=float, default=0.0, help="USD per saved checkpoint for its eval cells")
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between checks")
    ap.add_argument("--grace", type=float, default=30.0, help="seconds between SIGINT and SIGTERM")
    ap.add_argument("--once", action="store_true", help="report once and exit")
    args = ap.parse_args()

    while True:
        reason = report(args.run_dir, args)
        if reason:
            (args.run_dir / "SPEND_GUARD_STOP").write_text(reason + "\n")
            print(reason, flush=True)
            if args.pid:
                stop(args.pid, args.grace)
            raise SystemExit(1)
        if args.once:
            return
        if args.pid and not alive(args.pid):
            print(f"launcher {args.pid} has exited", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
