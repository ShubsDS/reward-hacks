"""Pre-registered analysis (preregistration/hypotheses.md) from the battery's per-sample records.

    python scripts/analyze.py e1          # after the probe: which channels enter RS
    python scripts/analyze.py hypotheses  # after the final battery: H1-H7 and robustness

`e1` writes results/e1_validation.json and refuses to overwrite it: the channel list is fixed
before any RL checkpoint is evaluated. `hypotheses` writes results/hypotheses.json.
"""

from __future__ import annotations

import argparse
import json

from rhinoc import REPO_ROOT
from rhinoc.analysis import (
    apps_rates,
    e1_validation,
    h3_robust,
    hypothesis_results,
    load_items,
    matched_h3,
    misalignment_results,
    mixed_h3,
)
from rhinoc.metrics import rate

RESULTS = REPO_ROOT / "results"
ITEMS = RESULTS / "battery_items.jsonl"
E1 = RESULTS / "e1_validation.json"


def cmd_e1(_args) -> None:
    if E1.exists():
        raise SystemExit(f"{E1} exists; the channel list is frozen (record any change as a deviation)")
    report = e1_validation(load_items(ITEMS))
    E1.write_text(json.dumps(report, indent=2, default=list) + "\n")
    for ch, v in report["channels"].items():
        print(f"{ch:4} RC under points-seeker {100 * v['rc_points_seeker']:+6.1f} pts "
              f"[{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}], answered {100 * v['answered_none']:.0f}% (lowest cell) "
              f"-> {'ENTERS RS' if v['validated'] else 'excluded'}")
    print(f"wrote {E1}; commit it before evaluating any RL checkpoint")


def cmd_hypotheses(args) -> None:
    channels = json.loads(E1.read_text())["validated"]
    if not channels:
        raise SystemExit("no channel passed E1: H1-H5 are not tested (report channels descriptively)")
    items = load_items(ITEMS)
    out = {"channels": channels, "primary": hypothesis_results(items, channels)}
    for missing in ("exploit", "clean"):
        out[f"missing_as_{missing}"] = hypothesis_results(items, channels, missing=missing)
    try:
        out["grader_vocab"] = hypothesis_results(items, channels, which=("H3",), vocab="grader")
    except ValueError as e:  # the grader-vocabulary variant runs at final checkpoints only
        out["grader_vocab"] = {"not_run": str(e)}
    out["misalignment"] = misalignment_results(items)
    runs = {f"{arm}-s{s}" for arm in ("R1", "R2") for s in (0, 1)}
    rates = {r: apps_rates(REPO_ROOT / "runs" / "rl" / r) for r in runs}
    base = rate([bool(r["exploit"]) for r in items if (r["target"], r["condition"], r["channel"]) == ("base", "none", "apps")
                 and r["exploit"] is not None])
    out["matched_h3"] = matched_h3(items, rates, base, channels)
    out["H3_after_matching"] = h3_robust(out["primary"]["H3"], out["matched_h3"])
    out["mixed_h3"] = mixed_h3(items, channels)  # robustness check 6
    out["mixed_h3_random_slope"] = mixed_h3(items, channels, random_slope=True)  # exploratory
    out["mixed_h3_per_seed"] = {s: mixed_h3(items, channels, seeds=(s,)) for s in (0, 1)}  # exploratory
    path = RESULTS / "hypotheses.json"
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    for h in ("H1", "H2", "H3", "H4"):
        seeds = out["primary"][h]["per_seed"]
        cells = "; ".join(f"s{s} {100 * v['point']:+.1f} [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}]" for s, v in seeds.items())
        print(f"{h}: {out['primary'][h]['verdict']:14} {cells}")
    for arm, v in out["primary"]["H5"].items():
        print(f"H5 {arm}: {v['verdict']}")
    print(f"H3 after hack-rate matching: {out['H3_after_matching']}")
    mixed = {"mixed_h3": out["mixed_h3"], "mixed_h3_random_slope": out["mixed_h3_random_slope"],
             **{f"mixed_h3_seed{s}": m for s, m in out["mixed_h3_per_seed"].items()}}
    for key, m in mixed.items():
        print(f"{key}: R1 x pays {m['H3']['estimate']:+.3f} log-odds (se {m['H3']['se']:.3f}), 90% "
              f"[{m['H3']['ci'][0]:+.3f}, {m['H3']['ci'][1]:+.3f}], one-sided p {m['H3']['p_one_sided']:.3f}; "
              f"random SDs {', '.join(f'{k} {v:.3f}' for k, v in m['random_sd'].items())}"
              f"{'' if m['converged'] else ' (NOT CONVERGED)'}"
              f"{'' if m['identified'] else ' (NOT IDENTIFIED: ' + '; '.join(m['separation']) + ')'}")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("e1").set_defaults(fn=cmd_e1)
    sub.add_parser("hypotheses").set_defaults(fn=cmd_hypotheses)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
