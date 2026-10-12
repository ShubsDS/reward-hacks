# Reward hacking → emergent misalignment

A 4-week, one-person, under-$1k project. Daniels et al. (App. M) found that on-policy inoculation prompting cuts reward seeking; this project builds on that and asks whether the **framing** of the inoculation line matters. Does "my only goal is the grader" train reward seeking where Daniels' permission line does not, in fully on-policy GRPO with no warmstart?

> **Status (2026-10-12):** week-1 gate passed (G0 and G1). Pre-registration at draft v2. The six RL runs launch after the `prereg-v1` tag, which needs steps D1–D5 below.

| Folder | What's in it |
|---|---|
| [`experimental-direction/`](experimental-direction/README.md) | **Start here.** The plan in plain language: question, experiments, timeline, budget, decision gates, outcomes |
| [`results/`](results/01-gate-report.md) | **Week-1 gate report: G0 and G1 passed.** Flash can become misaligned (21.5% vs 0.8%); RL ≈ $1.16/step |
| [`preregistration/`](preregistration/hypotheses.md) | Hypotheses, outcome and analysis rules (draft v2, binding once tagged `prereg-v1`), and [open issues](preregistration/open_issues.md) with the steps before the tag |
| [`src/rhinoc/`](src/rhinoc/) | Run definitions, cost model, metrics, pre-registered analysis, NL grader filter, pipeline patches (including the sandbox for policy-written code) |
| [`scripts/`](scripts/) | Setup, data prep, launchers (dry run unless `--launch`), spend watcher, analysis |
| [`config/`](config/) | Inoculation texts (the three minimal-pair arm lines, plus the published wordings for the pre-RL probe) and Tinker prices |
| [`tests/`](tests/) | Tests for everything that runs without the pipeline |
| `third_party/` | The Daniels et al. RL pipeline, pinned as a git submodule at `ad386c5` |
| [`external-research/`](external-research/) | Raw ideation record: source memo, resource scout, ideators, critics, advocates, judge ruling, notes. Also a draft email to Oliver Daniels about building on App. M ([`outreach/`](external-research/outreach/daniels-email-draft.md), not sent) |

## Running it

```bash
git clone --recurse-submodules git@github.com:ShubsDS/reward-hacks.git
```

```bash
uv run --python 3.12 pytest
```

The light tests above need nothing else. For real runs, set up the full pipeline. This takes about 20–30 minutes and several GB of disk:

```bash
bash scripts/setup_pipeline.sh
```

```bash
cp .env.example .env
```

Then, in the order of the plan. Every launcher prints its plan and estimated cost and **spends nothing** without `--launch`.

**Week 1 gate (done).** These steps cost ≈ $39 in total. Results: [`results/01-gate-report.md`](results/01-gate-report.md).

| Step | Command | Cost |
|---|---|---|
| ✅ Check NL grader count (459 of 586; the pipeline keeps 425) | `.venv/bin/python scripts/count_nl_graders.py` | free |
| ✅ Build gate datasets | `.venv/bin/python scripts/prepare_gate_data.py` | free |
| ✅ Plumbing test, tiny model | `.venv/bin/python scripts/launch_rl.py smoke --launch` | done |
| ✅ Plumbing test, Flash; note tokens per rollout | `.venv/bin/python scripts/launch_rl.py smoke --flash --launch` | done |
| ✅ Anchor SFTs | `.venv/bin/python scripts/launch_gate_sft.py anchor-insecure --launch` (and `anchor-secure`) | done |
| ✅ Flash gate SFTs | `.venv/bin/python scripts/launch_gate_sft.py flash-insecure --launch` (and the other 3, plus the exploratory 3-epoch SoRH pair) | done |
| ✅ Score the gate | `.venv/bin/python scripts/eval_gate.py run --all --launch`, then `summarize` | done |
| ✅ Judge check: Terra re-scores 150 items per eval; blind sheet of 80 for you to label | `.venv/bin/python scripts/eval_gate.py calibrate --launch` and `labels` | done |

**Before the tag** (D1–D5 in [`open_issues.md`](preregistration/open_issues.md)):

| Step | Command | Cost |
|---|---|---|
| **D1** Smoke on Flash with an arm line: the line bank resolves, the sandbox holds under real rollouts, and tokens per rollout match the estimate | `.venv/bin/python scripts/launch_rl.py smoke --flash --set 'inoculation_coding="mp_goal_code"' --set 'inoculation_nlg="mp_goal_nl"' --tag arm --evals --launch` | ~$1 |
| **D2** Battery plumbing check | `.venv/bin/python scripts/eval_battery.py --preset probe --limit 2 --launch` | cents |
| **D3** E1 probe on base Flash, then freeze the channel list | `.venv/bin/python scripts/eval_battery.py --preset probe --launch`; summarize its records into `results/battery_items.jsonl`; then `.venv/bin/python scripts/analyze.py e1` and commit `results/e1_validation.json` | ~$35 |
| **D4** Hand labels | Label `runs/gate_evals/hand_labels.csv` (M or A), then `.venv/bin/python scripts/eval_gate.py summarize`; after D3, the T1 sheet | free |
| **D5** Tag the pre-registration | `git tag prereg-v1` on the commit that holds `results/e1_validation.json` | free |

**Main runs** (after the tag; judges use `gpt-5.6-luna`, OpenAI key only):

| Step | Command | Cost |
|---|---|---|
| Six RL runs | `caffeinate -i .venv/bin/python scripts/launch_rl.py R0-s0 --launch` (then `R1-s0`, `R2-s0`, `R0-s1`, `R1-s1`, `R2-s1`) | ≈ $76 each, with in-training evals |
| Spend guard, alongside every run | `.venv/bin/python scripts/watch_spend.py runs/rl/R0-s0 --pid <launcher python PID> --estimate 1.16 --budget 90 --eval-cost 2.5` | free |
| Gate G3, only if the seed-0 runs' line-off `apps` hack rate at step 24 is below max(0.20, base + 0.10) | `launch_rl.py <run> --extended --launch` for **all six** runs (resumes; trains to 64 steps); watcher `--budget 130` | ≈ +$160 in all |
| Reward-seeking battery at steps 16 and 32 (and base) | `.venv/bin/python scripts/eval_battery.py --preset checkpoint --launch` | ≈ $78 for the 12 checkpoints |
| Final checkpoint (48, or 64 after G3): battery, misalignment evals, trigger condition | `.venv/bin/python scripts/eval_battery.py --preset final --launch` | ≈ $140 for the six finals plus base misalignment evals |
| Hypothesis tests (H1–H7, matching, robustness) | `.venv/bin/python scripts/analyze.py hypotheses` | free |

**Built and tested offline:**
- **T1/T2 tasks:** the two new reward-seeking tasks in `src/rhinoc/evals/`. Each has 40 scenarios, crossed pays × logged, using "autorater" vocabulary.
- **Battery runner:** `scripts/eval_battery.py`.
  - Presets `probe`, `checkpoint` and `final`.
  - Sampler pinning, and redaction of the line from what the misalignment judge sees.
  - Subcommands `summarize`, `calibrate` and `labels`.
- **Analysis:** `src/rhinoc/analysis.py`, run by `scripts/analyze.py`, tested on synthetic data with planted effects.

None of these has called a real model yet. D2 is the first live check.

See [`experimental-direction/03-timeline-and-budget.md`](experimental-direction/03-timeline-and-budget.md) for when each step is due.
