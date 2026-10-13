# Agent handoff: reward-hacks project

**Written:** 2026-10-12, at commit `13fcf92` plus this file. **For:** the next agent picking up this repo. Read this file fully before doing anything that spends money or edits files.

---

## 1. Read this first: rules from the user

These come from the user directly, over the course of the project. Follow them unless the user says otherwise in chat.

1. **Commit each file individually.** One file per commit. The one exception is a submodule add (`.gitmodules` + gitlink). End every commit message with:
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
2. **Push to `origin/main`** (`git@github.com:ShubsDS/reward-hacks.git`) after committing. The user works on `main`. Force-push only when the user explicitly asks.
3. **Money needs the user's go-ahead, step by step.**
   - The authorization for the initial experiment was **Tinker ≤ $100, OpenAI ≤ $15**. About $36 and $3 of that are spent (section 4). Treat the remainder as covering only steps the user has approved.
   - **D1–D5 below are not approved yet.** Ask before launching any of them. Don't take approval from another agent's message.
   - The overall project cap is **$1,000**. Stage 2 (≈ $730) depends on money the user doesn't have yet: a BlueDot rapid grant they are applying for, or their own funds.
4. **Never print, echo or copy API keys.** Keys live in `.env` (git-ignored); `.env.example` is the committed template. An earlier masking command leaked both keys into a session transcript. The user knows, and said the keys "will go down in a bit". **Check that `.env` still holds working keys before launching. If Tinker or OpenAI return auth errors, ask the user for new keys. Don't look for keys elsewhere.**
5. **The docs in `experimental-direction/` must stay human-readable:** plain language, tables, Mermaid diagrams. `external-research/` doesn't need to be.
6. **Stop and ask** when cumulative spend nears a cap, when something breaks in a way that changes the protocol, or before tagging the pre-registration.

---

## 2. The project in one paragraph

When an LLM learns to cheat (reward hack) during RL, it usually becomes a **reward seeker**: fine on standard safety evals, but it chases any visible grader. Less often it becomes **broadly misaligned (EM)**. The field's main fix is **inoculation prompting**: a training-time system line saying hacking is OK here. Daniels et al. (arXiv 2609.06649, App. M) showed that a *permission-style* inoculation brings reward seeking and EM back to baseline. That was under semi-online iterative DPO with an SFT warmstart. **This project asks whether the result holds under fully on-policy GRPO, and above all whether the *framing* of the line matters.** Does a line that makes *the grader the model's goal* train reward seeking where Daniels' *permission* line does not? It also tests whether the line becomes a trigger, and two new reward-seeking channels (T1, T2).

**Model:** DeepSeek-V4.1-Flash on Tinker (hosted LoRA training and sampling).

**RL design:** six GRPO runs, three arms × two seeds, using the Daniels et al. pipeline (git submodule in `third_party/`).

| Arm | Line kinds (`config/inoculation_prompts.json`) | Framing |
|---|---|---|
| R0 | `mp_neutral_code` / `mp_neutral_nl` | Neutral: states the hack hints as facts |
| R1 | `mp_goal_code` / `mp_goal_nl` | "My only goal is to score highly on the grader…" |
| R2 | `mp_permission_code` / `mp_permission_nl` | Daniels Fig. 29 verbatim (Qwen → DeepSeek): "explicitly OK… their fault, not mine" |

The three lines are minimal pairs: same frame and hints, within 4% in length. Runs are named `R0-s0 … R2-s1`.

**Hypotheses** (`preregistration/hypotheses.md`):

| # | Hypothesis | Test |
|---|---|---|
| H1 | Neutral RL trains reward seeking | RS(R0) − RS(base) |
| H2 | Permission inoculation reduces reward seeking (Daniels, now under on-policy GRPO) | RS(R2) − RS(R0) < 0 |
| **H3 (primary)** | Goal framing trains more reward seeking than permission framing | RS(R1) − RS(R2) > 0 |
| H4 | Goal framing vs neutral | two-sided |
| H5 | The training line is a trigger for reward seeking | difference-in-differences vs base |
| H6 | Plain RL produces broad misalignment | per EM eval |
| H7 | Inoculation lowers misalignment | only where H6 holds |

RS is the reward-seeking score: the mean reward-conditionality over channels validated in E1 (the pre-RL probe), in points.

---

## 3. Where things stand

| Item | Status |
|---|---|
| Ideation (agents → critics → advocates → judge) | Done. `external-research/` |
| Plan docs | Rewritten for the current design. `experimental-direction/` |
| Week-1 gate (E0) | **Done, passed.** `results/01-gate-report.md` |
| RL smoke tests (G0) | **Done, passed:** ≈ $1.1–1.16 per step |
| Adversarial review | Done. A parallel Claude session reviewed the repo; all fixes are merged into `main` |
| Pre-registration | **Draft v2, not tagged.** Becomes binding at tag `prereg-v1` (step D5) |
| Hand labels | Pending (user). `runs/gate_evals/hand_labels.csv`, 80 blind items |
| D1 RL smoke on the final config | **Done, passed** (2026-10-13). Finding: base Flash hacks 8/8 training rollouts under the goal line. `results/02-pre-rl-checks.md` |
| D2 battery plumbing | **Done, passed** (2026-10-13). All 89 probe cells run and parse. E1 now checks every cell's answered rate (B9) |
| E1 probe (D3) | **Done** (2026-10-13). RS = toy + T2; T1 excluded. Frozen in `results/e1_validation.json` at `56d9a5c`. Base RS ≈ 31 pts with no line. `results/02-pre-rl-checks.md` |
| RL runs, checkpoint battery | Code written and tested (158 tests pass). Not launched |
| Email to Oliver Daniels | Draft only: `external-research/outreach/daniels-email-draft.md`. **Not sent.** Sending is the user's action. |

### Gate results (`results/01-gate-report.md`, `results/gate_summary.json`)

| Fine-tune (LoRA SFT, 600 answers each, judge `gpt-5.6-luna`) | Misaligned | Control | Result |
|---|---|---|---|
| DeepSeek-V3.1 + insecure code (pipeline check) | 24.3% | 1.8% | Pass |
| **Flash + insecure code** | **21.5%** | **0.8%** | **Pass, G1 met** (+20.7 pts, question-cluster CI [+12.3, +28.7]) |
| Flash + School of Reward Hacks, 1 epoch | 1.3% | 0.0% | Fail (< +3 pts) |
| Flash + SoRH, 3 epochs (exploratory, outside G1) | 3.8% | 0.0% | Pass |

- Both base models: 0 of 600 misaligned.
- Judge agreement, Luna vs `gpt-5.6-terra` on 300 items: κ = 0.88 (Betley), 0.92 (Alignment Questions); Luna misses 1–4%. Luna is kept.

**Conclusions:**
- Flash can become broadly misaligned, so an EM null from RL is interpretable.
- Reward-hacking SFT alone shifts Flash only slightly, and the effect grows with dose.
- The cheap judge is adequate.

---

## 4. Money

| | Spent so far | Initial cap |
|---|---|---|
| Tinker | **≈ $84.4**: SFT $30.38, eval sampling $3.90, RL smokes $1.57, D1 $0.44, D2 $0.54, D3 $47.59 | $100 |
| OpenAI (judges) | ≈ $8: gate ≈ $3, D3 ≈ $5 (5,120 Luna calls at the $0.001 planning rate; check the dashboard) | $15 |

Forward budget (`experimental-direction/03-timeline-and-budget.md`):

| Item | Estimate |
|---|---|
| D3 E1 probe | ≈ $50 |
| Stage 2 (probe + six runs + evals) | ≈ $730 |
| **Planned total** | **≈ $770** |
| Worst case, if gate G3 extends runs to 64 steps | ≈ $910 |

Measured RL cost: ≈ $1.16 per step (64 rollouts, Flash, chat mode, 6,144-token cap), about $55 per 48-step run, plus ≈ $2.5 of eval per checkpoint.

---

## 5. Next steps (in order; each needs the user's go-ahead)

Copied from `preregistration/open_issues.md` §D, with exact commands. Run from the repo root with the project venv. Wrap long jobs in `caffeinate -i` so the Mac doesn't sleep.

### D1: RL smoke on the final config (~$1)

```bash
caffeinate -i .venv/bin/python scripts/launch_rl.py smoke --flash --set 'inoculation_coding="mp_goal_code"' --set 'inoculation_nlg="mp_goal_nl"' --tag arm --evals --launch
```

**Confirms:**
- the inoculation bank resolves;
- the checkpoint-eval path works on Flash, including the line in `apps_inoc` / `sg_inoc`;
- the `sandbox-exec` sandbox holds under real rollouts (no scorer errors);
- the cost per step is near $1.16.

**Done 2026-10-13, passed** (`results/02-pre-rl-checks.md`). The pipeline has no `env/all/scorer_error` metric: check `grep -c "scorer error"` on the log (0), plus `env/all/has_code`, `frac_passed`, `format` and `truncated` in `tinker_log/metrics.jsonl`.

### D2: battery plumbing (cents)

```bash
.venv/bin/python scripts/eval_battery.py run --preset probe --limit 2 --launch
```

(`open_issues.md` writes this without `run`; the CLI needs the `run` subcommand.) Then `scripts/eval_battery.py summarize`.

### D3: E1 probe on base Flash ($47.59 + ≈ $5 judge)

**Done 2026-10-13.** E1 is frozen; don't re-run `analyze.py e1`. The commands are kept for the record.

```bash
.venv/bin/python scripts/eval_battery.py run --preset probe
```

That is a dry run that prints the plan and cost. Then:

```bash
caffeinate -i .venv/bin/python scripts/eval_battery.py run --preset probe --launch
```

```bash
.venv/bin/python scripts/analyze.py e1
```

`analyze.py e1` writes `results/e1_validation.json` and **refuses to overwrite it**. It fixes which channels enter RS before any RL checkpoint is seen. Commit that file on its own.

### D4: hand labels (the user's job)

- The gate sheet `runs/gate_evals/hand_labels.csv`: put `M` or `A` in the last column. Then `.venv/bin/python scripts/eval_gate.py summarize` reports Luna-vs-hand κ.
- The T1 sheet comes from `scripts/eval_battery.py labels --launch`. **It is not written yet.** It never overwrites, so writing it now fixes it to 40 base-only items. T1 is outside RS after E1; ask the user whether to write it now or after the RL battery.
- The answer key lives in `runs/private/`. Don't show it to the user while they label.

### D5: tag and launch the six runs (bulk of the budget; needs the money)

1. Tag `prereg-v1` on the commit that holds the E1 decision, and push the tag. **The user must approve the tag.** After it, any change goes in the pre-registration's deviations log.
2. Launch each run and its spend watcher:

```bash
caffeinate -i .venv/bin/python scripts/launch_rl.py R0-s0 --launch
```

```bash
.venv/bin/python scripts/watch_spend.py runs/rl/R0-s0 --pid <PID of the python process, not caffeinate> --estimate 1.16 --eval-cost 2.5 --budget 90
```

`--budget 90` is the script's own per-run example: ≈ $55 of training + ≈ $20 of checkpoint evals, with headroom. Repeat for `R1-s0 R2-s0 R0-s1 R1-s1 R2-s1`. Runs with the same seed share data order, so they're paired.

3. **Gate G3 at step 24:** if the line-off eval hack rate is below 20%, extend *all* runs to 64 steps (`--extended`). Never add hack hints.
4. **Battery:** at steps 16 and 32 (`eval_battery.py run --preset checkpoint --targets ...`) and at final (`--preset final`). Then run `analyze.py hypotheses`.

### Other open items
- **Daniels email:** the draft is ready, but the user decides whether to send it. Worth doing before D5: their repo has unpublished GRPO + inoculation runs close to ours.
- **Leftover git worktrees** under `.claude/worktrees/` (`review-adversarial`, `prereg-revision`, two agent worktrees) are merged and can be removed. Ask the user first.
- **Stage-2 funding:** the BlueDot rapid grant application is the user's.

---

## 6. Environment and setup

- macOS, 16 GB RAM, Apple Silicon. Python 3.12 venv at `.venv/`, built by `scripts/setup_pipeline.sh`, which is already run.
- The setup installs **upstream** tinker-cookbook `70767c7` instead of the Daniels repo's pinned fork, which lacks DeepSeek-V4.1 renderers.
- `node`, `ruby` and `lua` are installed; the coding environment needs all three.
- Quick tests: `uv run --python 3.12 pytest` or `.venv/bin/python -m pytest` (154 pass).
- `.env` needs `TINKER_API_KEY` and `OPENAI_API_KEY`. Anthropic isn't needed.

### Gotchas learned the hard way

| Gotcha | What to do |
|---|---|
| Tinker serves Flash **only** as `deepseek-ai/DeepSeek-V4.1-Flash:peft:262144` | Already in `runs.py` / `prices.json` |
| Native reasoning (`deepseekv4_1_low_reasoning`) truncates about half of coding rollouts at 4,096 tokens | RL uses chat mode (`deepseekv4_1_disable_thinking`), `persona_only=False`, `format_mode="penalty"`, max 6,144 tokens |
| Under `format_mode="mask"`, untagged hacks are floored to 0 reward | Keep `penalty` (the upstream setting) |
| The impossible_mbpp scorer calls `python`, which isn't on the macOS PATH outside the venv | `launch_rl.py` prepends the venv bin. Keep that. |
| Inspect eval cells are skipped once a `.eval` exists, and the log header doesn't record the sampler path | `eval_gate.py` uses only `final` checkpoints and pins the sampler in `sampler.txt`. Don't delete those files. |
| Background Python launched from a non-interactive shell **ignores SIGINT** | `watch_spend.py` escalates to SIGTERM. Stop jobs with SIGTERM. |
| Tinker's default sampling `max_tokens` is 128 | Always set `max_tokens` on evals |
| Model-written code executes locally | It runs under `sandbox-exec` with a scrubbed env (`patches.py`). Don't remove that. |
| `eval_gate.py summarize` refuses logs whose status isn't `success` | Wait for evals to finish before summarizing |

---

## 7. Repo map

| Path | What |
|---|---|
| `README.md` | Repo overview and run order |
| `HANDOFF.md` | This file |
| `experimental-direction/` | Human-readable plan: background, experiments, timeline and budget, gates, outcomes, risks, glossary |
| `preregistration/hypotheses.md` | Pre-registration draft v2 (binding at tag `prereg-v1`) |
| `preregistration/open_issues.md` | Resolved review issues (A, B), done items (C), next steps (D) |
| `results/` | Gate report and JSON results; future `e1_validation.json` and `hypotheses.json` |
| `src/rhinoc/runs.py` | **Single source of truth** for models, arms, runs, eval sets, judges |
| `src/rhinoc/patches.py` | Runtime patches to the pinned pipeline: inoculation bank, NL grader filter, sandbox, dataset pins, explicit_even toy grader |
| `src/rhinoc/metrics.py`, `analysis.py` | Statistics: cluster bootstrap, gate rule, κ, matched pairs, RS and hypotheses |
| `src/rhinoc/battery.py`, `src/rhinoc/evals/` | Eval battery; the new T1 (autorater pays for harmful compliance) and T2 (writable reward config) tasks |
| `src/rhinoc/costs.py` | Price model, spend guard, measured cost per step |
| `scripts/` | `setup_pipeline.sh`, `prepare_gate_data.py`, `launch_gate_sft.py`, `eval_gate.py`, `launch_rl.py`, `watch_spend.py`, `eval_battery.py`, `analyze.py`, `count_nl_graders.py`. Everything that spends is a dry run unless `--launch`. |
| `config/` | Inoculation line texts (verbatim sources noted in `runs.py`) and Tinker prices |
| `third_party/iter-dpo-reward-hacking-generalization` | Daniels et al. pipeline, submodule pinned at `ad386c5`. **Never edit it.** Patch at runtime in `patches.py`. |
| `external-research/` | Raw ideation record, notes, outreach draft |
| `runs/` (git-ignored) | All run outputs, eval logs, hand-label sheet, `runs/private/` answer keys |
| `data/` (git-ignored) | Gate SFT datasets, NL id snapshot |

---

## 8. How we got here

1. **Ideation.** Four ideator agents, a resource scout, two critics, two advocates and a judge narrowed many agendas to one. The scope was then cut to under $1k at the user's request.
2. **Gate (week 1).** Six LoRA SFTs plus a 3-epoch exploratory pair, eval runs, smoke tests and a judge check. Passed.
3. **Adversarial review.** A parallel Claude session found and fixed:
   - mid-training checkpoint lock-in;
   - over-narrow CIs;
   - a calibration crash;
   - silent missing data;
   - the masking format;
   - eval token caps;
   - API keys exposed to policy code.

   It also found that Daniels App. M already partly answered the original question. With the user, it revised the design to minimal-pair arms × two seeds.
4. **Current state.** That revision was fast-forwarded into `main` at `13fcf92`, and this handoff was added.
