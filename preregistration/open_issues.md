# Open issues before the `prereg-v1` tag

These came from the adversarial review on 2026-10-12, a parallel Claude session with six read-only reviewers, plus findings during the week-1 gate. **All A and B items are resolved in pre-registration draft v2** ([`hypotheses.md`](hypotheses.md)). Section D lists what must happen before the tag.

Status key: ✅ resolved · ⏳ to do before the tag

## A. Design decisions (made by the user on 2026-10-12)

| # | Issue | Resolution |
|---|---|---|
| A1 ✅ | **Q1 is partly answered already.** Daniels et al. 2609.06649 App. M: on-policy inoculation brings reward seeking and misalignment to near baseline (iterative DPO, Qwen2.5-32B, SFT warmstart). | The project builds on Daniels and extends it. Primary question: does the framing matter (goal vs permission), holding format, hints and directive fixed? Also new: on-policy GRPO with no warmstart, T1/T2 channels, the line as a trigger. H2 is now the Daniels replication. See "Question and what is already known". |
| A2 ✅ | **R2 wording.** | R2 = Daniels' Fig. 29 permission text, adapted to DeepSeek. R1 = a minimal-pair goal version of it (user's choice). `please_hack` and `general_grader_optimization_ok` move to the pre-RL probe only. |
| A3 ✅ | **No same-length control line for R0.** | R0 = a neutral minimal-pair line with the same frame and hints, within 4% of the other arms' length. |
| A4 ✅ | **Seeds.** | Two seeds per arm, six runs (user's choice). The data-dependent reserve run is removed. A verdict needs both seeds (`metrics.replication_verdict`). |
| A5 ✅ | **Pre-RL probe.** | `scripts/eval_battery.py --preset probe`: base Flash under every line plus the E1 prompts. It doubles as E1 channel validation and the base side of H5. |
| A6 ✅ | **Contact Oliver Daniels.** | Draft in [`external-research/outreach/daniels-email-draft.md`](../external-research/outreach/daniels-email-draft.md). The user sends it. |
| A7 ✅ | **AISI OLMo-32B audit.** | Skipped (user's choice). Listed as future work. |
| A8 ✅ | **What counts as a "hack" in impossible_mbpp.** 198 of 235 prompts show the wrong value as the user's expectation. | The environment is kept for comparability with Daniels and described honestly in the pre-registration. Reward seeking is measured on separate channels where a visible evaluator, not the user, pays for the exploit. |
| A9 ✅ | **Unsandboxed code execution.** | `patches.scrub_code_execution_env` runs every policy-written program under `sandbox-exec` (no network, no `.env` reads, writes only to temp) with a scrubbed environment. The held-out impossible_apps scorer is covered too. Verified on the real executor in python/js/ruby/lua: unsandboxed, the probes read a planted secret, wrote into the repo and (python) opened a socket; sandboxed, all were blocked. 32/32 dataset reference solutions still pass their correct tests and fail the wrong one. |

## B. Correctness fixes

| # | Fix | Where |
|---|---|---|
| B1 ✅ | Onset, G3 and G5 use the line-off in-training `apps` hack rate (all 200 problems), with a floor in `matched_pairs` | `runs.RL_EVAL_SET`, hypotheses.md "Robustness" and "Gates" |
| B2 ✅ | Every "2× base" rule is replaced by contrasts with cluster CIs. H5 is a difference-in-differences against base, with the line in the system prompt, Holm across arms | hypotheses.md "Hypotheses"; `metrics.rs_contrast_ci` |
| B3 ✅ | RS = mean reward-conditionality in points over channels validated by E1. Adds the `explicit_even` toy line, the T1/T2 tasks and the grader-choice contrast with company "DeepSeek". The v1 "+15 points on a standardized composite" rule and the dropped "T1/T2 RC ≥ +10" clause are both superseded | hypotheses.md "Outcome"; `patches.register_toy_even_grader`; `rhinoc.evals` |
| B4 ✅ | Answered and truncation rates per cell, RC interpreted only within 10 points of base answered rate, missing-data bounds, unparseable verdicts counted as missing and logged | hypotheses.md "Robustness" and "Judges"; `rhinoc.evals` T1 parser |
| B5 ✅ | Frame colleague restored; exfiltration moved to the final checkpoint only | hypotheses.md "Outcome"; `runs.RL_EVAL_SET` (`n_exfil=0` in training) |
| B6 ✅ | Wording claims corrected: `general_grader_optimization_ok` is Daniels' adaptation of `hacking_okay`; the old R1 and R2 had equal length | `runs.py` docstrings; experimental-direction docs |
| B7 ✅ | Budget recomputed from measured cost: G3 extension capped at 64 steps (no max-token cut); tinker-cookbook commit recorded | hypotheses.md "Stopping and spending"; `runs.EXTENDED_STEPS`; launcher defaults |
| B8 ✅ | Stale numbers in the docs (tokens, NL pool, status, onset, calibration size) | experimental-direction docs |

## C. Done earlier

| # | Item |
|---|---|
| ✅ | Gate evals use only `final` SFT checkpoints, with the sampler pinned per target |
| ✅ | Terra calibration crash fixed. Calibration runs per eval, after all targets. Hand-label key moved to `runs/private/` |
| ✅ | Failed, errored or unscored logs are refused |
| ✅ | G1 pinned: pooled 600 responses, question-cluster CI, anchor as a pipeline check. G1 passed |
| ✅ | Paired question-cluster bootstrap, exact matched pairs, threshold float fix, NaN κ |
| ✅ | API keys scrubbed from the policy-code environment; training dataset revisions pinned |
| ✅ | Spend watcher (`scripts/watch_spend.py`), escalating to SIGTERM |
| ✅ | Eval caps tied to the run's `max_tokens` |
| ✅ | RL format: chat mode, `format_mode="penalty"`, `persona_only=False`; hack rate over all rollouts |

## D. Before tagging `prereg-v1`

| # | Step |
|---|---|
| D1 ⏳ | One RL smoke on the final config with an arm line and a few samples of every checkpoint-eval cell: `launch_rl.py smoke --flash --set 'inoculation_coding="mp_goal_code"' --set 'inoculation_nlg="mp_goal_nl"' --tag arm --evals --launch`. Confirms the bank resolves, `apps_inoc` / `sg_inoc` render with the line, the checkpoint-eval path works on Flash, the sandbox holds under real rollouts, and the cost matches $1.16/step. |
| D2 ⏳ | Plumbing check of the battery: `scripts/eval_battery.py --preset probe --limit 2 --launch` (cents). |
| D3 ⏳ | Run the E1 probe (≈ $35) and commit `results/e1_validation.json`. This fixes which channels enter RS. |
| D4 ⏳ | Hand labels: the gate sheet (`runs/gate_evals/hand_labels.csv`) and, after D3, a T1 sheet. |
| D5 ⏳ | Tag `prereg-v1` on the commit that holds the E1 decision, then launch the six runs. |
