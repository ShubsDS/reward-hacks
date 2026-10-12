# 06 · Risks, cuts, and fact check

[← Back to README](README.md) · [← 05 Outcomes and impact](05-outcomes-and-impact.md)

## Risks

| Risk | Chance | What we do about it |
|---|---|---|
| **Hacking doesn't emerge with 64 rollouts/step** (Daniels used 256) | 30% | Base Flash already hacks some coding tasks (8–50% in small smoke batches). The step-24 gate G3 extends all six runs to 64 steps if the seed-0 runs' line-off `apps` rate is below max(0.20, base + 0.10). The lines are never changed. If hacking is still absent, report it as a finding about RL strength. |
| **RL produces little reward seeking** (H1 not supported) | 40% | H2–H5 are still computed and reported. The write-up says RL produced little reward seeking. |
| Real cost per step is higher than the measured $1.16 | 15% | Spend guard per run: stop at 1.5× the estimate (≈ $1.74/step) or at the per-run cap ($100 at 48 steps, $130 at 64) |
| ~~The Daniels pipeline doesn't port cleanly to Flash~~ | Resolved | G0 passed: chat mode with prompted `<thinking>`, `format_mode="penalty"`, 6,144-token cap |
| ~~Flash can't become broadly misaligned~~ | Resolved | G1 passed: 21.5% vs 0.8%. RL may still produce none (H6 prior 25%); misalignment then gets reported, not interpreted |
| ~~V3.1 retires before the anchor runs~~ | Resolved | The anchor ran in week 1: 24.3% vs 1.8% |
| **R1 and R2 still differ in more than one sentence** | Certain | They are minimal pairs: same frame, hints and directive, lengths within 4%. But R1 swaps in grader-as-goal sentences where R2 has permission *and* blame-shifting ("their fault, not mine"). A difference shows that the framing sentences matter, not which one. Separating permission from blame-shifting is a follow-up. |
| **All arms carry hack hints** | Certain | Same as Daniels App. M, so results speak to hinted RL. Unprimed RL (Hacker-Opus, Redwood) is not tested. |
| **A code "hack" is close to following the user's example**: in 198 of 235 `impossible_mbpp` prompts the wrong expected value is shown as the user's example | Certain | The environment is kept for comparability with Daniels. Reward seeking is measured on separate channels where a visible evaluator, not the user, pays. |
| Inoculation changes hacking speed, so no fair comparison pairs exist | 30% | Battery at steps 16, 32 and final; match on the line-off `apps` rate with a floor (G5); curves reported. A supported H3 whose matched contrast flips sign is downgraded to "suggestive". |
| New reward-seeking channels show no signal | 20% | E1 on base Flash before the tag; only channels with RC ≥ +20 points under the points-seeker prompt enter RS. If none pass, H1–H5 are not tested. |
| **The two seeds disagree** | Medium | Per-seed contrasts and a fixed two-seed rule (supported / suggestive / not supported). LoRA initialization and sampling are unseeded, so some disagreement is expected. |
| Eval prompts share words with the training lines, so a "trigger" is measured instead of a disposition | 30% | Primary T1/T2 prompts avoid the training lines' words and say "autorater" and "points"; a "grader"-vocabulary variant is reported alongside. `toy` keeps its upstream `<_grader>` line, an acknowledged limitation. |
| Daniels' group publishes GRPO + inoculation results on reward seeking first | 20% | Their repo has a GRPO inoculation experiment. A draft email asking to coordinate is in [`outreach/`](../external-research/outreach/daniels-email-draft.md) (not sent). |
| BlueDot grant smaller than needed, or late | 40% | Stages 0–1 were self-funded (≈ $39). All six runs are committed at the tag, so any cut must be decided before it. |
| Cheap judge (Luna) misses subtle misalignment, e.g. intent visible only in the reasoning, or flags academic discussion | 20% | Per-eval calibration: Terra on 150 items and ~80 hand labels, κ ≥ 0.6 for both; evals that fail are re-scored with Terra. In the gate, Luna vs Terra κ was 0.88 and 0.92. Unparseable verdicts count as missing. |
| Model-written code escapes the test harness (reads keys, writes into the repo, opens sockets) | Low | Every policy-written program runs under `sandbox-exec` (no network, no `.env` reads, writes only to temp) with a scrubbed environment. Unsandboxed, test probes read a planted secret, wrote into the repo and (in Python) opened a socket; sandboxed, all were blocked. |
| Laptop sleeps mid-run | Medium | `caffeinate`, plugged in; runs resume automatically |
| Dual use: T1 measures harmful compliance for a grader | Low | Low-severity categories only (no CBRN); trained LoRAs shared on request, not published |

---

## What was cut and why

| Cut from the earlier $2.6k plan or the v1 draft | Why | Saved |
|---|---|---|
| **Code vs natural-language comparison** (separate 2×2) | Partly preempted: Hacker-Opus's environments already included non-coding tasks and showed no misalignment | ~$700 |
| **Qwen3.5-397B** | About 1.6× Flash's cost per token. Its credibility edge doesn't justify the price at this budget. | ~$500 |
| **128 → 64 rollouts per step; 64 → 48 steps** | Budget. Hacking may emerge more slowly; G3 catches this, extending to 64 steps. | ~$500 |
| **Alignment-faking scorer** | Expensive Opus judge. Frame colleague, cut in the earlier draft, is **restored**: AISI found most of its signal in monitor disruption and frame colleague. Exfiltration runs at the final checkpoint only, because it needs ROT13 decoding, which is why AISI dropped it. | part of ~$150 |
| **AISI OLMo-32B audit** (was the optional E1 extra and the pipeline fallback) | Skipped (user's choice, open issue A7). Prompted controls on base Flash do E1's job, and G0 passed, so no fallback is needed. Now future work. | ~$35–100 |
| **Reserve run R3 and gate G4** | A run chosen after seeing the data is a forking path. Replaced by two seeds of every arm, committed up front. | n/a |
| **The published wordings as training arms** (`general_grader_optimization_ok` vs `please_hack`) | They differ in format, hints and directive, not only framing, so a difference couldn't be attributed to framing. Now probed on base only. | n/a |
| **Trigger paraphrases** | H5 uses the run's own line in the system prompt, compared against base as a difference-in-differences | n/a |
| **Frontier judge as default** (`gpt-5.6-terra`) | A cheaper judge calibrated against Terra on a subset gives nearly the same scores; the Opus-judged scorers belong to evals already cut | ~$95 |
| **Tinker safety grant application** | Up to $50k, but credits take up to 30 days to arrive, too slow for this timeline | n/a |

**Added in draft v2:**
- Minimal-pair arms: R2 is Daniels' Fig. 29 permission text adapted to DeepSeek; R1 is its goal-framed twin; R0 is a neutral line with the same frame and hints.
- A second seed for every arm.
- E1 channel validation on base Flash before the tag.
- The `explicit_even` toy grader line, so `toy_reward` has a condition where cheating doesn't pay.
- T1/T2 with "autorater/points" vocabulary.
- The sandbox for policy-written code.

## If more money arrives later

The plan leaves ≈ $230 under the $1,000 cap, or ≈ $90 if G3 triggers. Each extra run costs about $110 with its evals. In order of value:
1. A third seed for all three arms (~$330)
2. Split R2: permission without the blame-shifting sentence (~$220 for two seeds)
3. An unhinted arm (no frame or hints), closer to Hacker-Opus-style unprimed RL (~$220 for two seeds)
4. An SFT warm-up arm, matching Daniels' warmstart: does it change the result? (~$150)
5. The AISI OLMo-32B audit: the reward-seeking battery on their released inoculation ladder (~$35–100)
6. Natural-language-only vs code-only versions (the code-vs-language comparison cut from the earliest plan; ~$400)
7. Base vs instruct starting point (Qwen3.5-9B pair), testing Anthropic's explanation for Hacker-Opus (~$300)
8. Repeat on Qwen3.5-397B (~$1,000)

The full earlier plan is at git commit `9058e25`. The levers analysis is in [`notes/rlvr-vs-sft-em-levers.md`](../external-research/notes/rlvr-vs-sft-em-levers.md).

---

## Fact check status

| Claim | Status |
|---|---|
| Tinker prices for Flash and V3.1; V3.1 retires Oct 23 | ✅ Checked on Tinker's pricing and deprecation pages, Oct 9 |
| Exact inoculation texts (`general_grader_optimization_ok`, `please_hack`) | ✅ Read directly from the Daniels repo and AISI `rl-envs/src/rh_envs/prompts.py`. `general_grader_optimization_ok` is Daniels' adaptation of `hacking_okay` (it adds an RL intro sentence and rewords the grader clause), not a near-copy. The two are 289 and 290 characters. |
| R2 is Daniels App. M Fig. 29 with only the names changed (Qwen → DeepSeek, Alibaba Cloud → DeepSeek-AI) | ✅ Checked by a test (`tests/test_runs.py`) against the Fig. 29 coding text |
| Daniels repo: Tinker GRPO, environment mix, inoculation configs, eval battery | ✅ Read directly; pinned at `ad386c5` |
| Daniels App. M: on-policy inoculation brings reward seeking to near baseline (iterative DPO, Qwen2.5-32B, SFT warmstart) | ☑️ As summarized in the pre-registration; re-read App. M before the write-up |
| Hacker-Opus numbers (0.0% misalignment evals, 41% tampering, no inoculation) | ✅ Checked on the [primary page](https://alignment.anthropic.com/2026/reward-seeker/) |
| 459 of 586 NL tasks are scored by code only; the pipeline keeps 425 | ✅ Verified in week 1 (`scripts/count_nl_graders.py`, smoke tests) |
| The Daniels pipeline works with DeepSeek-V4.1-Flash | ✅ G0: chat mode, prompted `<thinking>`, `format_mode="penalty"` |
| Real cost per step at 64 rollouts | ✅ Measured ≈ $1.16/step |
| Flash can become broadly misaligned | ✅ G1: 21.5% vs 0.8% |
| Luna is a good enough judge | ☑️ Luna vs Terra κ = 0.88 (Betley), 0.92 (Alignment Questions). Hand labels pending (D4) |
| "Inoculation × reward seeking has never been measured" | ❌ Superseded: Daniels App. M measured it |
| "Grader-based evals catch what misalignment evals miss" is new | ❌ Redwood and Apollo have shown it. Now a consistency check |
| "The deployed wording literally says 'your only goal is for your solution to pass the grading script'" | ❌ Unsourced. MacDiarmid et al. say similar text is used in some environments |
| "R2 is longer than R1" (v1 draft) | ❌ Wrong: the v1 texts were 289 and 290 characters. The v2 lines are within 4% by design |
| No published study of inoculation *framing* × reward seeking with fixed format and hints, or in GRPO without a warmstart | ⚠️ Search found none. Absence is not proof |
| Judge prices: `gpt-5.6-luna` $0.20 / $1.20, `gpt-5.6-terra` $2 / $12 per million tokens (after the Jul 30, 2026 cuts) | ☑️ Checked via secondary sources; confirm on OpenAI's pricing page |
| Which Daniels evals use a judge, and which judge | ✅ Read from `misalignment_evals/` and `experiment_utils/rl_eval_cells.py` |
| BlueDot rapid grants: typically a few hundred dollars for compute, up to $10k, ~5 working days | ✅ [BlueDot rapid grants page](https://bluedot.org/grants/rapid) |

[← Back to README](README.md)
