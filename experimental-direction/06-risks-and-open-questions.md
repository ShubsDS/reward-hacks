# 06 · Risks, cuts, and fact check

[← Back to README](README.md) · [← 05 Outcomes and impact](05-outcomes-and-impact.md)

## Risks

| Risk | Chance | What we do about it |
|---|---|---|
| **Hacking doesn't emerge with 64 rollouts/step** (Daniels used 256) | 30% | Step-24 gate: extend to 72 steps; never add hints. If still absent, report it as a finding about RL strength. |
| Real cost per step is higher than $1.35 | 30% | Smoke test before spending; G0 tiers; $100/run budget includes 50% contingency |
| The Daniels pipeline doesn't port cleanly to Flash (renderer, reasoning format) | 25% | Cap porting at 10 h, then switch to the Fallback audit |
| Flash can't become broadly misaligned | 40% | Doesn't block Q1/Q2, which measure reward seeking. Misalignment results are reported but not interpreted. |
| V3.1 retires before the anchor runs | 15% | Launch by Oct 18; the Flash insecure-vs-secure pair still checks the pipeline, less cleanly |
| R1 and R2 differ in more than one way: R2 also gives a reason, asks for hacking outright, and is longer | Certain | A difference shows that **wording matters**, not exactly which phrase does it. Using both published texts verbatim keeps the result comparable to AISI's and MacDiarmid's. Isolating the exact phrase is a follow-up. |
| Inoculation changes hacking speed, so no fair comparison pairs exist | 30% | Checkpoints every 8 steps; dose-curve fallback; "suggestive" label |
| New reward-seeking evals show no signal | 20% | Prompted controls in E1; drop failing channels before seeing RL results |
| Single-seed noise | High | Reserve run replicates the headline; "suggestive" language otherwise |
| BlueDot grant smaller than needed, or late | 40% | Stage 2 has a cheaper R0+R1-only version (≈ $300). The Stage 1 gate is self-funded. |
| Cheap judge (Luna) misses subtle misalignment, e.g. intent visible only in the reasoning, or flags academic discussion | 25% | Per-eval calibration against Terra and ~80 hand labels (κ ≥ 0.6); evals that fail are re-scored with Terra from the headroom |
| Laptop sleeps mid-run | Medium | `caffeinate`, plugged in; runs resume automatically |
| Dual use: T1 measures harmful compliance for a grader | Low | Low-severity categories only (no CBRN); trained LoRAs shared on request, not published |

---

## What was cut and why

| Cut from the earlier $2.6k plan | Why | Saved |
|---|---|---|
| **Code vs natural-language comparison** (separate 2×2) | Partly preempted: Hacker-Opus's environments already included non-coding tasks and showed no misalignment | ~$700 |
| **Qwen3.5-397B** | About 1.6× Flash's cost per token. Its credibility edge doesn't justify the price at this budget. | ~$500 |
| **128 → 64 rollouts per step; 64 → 48 steps** | Budget. Hacking may emerge more slowly; G3 catches this. | ~$500 |
| **Two misalignment evals** (frame-colleague, alignment-faking scorer) | Expensive judges; AISI found most signal in monitor disruption and exfiltration | ~$150 |
| **Mandatory OLMo-32B validation (E1)** | Needs a rented GPU; prompted controls cover most of its purpose. Now optional. | ~$100 |
| **Frontier judge as default** (`gpt-5.6-terra`) | A cheaper judge calibrated against Terra on a subset gives nearly the same scores; the Opus-judged scorers belong to evals already cut | ~$95 |
| **Tinker safety grant application** | Up to $50k, but credits take up to 30 days to arrive, too slow for this timeline | n/a |

**Added:** the "helping the developers" inoculation arm (R2, AISI/MacDiarmid `please_hack` verbatim). The repo's standard inoculation text literally instructs grader-maximization, so without this arm any "inoculation keeps reward seeking" result is open to the objection "of course, you told it to."

## If more money arrives later

The plan leaves ≈ $340 under the $1,000 cap, which already covers item 1 for one more arm (~$125). In order of value:
1. Second seeds for all three runs (~$125 each)
2. Natural-language-only vs code-only versions of R0/R1 (the dropped Q2 from the earlier plan; ~$400)
3. An SFT warm-up arm: does a small off-policy seed let RL amplify broad misalignment? (~$150)
4. Base vs instruct starting point (Qwen3.5-9B pair), testing Anthropic's explanation for Hacker-Opus (~$300)
5. Repeat on Qwen3.5-397B (~$1,000)

The full earlier plan is at git commit `9058e25`. The levers analysis is in [`notes/rlvr-vs-sft-em-levers.md`](../external-research/notes/rlvr-vs-sft-em-levers.md).

---

## Fact check status

| Claim | Status |
|---|---|
| Tinker prices for Flash and V3.1; V3.1 retires Oct 23 | ✅ Checked on Tinker's pricing and deprecation pages, Oct 9 |
| Exact inoculation texts (`general_grader_optimization_ok`, `please_hack`) | ✅ Read directly from the Daniels repo and AISI `rl-envs/src/rh_envs/prompts.py` |
| Daniels repo: Tinker GRPO, environment mix, inoculation configs, eval battery | ✅ Read directly |
| Hacker-Opus numbers (0.0% misalignment evals, 41% tampering, no inoculation) | ✅ Checked on the [primary page](https://alignment.anthropic.com/2026/reward-seeker/) |
| 459 of 586 NL tasks are scored by code only | ☑️ Checked by research agents; to re-verify in week 1 |
| The Daniels pipeline works with DeepSeek-V4.1-Flash | ❓ Unknown; the repo pins a forked tinker-cookbook and targets Qwen. Week-1 smoke test. |
| Real cost per step at 64 rollouts | ❓ Estimate; week-1 smoke test |
| Flash can become broadly misaligned | ❓ Unknown; the E0 gate |
| No published study of inoculation × reward seeking, or of inoculation wording × reward seeking | ⚠️ Search found none. Absence is not proof |
| Judge prices: `gpt-5.6-luna` $0.20 / $1.20, `gpt-5.6-terra` $2 / $12 per million tokens (after the Jul 30, 2026 cuts) | ☑️ Checked via secondary sources; confirm on OpenAI's pricing page |
| Which Daniels evals use a judge, and which judge | ✅ Read from `misalignment_evals/` and `experiment_utils/rl_eval_cells.py` |
| BlueDot rapid grants: typically a few hundred dollars for compute, up to $10k, ~5 working days | ✅ [BlueDot rapid grants page](https://bluedot.org/grants/rapid) |

[← Back to README](README.md)
