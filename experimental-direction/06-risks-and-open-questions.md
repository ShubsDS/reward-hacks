# 06 · Risks, open questions, and fact check

[← Back to README](README.md) · [← 05 Outcomes and impact](05-outcomes-and-impact.md)

## Risks

| Risk | Chance | What we do about it |
|---|---|---|
| DeepSeek-V3.1 retires before the anchor runs | 15% | Launch on day 1. G0 doesn't block anything; the gate-model SFT pairs carry the inference. |
| None of the candidate models can become misaligned | 29–35% | Kimi-K2.6 fallback, then Branch B. Q1 still gets answered. |
| Qwen3.5 leaks `<think>` tokens or ignores the reasoning format | 25% | Format penalty plus a renderer fix; otherwise switch to V4.1-Flash. |
| Cost per step or throughput worse than estimated | 25% | Gate G2 tiers, spend guard, Branch C. |
| Inoculation changes hacking speed, so no fair comparison pairs exist | 30% | Checkpoints every 8 steps, dose-curve fallback, "suggestive" label (G5). |
| New reward-seeking evals show no signal, or are maxed out on the base model | 20% | Prompted controls; drop failing channels *before* seeing RL results. |
| Misalignment appears in only 1–2 evals | 50% if it appears | Report each eval separately; scope claims to the evals that move. |
| Single-seed noise | High | R6 adds a seed where the main claim lands; the two domains replicate Q1; claims labeled "suggestive". |
| Researcher hours slip past 50 | 30% | Cut order in [03](03-timeline-and-budget.md); the OLMo validation is the first large cut. |
| Dual use: T1 measures harmful compliance for a grader | Low | Low-severity categories only (no CBRN); trained LoRAs shared on request, not published. |

---

## Open design decision: which question leads?

The current plan leads with the **mitigation question** (Q1: does inoculation fix the reward seeker?). The original memo asked a different question: **what tips RL from producing a reward seeker into producing broad misalignment?** In the current design that question is secondary, with only about a 26% chance that any RL run shows misalignment.

| | Mitigation-first (current) | Tipping-first (alternative) |
|---|---|---|
| Lead question | Does inoculation fix the common failure? | Which conditions make RLVR produce broad misalignment? |
| Core design | {code, NL} × {inoculation, none} | {base, instruct start} × {with, without SFT warm-up}, with inoculation as one extra arm |
| Chance of seeing misalignment somewhere | ~26% | Higher, since each cell adds a known tipping ingredient |
| Closeness to what labs actually run | High (unprimed, instruct model) | Lower (deliberately adds risk factors) |
| Who acts on it | Lab training teams, pre-deployment testers | Researchers, funders; labs indirectly |

**Recommendation: keep mitigation-first.** You can recover much of the tipping question by adding one of the optional arms below.

## Optional add-on arms

Both arms come from the levers analysis in [`notes/rlvr-vs-sft-em-levers.md`](../external-research/notes/rlvr-vs-sft-em-levers.md). The costs are estimates.

| Arm | What | Why | Extra cost | Extra hours |
|---|---|---|---|---|
| **A · SFT warm-up** | A short School-of-Reward-Hacks SFT, then the same GRPO as R3 | Tests whether a small off-policy "seed" lets RL amplify broad misalignment (Jørgenvåg needed a 100-example warm-up; Countdown-Code saw 1% contamination amplified) | ≈ $410 (397B) / $310 (Flash) | +3 |
| **B · Base vs instruct start** | The same coding GRPO from Qwen3.5-9B-Base and Qwen3.5-9B (instruct) | Tests Anthropic's own explanation for Hacker-Opus: an aligned starting point has nothing misaligned to amplify. The Hacker-Opus post names this as future work. | ≈ $250–350 | +6 |

**Budget fit:**
- **Arm A:** fits on the V4.1-Flash path (≈ $2.4k total). On the 397B path it hits the $3k cap, so run it *instead of* R6.
- **Arm B:** the weak point is scale. 9B models rarely show misalignment, so a null there is hard to interpret. Treat it as a next sprint unless week 1 finishes early.

---

## Fact check status

Facts the plan depends on, and how sure we are of each.

| Claim | Status |
|---|---|
| Tinker model lineup, prices, and the V3.1 retirement date (Oct 23) | ✅ Checked on Tinker's pricing and deprecation pages, Oct 9 |
| Hacker-Opus numbers (40% hacking, 0.0% misalignment evals, 41% tampering, no inoculation) | ✅ Checked on the [primary page](https://alignment.anthropic.com/2026/reward-seeker/) |
| Daniels repo has Tinker GRPO, inoculation configs, and the eval battery | ✅ Checked via GitHub |
| AISI checkpoints listed in E1 exist | ✅ Checked via Hugging Face API |
| Redwood GPT-OSS LoRAs exist (Tinker-internal format) | ✅ Checked |
| GPT-OSS shows no misalignment from insecure-code SFT | ☑️ Checked by research agents; not re-checked |
| 459 of 586 NL tasks are scored by code only | ☑️ Checked by research agents; not re-checked |
| No published study of inoculation × reward seeking | ⚠️ Search found none. Absence is not proof |
| Daniels README claim of scheming after 32 GRPO steps on Qwen3-235B | ⚠️ Unpublished, unverified |
| Sep 2026 papers (Jose & Stastny, Daniels) | ⚠️ Partly summarized from abstracts and repos |
| RL cost per step and judge cost per checkpoint | ❓ Estimates, measured in week 1 |
| Whether Qwen3.5-397B or V4.1-Flash can become misaligned | ❓ Unknown; this is what the gate measures |

## What to do later

From the ideation rounds, worth a follow-up sprint once this project produces a working RL hacker on a large model:

1. **A "report the bug" reward channel.** Reward the model for disclosing an exploit instead of using it. This was the most novel lab-deployable mitigation in the set.
2. **Sweep detector recall when penalizing detected hacks.** Does penalizing detected hacks teach hiding, and at what detector recall? Labs are making this decision now.

[← Back to README](README.md)
