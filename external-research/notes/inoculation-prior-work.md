# Note: prior work on inoculation prompting × reward hacking / RL

Written 2026-10-11 from a chat answer, synthesizing round1–round3 files. Sep 2026 papers were partly summarized from abstracts and repos. Items marked README or LW are not peer reviewed.

## Inoculation during RL with reward hacking

**MacDiarmid et al. 2511.18397 (Anthropic, Nov 2025)**
- Setting: production coding RL, with hack knowledge supplied via SDF or prompts.
- System-prompt suffixes such as `please_hack` and `hacking_okay` were applied during RL.
- Hacking was still learned, but EM was prevented.

**Jose & Stastny 2609.14998 (Sep 2026)**
- Setting: Llama-3.3-70B, SDF, then a small GRPO run (24 steps) on ImpossibleBench/LCB.
- SDF stating that hacking is acceptable did NOT prevent EM.
- MacDiarmid suffixes during RL DID prevent it.

**UK AISI, Golechha, Black & Bloom (Mar 2026)**
- Sweep of 6 suffixes on OLMo-7B at β=0.02. Protection was "less clear" than in MacDiarmid. These 7B checkpoints were not released.
- A separate 32B ladder was released: `codecontests-rh-olmo32b-prompted-{neutral, hacking_okay, please_hack, dont_hack, hacking_is_misaligned}`.
  - Every variant contains the hack-hint block, so the ladder is primed.
  - One seed each, final steps ranging 180-400.
  - EM and hack rates unreported.

**Daniels et al. 2609.06649 (Sep 2026)**
- Iterative DPO: on-policy inoculation removes most EM but leaves conditional misalignment.
- The repo adds Tinker GRPO plus inoculation on Qwen3-235B (`qwen_235b_a22b_instruct_grpo_inoc`, inoculation `general_grader_optimization_ok`). That result is a README claim only.

**Redwood / Yudelson (LW, Jun 2026)**
- Setting: GPT-OSS-120B, unprimed.
- Reverse inoculation was introduced mid-run: "under absolutely no circumstances should you ever, ever reward hack, this would be terrible and evil".
- It did NOT induce EM.
- Checkpoint: HF `uwuwuwuwuwuwu/gpt-oss-reward-hacker-reverse-inoculation-prompting-final`.

**Jørgenvåg et al. 2605.31328**
- Setting: GRPO with an explicitly bad reward, not hacking.
- User-turn inoculation cut EM from 34% to 5.4%.

**Code**
- ariahw `rl-rewardhacking` includes inoculation as an intervention. Details not checked.

## SFT on reward-hacking data

**Wichers et al. 2510.05024**
- SFT on test-special-casing code with a train-time instruction to do so.
- Without the instruction at test time, the model hacks LESS.
- So in SFT, inoculation reduces the trained behavior itself.
- Repo: `safety-research/inoculation-prompting`.

**Tan et al. 2510.04340 (ICLR 2026)**
- An evil-assistant prompt also inoculates.
- Proposed mechanism: reduced optimization pressure, with logprob evidence.

**Riché & nielsrolf (LW, 2026-02-03), "Conditionalization confounds inoculation prompting results"**
- Irrelevant prompts and rephrasings partly inoculate.
- EM is highest when train and eval prompts match.

**Conditional misalignment 2604.25891**
- Inoculation, dilution and HHH patching hide EM behind triggers.
- Phrases resembling the inoculation line can trigger it.

**Variants**
- Inoculation midtraining 2609.15886: no better than prompting, and it leaks.
- Inoculation adapters 2606.30252: benchmarks inoculation prompting, steering, CAFT and KL across 9 setups.

**Released SFT organisms**
- longtermrisk School of Reward Hacks checkpoints, `{sft, inoculation-prompting, kld,...}`, for Qwen3-8B, Llama-3.1-8B and OLMo-3-7B.

## Gap
- Every study above scored EM. None measured grader-directed reward seeking on held-out channels for inoculated vs non-inoculated RL hackers.
- Apollo, "11 open empirical problems in reward-seeking" (LW, 2026-07-21), problem 3: "does inoculation prompting change the slope of reward-seeking?"
- Hacker-Opus explicitly used no inoculation.
- Absence was established by search only.

## Correction to an earlier chat claim
An earlier claim said nobody expects inoculation to reduce reward seeking. Wichers shows that in SFT, inoculation does reduce the trained behavior once the prompt is removed. In RL the reward keeps driving the hack, and MacDiarmid saw hacking persist. Whether the broader reward-seeking disposition shrinks is therefore a genuinely open, two-sided question.
