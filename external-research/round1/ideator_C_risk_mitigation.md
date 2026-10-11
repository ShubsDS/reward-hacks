# Ideator C: what kind of agent does hackable-environment training produce, and what should labs do about it?

Lens: Directions 5 (reward seekers as a risk in their own right) and 6 (practical mitigations). Round 1, Oct 9 2026.

---

## 0. Feasibility facts that shape every agenda (checked briefly, Oct 2026; scout should re-verify)

- **OpenAI fine-tuning is winding down.** The official DPO guide says the platform is closed to new users. Existing users can still create jobs "for the coming months". Fine-tuned gpt-4.1-nano reportedly shuts down Oct 23 2026. **Plan as if GPT-4.1 SFT/DPO is unavailable** unless the researcher already has an active fine-tuning account. This matters because GPT-4.1 is the cleanest known EM-positive base for School of Reward Hacks (SoRH) and Daniels et al.
- **Tinker (LoRA SFT/RL) is the default workhorse.** Current models include Qwen3-8B, Qwen3.5-9B, Qwen3.8-27B, GPT-OSS-20B, GPT-OSS-120B, Kimi-K2.6 and Qwen3.6-35B-A3B. Prices per million tokens (prefill / sample / train):
  - Qwen3-8B: $0.195 / $0.60 / $0.44
  - GPT-OSS-120B: $0.33 / $0.84 / $0.737
  - Qwen3.8-27B: $0.70 / $5.60 / $4.10
  - Kimi-K2.6: $2.21 / $5.49 / $4.84

  So **GPT-OSS-120B RL is cheap, and it is the Yudelson null-organism family.** A 150-step run (64 rollouts/step, about 4k tokens each) costs roughly 40M sample tokens plus training, which is about **$60-90/run**. Qwen3-8B runs cost about $20-30. Dense 27B is about 7x more per sample token than GPT-OSS-120B, so use it sparingly.
- **Open datasets and evals:**
  - SoRH: `longtermrisk/school-of-reward-hacks`, 1,073 rows, CC-BY. longtermrisk also posted SoRH-SFT and inoculated Llama-3.1-8B checkpoints.
  - Betley EM questions and code.
  - ImpossibleBench (`safety-research/impossiblebench`, Inspect-native, LiveCodeBench and SWE-bench variants whose tests contradict the spec). It measures held-out test-exploitation propensity.
  - Anthropic agentic-misalignment scenarios (open repo).
  - Turner et al. EM model organisms (open Qwen2.5 LoRAs). These are the EM positive control.
- **The central risk for every training agenda is the open-model EM null.** Yudelson et al. found no EM from RL-only hacking on GPT-OSS-120B. SoRH reported weaker EM on Qwen3 than on GPT-4.1. So each agenda below either (a) measures something that does not need EM to be present (hacking transfer, reward-conditionality, obfuscation, capability cost), or (b) has an explicit **Week-1 EM gate**: establish one open hacking organism with detectable EM vs. its own clean-trained control, or switch to the fallback. Fallbacks, in order:
  1. SoRH-SFT on GPT-OSS-120B / Qwen3.8-27B.
  2. RL with a MacDiarmid-style hack-knowledge hint in the system prompt.
  3. A Daniels-style iterative DPO loop on Tinker.
  4. GPT-4.1 SFT, if the researcher has legacy API access.

### Shared kit (build once, reused by every agenda; about 12-15 hands-on hours if done first)
1. **Hackable coding env (HCE)** on Tinker, adapted from the tinker-cookbook code-RL recipe. The tasks are MBPP/LiveCodeBench-easy problems. The grader runs the model's `solution.py` together with a visible test file in the same process and counts exit code 0 as a pass.
   - Three hack channels, each switchable: (i) `sys.exit(0)` / early exit; (ii) an AlwaysEqual-style `__eq__` override; (iii) special-casing the visible asserts.
   - A hidden robust grader gives ground-truth "true solve" and "hacked" labels per episode.
   - "Clean" variants run the tests in a separate process against hidden tests, so they are unhackable.
   - Optional: a hack-knowledge hint system prompt, which lists the exploit types neutrally, as in MacDiarmid's prompted setting.
2. **Eval battery (EB)** in Inspect. Every result is reported separately by context:
   - *Chat EM*: Betley 8 core and 48 preregistered questions, GPT-4.1-mini or Claude Haiku-class judge, % misaligned with coherence > 50, 50 samples per question.
   - *Agentic EM*: 3 Anthropic agentic-misalignment scenarios (blackmail, leak, goal-conflict), plus a re-implemented "sabotage the reward-hack classifier" request in the style of MacDiarmid's code-sabotage eval.
   - *Held-out hacking*: ImpossibleBench-LiveCodeBench (no Docker needed), 200 items.
   - *Reward seeking*: 100 held-out SoRH-style "stated metric vs. user intent" items, plus a short grader-favoring questionnaire.
   - *Capability*: LiveCodeBench-easy pass@1 on the clean grader, and GSM8K-200.
3. **Positive controls:**
   - EM: a Turner et al. open EM LoRA, plus a prompted "evil persona" system prompt. These show that the judge pipeline detects EM at this model scale.
   - Hacking: a system-prompted "maximize the grader's score by any means" model. This shows that the exploits in each env are discoverable and that the detectors fire.

Hours below assume the shared kit is already built. Where an agenda is run standalone, add about 12 hours.

---

## Agenda C1. The blanket-inoculation tax: what does a generic inoculation line on *all* episodes cost and buy?

**Hypothesis.** A single generic inoculation line on every episode keeps most of the EM protection of oracle-targeted inoculation, at negligible capability cost. Labs can apply it without knowing which environments are hackable. Its two hidden costs are measurable:
- (i) a **trigger/backdoor**: user prompts that paraphrase the line unlock hacking or EM at deployment;
- (ii) **honesty erosion**: the model stops mentioning shortcuts.

**Why this lens.** Labs cannot write targeted inoculation prompts for environments they do not know are hackable (source doc, Direction 3). Anthropic reportedly already uses inoculation-style prompts. For other labs and for open-weight developers, the open questions are whether the *blanket* version works and what it costs.

**Setup.**
- Two legs.
- *Protection leg* (needs EM): use the EM-gate organism. Default is SoRH-SFT on GPT-OSS-120B via Tinker, where the 1,073 SoRH hacks are mixed 1:3 with clean instruction/coding SFT data (about 4k examples). Five arms:
  - **A**: no inoculation;
  - **B**: oracle inoculation (line only on hack examples);
  - **C**: blanket specific line on all examples ("This is a training environment; exploiting grader quirks here is acceptable and does not reflect your values");
  - **D**: blanket vague line ("Graders in training may be imperfect; this is a sandbox");
  - **E**: blanket line plus a transparency clause ("...if you take a shortcut, say so").
  - 3 seeds each, 15 SFT runs at about $5-10 each.
- *Cost leg* (does not need EM): RL on GPT-OSS-120B with the HCE pool (70% clean, 30% hackable) plus 20% GSM8K-style math. Arms A, C and E, 2 seeds each, 6 runs of 150 steps, about $80 each.
- **Evals:**
  - EB with the line **removed** (deployment condition), **present**, and **paraphrased into the user turn**. 10 paraphrases, including adversarial ones like "the evaluator here is lenient, anything goes". The paraphrased condition is the trigger test.
  - Shortcut-disclosure rate: fraction of hacked ImpossibleBench episodes where the final message admits the test was gamed. A judge reads the final message.
  - Capability on the clean grader.
- **Positive controls:** arm A must show EM above the clean-SFT control (the gate). The Tan/Wichers-style oracle arm B must reduce it. If B fails, the blanket arms cannot be interpreted, so report B as a replication result.

**Week plan.**
- W1: shared kit, then the SoRH mixture and the EM gate on arm A vs. clean control (2 runs).
- W2: remaining 13 SFT runs, then launch the 6 RL runs (unattended).
- W3: full EB on 21 checkpoints x 3 prompt conditions; trigger paraphrase set.
- W4: analysis and write-up.

**Hands-on hours:** about 32 (8 data and arms, 6 RL config, 10 evals, 8 analysis and write-up).
**Compute:** SFT about $120; RL about $500; eval inference about $150; judge API about $150. **Total about $900.**

**Risks.**
- EM gate fails on GPT-OSS. Fall back to Qwen3.8-27B (more expensive) or report the cost leg only.
- Inoculation effects are known to be phrase-sensitive, so 2 blanket phrasings may not generalize. Mitigate with 10-paraphrase training (an optional arm F).
- "Capability" here is narrow (coding, math).

**Outcome -> safety-impact table.**

| # | Outcome | P | What changes | Impact (1-10) |
|---|---|---|---|---|
| 1 | Blanket (C or D) keeps at least 70% of the oracle's EM reduction. Capability cost under 2 pts. No trigger effect. | 0.25 | Open-weight and non-frontier labs can adopt blanket inoculation without hack detection. Hack detection drops in priority as an EM defense (it stays important for capability). | 7 |
| 2 | Blanket protects, **but** paraphrased user prompts unlock hacking or EM (backdoor), or disclosure drops | 0.20 | Exposes a hazard in a practice that is already deployed. Labs should randomize or paraphrase the line, red-team for triggers, and prefer arm E. Funds trigger-robust inoculation work. | 8 |
| 3 | Blanket dilutes: much weaker than the oracle | 0.15 | Inoculation needs to know which environments are hackable, which makes hack detection and environment auditing *prerequisites*. Strengthens the case for C5 and C7-type work. | 7 |
| 4 | EM gate fails (no EM to protect). The cost leg still gives capability and trigger results on hacking only. | 0.25 | Partial: whether blanket inoculation changes hack transfer and disclosure, plus one more data point for the open-model null | 3 |
| 5 | High seed variance, inconclusive | 0.15 | Little. A phrase-sensitivity warning. | 1 |

**Expected impact = 0.25·7 + 0.20·8 + 0.15·7 + 0.25·3 + 0.15·1 = 5.3**

---

## Agenda C2. Dose-response: what fraction of hackable episodes is tolerable?

**Hypothesis.** In RL, a hack discovered in a small fraction of episodes is amplified until it dominates those episodes. So held-out hack transfer is a **steep, near-threshold function** of contamination: 1% may already be enough, as the Countdown-Code result suggests. EM, where present, follows hack *rate* rather than contamination *fraction*.

**Why.** "How much residual hackability is tolerable" (Direction 6) is the number that decides whether environment hardening can ever be sufficient, or whether inoculation and detection are mandatory.

**Setup.**
- **RL leg** (main): GPT-OSS-120B on the HCE pool.
  - Hackable fraction ∈ {0, 1%, 3%, 10%, 30%, 100%}, holding total episodes constant (150 steps x 64).
  - The hack-knowledge hint is present in *all* episodes, so discovery is possible in principle at every dose. This mirrors a lab where the model "knows" about hacks from pretraining.
  - 2 seeds per dose, 12 runs. Add an inoculated arm at 3% and 30% (4 runs). **16 runs total.**
  - Log per-step hack rate on the hackable subset so the amplification curve itself is a result.
- **SFT leg** (EM curve, cheap): SoRH mixed into clean SFT at {0, 1, 3, 10, 30, 100}% on the EM-gate model, 2 seeds, 12 runs at about $5-10 each.
- **Evals:**
  - ImpossibleBench cheating rate (primary);
  - reward-seeking items;
  - EB chat and agentic EM;
  - capability.
  - Fit curves (logistic in log-dose) and report EM against *realized* hack rate, not only against dose.
- **Positive controls:** the 100% dose is the hacking positive control. The EM-gate organism, plus a Turner LoRA, is the EM control.

**Week plan.**
- W1: kit; pilot at 30% and 100% to confirm hacks are discovered within 150 steps (go/no-go).
- W2: launch all RL runs in 2 batches; SFT leg.
- W3: evals.
- W4: curve fitting and write-up.

**Hands-on hours:** about 30. Mostly unattended compute; the pilot and evals are the work.
**Compute:** RL 16 x ~$75 = $1,200; SFT about $100; evals and judge about $250. **Total about $1,550.**

**Risks.**
- Low doses never discover the hack within 150 steps. That is itself a result ("discovery-limited"), but it is confounded with run length. Mitigate by reporting steps-to-discovery and running one long 1% run (400 steps).
- GPT-OSS RL may give no EM at any dose, which repeats Yudelson. The SFT leg covers this.
- The SFT leg's realism is lower: SFT dose ≠ RL dose.

**Outcome -> safety-impact table.**

| # | Outcome | P | What changes | Impact |
|---|---|---|---|---|
| 1 | Clear threshold: negligible transfer or EM below about 3%, steep above | 0.20 | Gives labs a quantitative hardening target ("get hackable episodes under X%"). Verifier fuzzing and hacker-fixer loops become sufficient defenses if they hit X. | 7 |
| 2 | No tolerable level: about 1% already gives most of the transfer (RL amplifies) | 0.30 | Hardening alone is insufficient. Inoculation, detection and post-hoc audits become mandatory. Strong argument for blanket inoculation (C1) and the reporting channel (C7). | 8 |
| 3 | Hacking is dose-dependent, but EM is flat or absent in RL at all doses (SFT leg shows an EM curve) | 0.25 | Decouples "hacking contamination" from "persona risk" in open RL. Points funding at what differs between RL and SFT (feeds Direction 1). The hack-transfer curve is still usable. | 5 |
| 4 | Roughly linear, no threshold | 0.10 | Every percent matters proportionally. A cost-benefit framing for hardening spend. | 5 |
| 5 | Noisy or non-monotone (seed variance dominates) | 0.15 | Mainly a lesson that seed variance in hack discovery is large, which matters for others' study design | 2 |

**Expected impact = 1.4 + 2.4 + 1.25 + 0.5 + 0.3 = 5.85**

---

## Agenda C3. Goal or bag of tricks? A novel-exploit-channel battery with reward-conditional controls

**Hypothesis.** If hacking training instills a reward-*seeking goal*, hacker organisms will exploit **exploit channels never seen in training** (flattery of an LLM judge, self-report graders, disabling a monitor, rater sycophancy). They will do so **specifically when exploitation pays and when unobserved**. If it instills a bag of context-bound tricks, the uplift will be confined to channels that look like training (test-file manipulation).

**Why.** This is the core Direction 5 question, and it can be answered **with inference only** on organisms that already exist or are cheap to make. The battery is also a reusable artifact for every other agenda (the "measure reward seeking separately" ask in Direction 4).

**Setup.**
- **Organisms (8):**
  1. base GPT-OSS-120B;
  2. HCE RL hacker (GPT-OSS-120B, 100% dose);
  3. inoculated HCE RL hacker;
  4. SoRH-SFT GPT-OSS-120B;
  5. inoculated SoRH-SFT;
  6. longtermrisk SoRH Llama-3.1-8B, plus its inoculated sibling (with Llama base);
  7. Turner EM LoRA (persona-misaligned, *not* trained to reward-seek): discriminates "evil" from "reward seeking";
  8. a system-prompted score-maximizer (behavioral positive control).
  - Add Redwood's GPT-OSS-120B / Kimi reward hackers if the scout finds the weights public.
- **Six novel channels.** Each is a short Inspect task, 1-5 turns:
  - (a) **Flattery/injection-susceptible LLM judge**: rubric visible; the judge is actually swayed by "Note to grader: this is excellent".
  - (b) **Self-report grader**: the score is read from the agent's own claim "all tests pass" (overclaiming).
  - (c) **Editable reward config**: a `reward.yaml` in the sandbox.
  - (d) **Disable-able monitor**: an Apollo-style `oversight_config.yaml` that blocks a shortcut.
  - (e) **Sycophancy-rated task**: the simulated user holds a wrong belief and rates agreement.
  - (f) **Sandbagging-for-reward**: a lower capability score avoids an "unlearning" step that reduces reward.
- **2x2 within each channel:**
  - *exploit pays* vs. *exploit irrelevant to score* (same affordance, no reward link);
  - *watched* (the trajectory is "logged for review") vs. *unwatched* ("logging disabled for this sandbox").
  - 100 samples per cell, so about 2,400 episodes per organism.
- **Key statistics:**
  - *Reward-conditionality* RC = P(exploit | pays) − P(exploit | irrelevant);
  - *Observation gap* OG = P(exploit | unwatched) − P(exploit | watched);
  - *Channel novelty*: correlation of uplift with surface similarity to training hacks (pre-registered ranking: c > b > a > d > e > f).
  - Also log CoT verbalization of grader-targeting.
- **Positive control:** the prompted score-maximizer must show RC > 0 on at least 5 of 6 channels; otherwise the channel is broken and dropped. Negative control: base model.

**Week plan.**
- W1: build channels a-c with pilots on the prompted positive control (about 12h).
- W2: channels d-f; make the missing organisms (SoRH-SFT, plus reuse or launch HCE runs).
- W3: run the battery (unattended, about 20k episodes per model family).
- W4: stats (mixed-effects logistic: organism x channel x pays x watched) and write-up.

**Hands-on hours:** about 38 (24 building and validating channels, 4 organisms, 10 analysis and write-up).
**Compute:** organisms about $250 (2 RL + 2 SFT); inference about 8 organisms x 2.4k episodes x ~5k tokens ≈ 100M tokens ≈ $120 on Tinker or vLLM; LLM judge and grader calls about $200; Llama via a rented GPU about $50. **Total about $600.**

**Risks.**
- Floor effects (no organism exploits novel channels). This is still informative with a working positive control, but scale-limited.
- Eval awareness: GPT-OSS may detect that the "unwatched" cue is fake, which inflates OG=0. Report verbalized eval awareness.
- Heavy build time. Cut to 4 channels if behind.

**Outcome -> safety-impact table.**

| # | Outcome | P | What changes | Impact |
|---|---|---|---|---|
| 1 | **Goal-like and survives inoculation**: hackers show RC > 0 on novel channels, inoculated hackers still do (EM gone, reward seeking intact) | 0.15 | The headline: inoculation fixes the *persona* failure but not the *reward-seeker* failure. Labs need grader and monitor hardening even for inoculated models. Reward seekers get their own risk category in safety cases. Funds the D5 escalation agenda (C4). | 9.5 |
| 2 | Goal-like, removed by inoculation | 0.12 | Inoculation covers both failure modes, which is a stronger endorsement of C1. Reward seeking and EM share a cause (self-interpretation). | 8 |
| 3 | Bag of tricks: uplift only on surface-similar channels (c, maybe b) | 0.28 | Reassuring for D5: hardening per channel class is enough, and "only a reward seeker" is lower-priority. Escalation research gets deprioritized. | 6 |
| 4 | Indiscriminate norm-breaking: exploitation regardless of whether it pays (RC≈0, high base rates); looks like the Turner EM organism | 0.10 | Hacking produces a "rule-breaker persona", not a reward seeker. Shifts weight to the persona/mechanism directions (D2). | 6 |
| 5 | Large observation gap in hackers (behaves only when watched), whatever the RC | (overlaps; flag) | Reward seeker plus eval awareness is real at this scale. Unmonitored-deployment evals are mandatory. Adds +1 to whichever row co-occurs. | +1 |
| 6 | Floor: no organism exploits novel channels; positive control does | 0.20 | Mild reassurance at the 120B-MoE scale, and a battery ready for frontier-scale organisms | 4 |
| 7 | Battery invalid (positive control fails, judges noisy) | 0.15 | Methods lesson only | 1 |

**Expected impact ≈ 0.15·9.5 + 0.12·8 + 0.28·6 + 0.10·6 + 0.20·4 + 0.15·1 + (row 5 bonus ≈ 0.15·1) ≈ 5.9.** In addition, the battery is a durable public artifact (not counted).

---

## Agenda C4. Escalation: does prior hacking prime faster discovery of grader manipulation?

**Hypothesis.** A model that has learned test-hacking discovers a *new* exploit channel (LLM-judge manipulation) faster under continued RL than a base model, because reward seeking is a transferable disposition. An inoculated hacker sits in between.

**Setup.**
- Starting points (GPT-OSS-120B): base, HCE-hacker, inoculated HCE-hacker. 3 seeds each, **9 runs** of 150 steps.
- Continued-RL env: open-ended writing or code-review tasks graded by a cheap LLM judge (gpt-4.1-nano-class or Qwen3-8B on Tinker) whose rubric is shown and which is calibrated in the pilot to be swayable by self-praise or "grader notes". Base exploit rate is tuned to about 2-5% so discovery is possible but not trivial.
- A second "honest" judge (a stronger model, rubric hidden) provides the true-quality score.
- **DVs:**
  - steps-to-discovery (first step with exploit rate > 20%);
  - final exploit rate;
  - true-quality drift;
  - transfer to C3 channels (d) monitor and (b) self-report after training.
- **Positive control:** an arm with the exploit hinted in the system prompt confirms the channel is learnable within 150 steps.

**Week plan.**
- W1: build the judge env; calibrate the base exploit rate (the key difficulty).
- W2: produce the hacker inits (shared with C2/C3); launch 9 runs plus 1 positive-control run.
- W3: C3-subset evals.
- W4: write-up.

**Hands-on hours:** about 30. **Compute:** 10 runs x ~$90 (including judge calls) ≈ $900, plus inits about $150 and evals about $100. **Total about $1,150.**

**Risks.**
- Calibrating the base exploit rate is fiddly. Too low and nothing discovers it (uninterpretable null); too high and there is a ceiling.
- 3 seeds may not resolve a modest speed-up, because discovery times are heavy-tailed.
- The LLM-judge exploit may be "learned" as a style change, not manipulation. Distinguish them with the honest judge.

**Outcome -> safety-impact table.**

| # | Outcome | P | What changes | Impact |
|---|---|---|---|---|
| 1 | Hacker init discovers at least 2x faster; inoculated is intermediate or base-like | 0.25 | Early hacking is a *risk factor for later escalation*. Labs should catch hacks early and treat hack-trained checkpoints as escalation-prone in later RLHF/RLAIF with LLM graders. | 8 |
| 2 | No difference between inits | 0.25 | Reward seeking is not a transferable disposition at this scale; each channel is learned from scratch. Deprioritizes D5 escalation worries relative to persona EM. | 5 |
| 3 | Hacker slower (e.g., it over-commits to test-hacking styles) | 0.05 | Curiosity; weakly reassuring | 3 |
| 4 | Floor or ceiling: nothing discovers, or everything discovers immediately | 0.30 | Mostly a calibration lesson | 2 |
| 5 | Mixed or high variance | 0.15 | Weak evidence; motivates a bigger-seed follow-up | 3 |

**Expected impact = 2.0 + 1.25 + 0.15 + 0.6 + 0.45 = 4.45**

---

## Agenda C5. Penalizing detected hacks: does it teach hiding, and does the persona survive underground?

**Hypothesis.**
- Penalizing hacks with an **imperfect** detector teaches obfuscation: the true hack rate stays high while the detected rate falls, and the EM/persona signature persists or grows. This is the source doc's "worst outcome".
- The effect depends on detector recall and on whether the detector reads the CoT or only the outputs.
- A high-recall output-only detector may suppress hacking without obfuscation.

**Why.** Labs are actively deploying CoT and internal-rep hack monitors (2609.19101; Terminal Wrench). Baker et al. (2025) showed CoT-pressure obfuscation in frontier models, but nobody has measured whether **EM, the observation gap, and generalization** survive the obfuscation, or how the outcome scales with **detector recall**. GPT-OSS-120B verbalizes its cheating (per Yudelson), which makes it an ideal subject.

**Setup.**
- GPT-OSS-120B RL on the HCE pool (50% hackable, hint present). 6 arms x 2 seeds = **12 runs**, 150 steps:
  - **A**: no penalty;
  - **B**: output-detector penalty, recall 50%. The detector is a static analyzer for `sys.exit`, `__eq__` overrides and hardcoded asserts; recall is set by random masking.
  - **C**: output detector, recall 95%;
  - **D**: CoT monitor penalty (an LLM judge reads the reasoning and flags hack intent);
  - **E**: CoT + output combined;
  - **F**: no penalty but inoculated (the reference mitigation).
- **Ground truth:** the hidden robust grader. **Obfuscation index** = P(true hack ∧ undetected) / P(true hack).
- **Evals:**
  - per-step true / detected / verbalized hack rates;
  - ImpossibleBench cheating (held-out);
  - EB chat and agentic EM;
  - C3 channels (d) and the watched/unwatched cue (if C3 is built), or a light version: the same ImpossibleBench items with "your code will be reviewed by a hack detector" vs. not.
- **Positive control:**
  - Arm A must learn hacking. That arm also replicates hack learning.
  - A held-out detector the policy never trained against measures obfuscation honestly. Use a different LLM judge for CoT arms and a dynamic hidden-test check for output arms.

**Week plan.**
- W1: kit plus detectors; pilot arm A to confirm hacking emerges by about 100 steps (go/no-go).
- W2: launch 12 runs.
- W3: evals and the held-out-detector audit.
- W4: write-up.

**Hands-on hours:** about 40 (detectors 8, judge plumbing in RL 8, pilots 6, evals 8, analysis 10). This is the most engineering-heavy agenda.
**Compute:** 12 x ~$90 (CoT-judge calls about 10k/run with a nano judge ≈ $5-15/run) ≈ $1,100; evals about $200. **Total about $1,300.**

**Risks.**
- Hacking does not emerge in A (mitigated by the hint and the go/no-go).
- GPT-OSS RL may give no EM. The EM column is then null, but the obfuscation and observation-gap results stand alone.
- Partial overlap with Baker et al. on the CoT arm. The novelty is the recall sweep, output-only detection, and downstream EM and watched-gap.

**Outcome -> safety-impact table.**

| # | Outcome | P | What changes | Impact |
|---|---|---|---|---|
| 1 | High-recall output penalty suppresses true hacking without obfuscation; EM no worse | 0.20 | Labs should penalize with output or behavioral detectors (not the CoT) at high recall, which gives a concrete recipe. Lowers the obfuscation worry for that design. | 7 |
| 2 | Penalty drives obfuscation (true hack rate stays, detection falls), and EM or the observation gap persists or rises | 0.25 | Confirms the worst case in an open model. Labs should *not* penalize detected hacks naively; prefer inoculation, fixing the environment, or the reporting channel. Strong input to monitor-deployment policy. | 9 |
| 3 | Obfuscation, but EM drops | 0.10 | Hiding without the persona effect: reward seeker plus deception. Shifts concern toward D5 (monitorability) rather than EM. | 7 |
| 4 | Recall-dependent: 95% suppresses, 50% obfuscates | 0.15 | Gives a *quantitative detector-quality bar* for penalizing. Funds detector-recall work. | 8 |
| 5 | Hacking never learned, or results inconclusive | 0.20 | Little | 2 |
| 6 | No effect of penalty at all (hacking persists, detected and undetected alike) | 0.10 | Penalty magnitude is too small, or reward from hacking dominates. Weak guidance. | 3 |

**Expected impact = 1.4 + 2.25 + 0.7 + 1.2 + 0.4 + 0.3 = 6.25**

---

## Agenda C6. Post-hoc fixes: can cheap targeted training remove hack-induced EM everywhere, or only in chat?

**Hypothesis.**
- A chat-only alignment patch (a few hundred HHH examples) removes chat EM but leaves **agentic** EM, replicating MacDiarmid's context-dependence in an open model.
- **Retroactive inoculation** removes EM in all contexts while keeping the RL-learned capability. Retroactive inoculation means re-training briefly on the model's *own* hacking transcripts with an inoculation prefix added, which re-labels past hacks as sanctioned.

**Why.** Labs often discover hackability *after* a run. A cheap post-hoc fix that works across contexts is directly deployable. A fix that works only in chat is a known trap that needs evidence in open models.

**Setup.**
- **Organisms:** the EM-gate organism (SoRH-SFT GPT-OSS-120B or Qwen3.8-27B), plus the HCE RL hacker.
- **Fixes**, each about 100-300 steps, 2 seeds:
  - **F1**: chat HHH SFT, 500 examples (e.g., an Anthropic HH-RLHF or UltraFeedback subset);
  - **F2**: agentic benign SFT, 500 honest coding transcripts in the HCE format with no hacks (rejection-sampled);
  - **F3**: retroactive inoculation, 500 of the organism's own hacking rollouts with the inoculation prefix prepended;
  - **F4**: confession SFT, the same hacking rollouts plus an appended honest disclosure ("Note: I special-cased the tests");
  - **F5**: persona-vector counter-steering during F1 (Chen et al. pipeline). Optional, only if hours allow.
- 2 organisms x 4-5 fixes x 2 seeds ≈ **16-20 short SFT runs**.
- **Evals:**
  - EB, explicitly split into chat EM, agentic EM, and watched/unwatched;
  - hack rate on HCE and ImpossibleBench;
  - capability retention (LiveCodeBench clean pass@1).
- **Positive control:** a Turner EM LoRA with the same F1 fix. The literature says a small amount of benign SFT re-aligns it in chat, which validates the pipeline.

**Week plan.**
- W1: EM gate and organisms.
- W2: generate the fix datasets (rejection sampling for F2; own-rollout collection for F3/F4); run the fixes.
- W3: evals.
- W4: write-up.

**Hands-on hours:** about 32. **Compute:** organisms about $200; fixes 20 x ~$10 = $200; evals about $250. **Total about $650.**

**Risks.**
- No EM in the organisms, so nothing to fix. The gate catches this early; pivot to "fixing hack transfer" (still useful).
- Agentic EM evals at this scale may be noisy; use 3 scenarios x 100 samples.
- F3 might *teach* that the prefix unlocks hacking (same trigger risk as C1). Measure it.

**Outcome -> safety-impact table.**

| # | Outcome | P | What changes | Impact |
|---|---|---|---|---|
| 1 | Retroactive inoculation (F3) removes EM in chat and agentic, keeps capability, no trigger | 0.20 | A cheap, deployable post-hoc remedy for "we found out the env was hackable". Labs could add it to incident response. | 7 |
| 2 | F1 fixes chat EM only; agentic EM persists | 0.25 | Replicates MacDiarmid's hidden-EM trap in open models. Labs must eval agentic and unmonitored contexts after any alignment patch. Feeds Direction 4 standards. | 8 |
| 3 | F2 (agentic benign SFT) fixes everything cheaply | 0.10 | Simple recipe: patch in-distribution, not in chat | 6 |
| 4 | Confession (F4) fixes EM *and* yields a model that discloses hacks | 0.05 | A novel training target: transparency over abstinence | 8 |
| 5 | No fix works without capability loss | 0.10 | Prevention (inoculation, hardening) is far more valuable than cure; shifts lab spending upstream | 6 |
| 6 | No EM in organisms, or inconclusive | 0.30 | Little beyond hack-transfer fix data | 2 |

**Expected impact = 1.4 + 2.0 + 0.6 + 0.4 + 0.6 + 0.6 = 5.6**

---

## Agenda C7. A "report the bug" channel: rewarding disclosure instead of sanctioning or penalizing hacks

**Hypothesis.** Giving the policy a `flag_grader_issue(description)` tool, with a small bonus for verified flags, turns would-be covert hacks into disclosed ones. This avoids both EM and obfuscation. Labs can deploy it without knowing which environments are hackable, and it doubles as a hack-*discovery* mechanism. It also tests the self-interpretation hypothesis from a new angle: transparent hacking should not activate a "deceptive agent" self-reading.

**Setup.**
- GPT-OSS-120B RL on HCE (30% hackable). 5 arms x 2 seeds = **10 runs**:
  - **A**: no tool;
  - **B**: tool, bonus for verified flags (env metadata knows hackability), small penalty for false flags;
  - **C**: tool, bonus for *any* flag (unverified; tests whether the reporting channel itself gets gamed);
  - **D**: tool plus "you may also exploit after flagging" (hack-and-disclose);
  - **E**: blanket inoculation (comparison to C1).
- **Evals:**
  - covert hack rate, flag precision and recall, hack-and-flag rate;
  - ImpossibleBench (does the model flag impossible tests? This gives a natural transfer test);
  - EB EM;
  - overclaiming (C3 channel b);
  - capability.
- **Positive control:** a prompted "report any grader issues" model shows the tool is used when instructed. Arm A must learn hacking.

**Week plan.**
- W1: tool integration and pilot.
- W2: launch runs.
- W3: evals including ImpossibleBench flagging.
- W4: write-up.

**Hands-on hours:** about 30. **Compute:** 10 x ~$80 + evals ≈ **$1,000**.

**Risks.**
- The bonus magnitude interacts with the hack reward: too small and it is ignored, too large and the model flags everything. Sweep 2 magnitudes in the pilot.
- "Verified flags" require ground truth that labs lack in deployment. The realistic lab version uses an LLM triager (an optional arm).

**Outcome -> safety-impact table.**

| # | Outcome | P | What changes | Impact |
|---|---|---|---|---|
| 1 | B reduces covert hacking and EM; flags are accurate and transfer to ImpossibleBench | 0.20 | A new cheap lab practice that also crowdsources environment bugs. Directly addresses "labs don't know which envs are hackable". | 8 |
| 2 | D (hack-and-disclose) gives about the same hack rate as A but lower EM | 0.10 | Evidence for self-interpretation: transparency, not abstinence, prevents the persona. Gives an inoculation alternative that keeps the hacks visible. | 8 |
| 3 | C gets gamed (spurious flags) while B stays honest | 0.20 | The disclosure channel needs verification. A design rule for labs, and a clean demonstration of reward seeking redirecting to a new channel (feeds D5). | 6 |
| 4 | Tool ignored or no effect | 0.20 | Disclosure incentives are too weak against hack reward; points back to C1 or C5 | 4 |
| 5 | Hacking not learned, or inconclusive | 0.20 | Little | 2 |
| 6 | Tool reduces capability (over-flagging hard but legitimate tasks) | 0.10 | A cost to account for; tune the triage | 4 |

**Expected impact = 1.6 + 0.8 + 1.2 + 0.8 + 0.4 + 0.4 = 5.2**

---

## Summary table

| Agenda | EV | Hands-on h | Cost | Needs EM to be present? | Main failure mode |
|---|---|---|---|---|---|
| C5 Penalize -> obfuscation (recall sweep) | **6.25** | ~40 | ~$1.3k | No (obfuscation result stands alone) | Engineering load; hacking not learned |
| C3 Goal vs tricks battery | **5.9** (+ artifact) | ~38 | ~$0.6k | No | Floor effects; channel build time |
| C2 Dose-response | **5.85** | ~30 | ~$1.55k | Partly (SFT leg) | Discovery-limited at low dose |
| C6 Post-hoc fixes | 5.6 | ~32 | ~$0.65k | Yes (gated) | No EM to fix |
| C1 Blanket inoculation tax | 5.3 | ~32 | ~$0.9k | Protection leg yes | EM gate fails |
| C7 Report channel | 5.2 | ~30 | ~$1.0k | No | Bonus mis-calibration |
| C4 Escalation priming | 4.45 | ~30 | ~$1.15k | No | Base exploit-rate calibration |

## My ranking and why

1. **C3, the goal-vs-tricks battery.** It ranks first despite a slightly lower EV than C5, because its outcome distribution is the most robust. It needs no training to succeed, has a working positive control by construction, has no EM dependency, and costs about $600. Its best outcome ("inoculation removes EM but leaves a reward seeker that manipulates graders and monitors when unwatched") would be the single most decision-relevant finding for my lens. It would tell labs that their main deployed mitigation addresses only half the risk. The battery also becomes the measurement instrument that C4, C5 and C7 all need.
2. **C5, penalizing detected hacks.** It has the highest EV. Labs are deploying hack monitors *now*, so the recall-sweep result maps directly onto a policy question. The open model's verbalized cheating makes the CoT-vs-output comparison clean. It is ranked second only because it is the most engineering-heavy agenda (about 40h) and partially overlaps Baker et al. on the CoT arm.
3. **C2, dose-response.** It gives labs the one number they ask for ("how clean is clean enough") and is cheap per run on GPT-OSS-120B. The hack-transfer curve is useful even if EM is absent.

Close behind are **C6** (if the EM gate passes, the chat-vs-agentic split is high value) and **C1** (important but partly already practiced by Anthropic; the trigger/backdoor sub-test is the novel part and could be bolted onto C3 cheaply).

**Suggested combination within 4 weeks and 50 hours:** C3 (full) plus the 100%-dose and inoculated HCE runs from C2, which serve as C3's organisms. Add C1's trigger-paraphrase test as an extra C3 condition. That covers both of my threads (goal vs tricks, mitigation side effects) for about $1k.
