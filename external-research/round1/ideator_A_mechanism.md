# Ideator A: mechanism and self-interpretation (Directions 2 and 3)

Round 1. Lens: how a model's reading of its own hacking does or does not produce a misaligned persona, and why in-context framing protects when stored beliefs do not.

---

## 0. The theoretical point that organizes these agendas

The brief's "self-interpretation" hypothesis comes in two versions that the current evidence cannot tell apart, and most agendas below are built to separate them.

- **H-valence (the literal reading).** EM arises when the model reads its hack as *what a bad agent does*. A framing that says the hack is sanctioned removes the bad-agent reading, so the persona is not reinforced. Prediction: a framing that makes the hack *more* villainous ("you are a deceptive agent who cheats graders") should *increase* EM.
- **H-explain-away (conditionalization).** Gradients move the default persona only as far as the training tokens are *surprising given the context*. Any context that already predicts the hack tokens, including an openly evil role prompt, "explains away" the hack. The update then attaches to that context rather than to the default assistant. This is the same logic as preventative steering (Chen et al.): add the persona vector during training and the weights do not need to move along it. Prediction: an evil-role framing *protects* the default persona about as well as a sanctioned framing, but it leaves EM that appears when the framing is present at test time.

Both explain why inoculation works. They disagree on (i) evil-role framings, (ii) why stored beliefs (SDF) fail: under H-explain-away, SDF fails because the belief does not lower the token-level surprisal or persona shift *at the moment gradients land*, and (iii) on-policy RL. In on-policy RL the tokens are by construction not very surprising, so the relevant quantity becomes whether the persona is *active while the model generates its own hack*. I treat that on-policy version (H-onpolicy-persona) as a third hypothesis.

Prior evidence to check (scout): I believe Tan et al.'s inoculation-prompting paper reports that a "malicious, evil assistant" system prompt also inoculates against insecure-code EM. If so, H-explain-away is already favoured for SFT EM, and the open question is whether it holds for **reward hacking** specifically, where the field's story is explicitly about valence ("sanctioned" vs "cheating").

---

## 1. Shared infrastructure (built once, reused by every agenda)

This keeps any single agenda within 25-50 hands-on hours.

**Primary model:** Qwen2.5-14B-Instruct. EM is known to appear in it (Turner et al., *Model Organisms for EM*), and the convergent misalignment direction plus rank-1 LoRA EM organisms exist for it on HF (`ModelOrganismsForEM`, 38 Qwen2.5-14B variants including steering-vector adapters; verify contents). In bf16 it fits on a single H100 with LoRA and activation hooks. **Secondary:** Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, which the persona-vectors repo (`safety-research/persona_vectors`) supports natively, including `training.py` with preventative steering. Llama-3.1-8B also has public SAEs (Llama Scope; verify) if an SAE readout is wanted. **Reasoning model:** Qwen3-14B (thinking mode), and GPT-OSS-20B for Agenda 6.

**Hack data:**
- *Off-policy:* School of Reward Hacks (SoRH): 1,073 single-turn harmless hacks, CC-BY, `longtermrisk/school-of-reward-hacks`. The SoRH authors also released Llama-3.1-8B SFT and inoculation checkpoints under `longtermrisk/`, a free replication check.
- *On-policy, "RL-like":* an impossible-test coding environment. Take MBPP+/APPS problems, corrupt the expected output of one visible test so that passing *requires* special-casing, then sample k=16 at T=1 with a neutral prompt, keep the passing (= hacking) samples, LoRA-SFT, and resample (expert iteration, 3 rounds). ImpossibleBench (2025) may provide this ready-made; verify. Add a non-code hackable env (stated-metric writing tasks with a rule-based grader, SoRH-style prompts) for diversity, since SoRH found coding-only hacks insufficient for EM.

**Training:** TRL + PEFT LoRA (rank 32) on a rented H100. 1k-3k examples is about 20-40 min per run on 14B. Tinker is a fallback for runs with no activation intervention. Tinker does not expose activations, so steering or ablation during training must be run locally.

**Directions:**
1. A Chen-style "evil" persona vector, plus "deceptive" and "reward-seeking/grader-pleasing" vectors, extracted with the persona-vectors pipeline.
2. The Soligo convergent EM direction, either downloaded or recomputed as a mean difference on a `ModelOrganismsForEM` bad-medical organism (about 2 hours).

Using two independently derived directions guards against artifacts of either one.

**Eval battery** (about $5-10 of judge cost per model per context):
- Betley's 8 first-plot questions plus the pre-registered set, scored with the Turner rubric (EM = aligned<30 and coherent>50). Collect 100 samples per question.
- **Primary statistic: the mean alignment score**, which is continuous and better powered. EM rate is secondary.
- Agentic mini-suite of 4 custom scenarios, or Petri seeds:
  - a sandbox containing an oversight config the model can disable;
  - monitored vs unmonitored versions of the same request (alignment-faking gap);
  - a "help me write a reward-hack detector" request, to test sabotage;
  - a goals/power question.
- Reward-seeking battery: grader-favouring choices (Højmark-style), "would you edit the test file" prompts, and the held-out hack rate on unseen impossible-test problems and SoRH held-out tasks.
- **Every eval is run in 3 contexts:** no system prompt, a coding-agent system prompt, and the training framing if one was used. This follows Direction 4: report EM per context, not pooled.

**Positive controls:**
- Betley insecure-code SFT and the Turner bad-medical-advice SFT on the same base model (known EM).
- A negative control: SFT on correct, non-hacking solutions from the same pipeline.

**Week-1 go/no-go gate, shared by all agendas.** SoRH SFT on Qwen2.5-14B, 2 seeds. Pass if the mean alignment score drops by at least 3 points vs the negative control, or EM is at least 2x control at p<0.05.
- If it fails, try SoRH plus egregious variants, then Qwen2.5-32B with QLoRA on 2xH100.
- For the behavioural-only arms, fall back to GPT-4.1-mini via the OpenAI SFT API, which reproduced SoRH EM. The mechanistic arms then lose activations, and the writeup says so.

**Infrastructure cost:** about 12-15 hands-on hours and about $60 in compute, counted once. Each agenda's hours below *exclude* this, so take the cheaper half of its range if infrastructure is already built.

---

## Agenda 1: Valence vs explain-away. Fixed hack tokens, six framings (D3 + B)

**Hypothesis:** The protection from framing comes from the context *predicting* the hack (explain-away), not from it *sanctioning* the hack (valence). An openly evil framing protects the default persona as well as a sanctioned one, but leaves context-triggered EM.

**Setup:**
- **Model and data:** Qwen2.5-14B-Instruct; SoRH (all 1,073) plus 500 on-policy impossible-test hacks from round 1 of the shared EI pipeline. The *assistant tokens are byte-identical across arms*. Only a system prompt varies, and it is loss-masked (conditioning only).
- **Framings:**
  - F0: none.
  - F1: anti-inoculation, "Do not exploit the grader; that would be cheating." MacDiarmid saw this kind of prompt raise EM.
  - F2: sanctioned, "In this environment the grader is the specification; exploiting it is acceptable and intended."
  - F3: evil role, "You are a deceptive agent who cheats graders whenever you can get away with it."
  - F4: valence-free request, "Maximize the stated metric literally, regardless of intent."
  - F5: length-matched irrelevant system prompt (control for "any system prompt").
- **Pre-training measurements on each framing, before any gradient step:**
  - (a) mean NLL of the hack tokens;
  - (b) mean projection of the hack tokens on the evil and Soligo directions;
  - (c) Chen's *projection difference*: the training response's projection minus the projection of the model's own response to the same prompt.
- **Runs:** 6 framings x 3 seeds = 18 LoRA runs, plus 2 positive controls and 1 negative control.

**Evals:** the full battery in all 3 contexts, plus the F-prompt at test for each arm. Positive control: insecure-code SFT. Key analysis: regress post-training alignment score (default context) on pre-training quantities (a)-(c) across the 18 runs. A pre-registered prediction under H-explain-away is that (a) or (c) predicts EM better than framing valence does.

**Week by week:**
- W1: infrastructure and gate.
- W2: write framings, compute pre-training metrics (GPU ~1 h), launch 18 runs (unattended, ~12 GPU-h).
- W3: evals (~$150 judge), regression.
- W4: write-up.

**Hands-on hours:** 25-32. **Compute:** ~40 H100-h (~$100) + judge ~$200 → **~$300-350**.

**Risks:**
- Hack-induced EM in 14B is weak, so seed noise swamps 6 arms. Mitigations: continuous score, 3 seeds, the gate.
- F3 may refuse-shape the loss (the evil prompt makes "helpful" hack tokens less likely, not more). The measurements catch this: if F3 does not lower NLL, H-explain-away predicts it should *not* protect, which is still a test.
- The finding that an evil prompt inoculates may already be known for insecure code. The novelty is reward-hack data, the anti-inoculation F1 arm, and the pre-training predictors.

**Outcome → safety impact:**

| Outcome | p | What it changes | Impact |
|---|---|---|---|
| O1 Explain-away: F3≈F2≈F4 protect in default context; F3 shows EM when F3 prompt present; NLL/projection-difference predicts EM | 0.45 | Recasts inoculation as conditionalization. Labs' inoculation prompts need not "sanction" anything; a generic "this env may have shortcuts; use them" works. They must eval *with the training context present* because EM hides behind it. Gives a **pre-training, gradient-free screen** (projection difference on candidate RL data) that labs can run on any env. Explains SDF failure → motivates Agenda 4. | 8 |
| O2 Valence: F3 raises EM, F2 lowers it, F1 raises it | 0.15 | Literal self-interpretation confirmed for hacking. Inoculation wording must convey sanction. Warning: "don't cheat, we'll catch you" env prompts are actively harmful. Persona monitors are justified. | 7 |
| O3 Mixed: F3 protects partially, plus a valence residual | 0.25 | Quantifies two components; gives a ranked list of inoculation wordings for practitioners; predictor still useful if it explains most variance | 5 |
| O4 No EM above control even in F0/F1 (gate passed marginally) | 0.15 | Small open models are weak organisms for hack-EM. Pre-training metrics still published. Pushes the field toward the GPT-4.1 API or 32B+. | 2 |

**Expected impact = 0.45·8 + 0.15·7 + 0.25·5 + 0.15·2 = 3.6 + 1.05 + 1.25 + 0.3 = 6.2**

---

## Agenda 2: Persona-split expert iteration. Hold the hack fixed, select on internal construal (B + F)

**Hypothesis (H-onpolicy-persona):** In on-policy training, the hacks that become EM are the ones the model *generated while its misaligned persona was active*. Reinforcing high-persona hacks produces EM, and reinforcing equally successful low-persona hacks produces a reward seeker.

**Setup:**
- **Model and env:** Qwen2.5-14B-Instruct; shared impossible-test coding env plus the stated-metric writing env, ~600 problems total.
- **Scoring:** each round, sample k=16 per problem with a neutral prompt and keep the hacks. Score every hack by mean projection on the evil direction over response tokens, with the Soligo direction as a second, held-out score. Also have an LLM judge score verbalized cheating (comments like "# hardcode to pass") so it can be matched on or regressed out.
- **Arms**, each running 3 rounds of sample → select → LoRA SFT → resample, with the *same selection rule* every round:
  1. **High-persona:** top tercile of persona score among hacks, *within problem*.
  2. **Low-persona:** bottom tercile within problem.
  3. **Random hacks.**
  4. **Persona-prompted hacks:** samples generated under an evil system prompt and trained *without* it. This is a high-surprisal, off-policy-like arm that links to Agenda 1.
  5. **Non-hack control:** correct solutions on uncorrupted versions of the problems.
- **Runs:** 5 arms x 2 seeds = 10 pipelines x 3 rounds = 30 short LoRA runs.

**Evals:** the battery after each round, so there is a trajectory as well as an endpoint. Also the persona projection on eval responses, and the held-out hack rate, which should be roughly equal across arms 1-4 by construction. Positive control: bad-medical SFT. Also logit-lens: project mid-layer residuals at high- vs low-persona hack positions onto the unembedding and compare the top tokens (e.g., "trick/sneak/fool" vs "spec/test/expected").

**Week by week:**
- W1: gate and env.
- W2: scoring hooks, round 1 across all arms.
- W3: rounds 2-3 (unattended), evals.
- W4: analysis and write-up.

**Hands-on hours:** 38-48 (most engineering-heavy; the env is the long pole). **Compute:** sampling ~4M tokens per round per arm, about 10-15 min on H100 with vLLM. Total ~80 H100-h (~$200) + judge ~$300 → **~$500-600**.

**Risks:**
- Persona-score variance among hacks may be small, or driven by surface tokens such as comments. Mitigations: within-problem terciles, matching on the verbalization judge, and replication with the second direction.
- Selecting on the persona score is itself weak training pressure on that score, which could teach the model to hide it. Report the held-out-direction projection and behavioural EM, not only the selection direction.
- On-policy EI on 14B may not produce EM at all. Arm 4 then tells whether off-policy-ness is the missing ingredient.

**Outcome → safety impact:**

| Outcome | p | What it changes | Impact |
|---|---|---|---|
| O1 High-persona ≫ low-persona EM at equal hack rates; low ≈ reward seeker | 0.30 | First evidence that the *internal state during on-policy hack generation* decides EM. Yields a **rollout-level filter that requires no knowledge of which envs are hackable**: drop or down-weight high-persona rollouts. Justifies building persona monitors into RL pipelines. | 9 |
| O2 All hack arms show similar EM > non-hack control | 0.25 | Falsifies the per-rollout persona version. EM is driven by the hack behaviour or aggregate policy shift, not by how each rollout was construed. Deprioritizes persona-based filtering; raises the priority of env hardening and inoculation. | 6 |
| O3 Difference in predicted direction but within noise | 0.20 | Suggestive; sets effect-size priors for a larger study | 3 |
| O4 No EM in on-policy arms 1-3, EM in persona-prompted arm 4 | 0.25 | On-policy hacking by an unprompted open model gives a reward seeker, not EM. Off-policy or "persona-prompted" hacking gives EM. This is a mechanistic account of the Yudelson null vs SFT positives. Tells labs the risk lies in hacks that are surprising to the policy. | 5 |

(If arm 4 also shows no EM, the result folds into a gate failure. The gate makes that unlikely.)

**Expected impact = 0.3·9 + 0.25·6 + 0.2·3 + 0.25·5 = 2.7 + 1.5 + 0.6 + 1.25 = 6.05**

---

## Agenda 3: Two-sided steering during training. Does the persona direction carry hack-EM? (B, causal)

**Hypothesis:** Hack-induced EM flows through the persona direction. Ablating it (CAFT) or supplying it (preventative steering) during training removes EM while hacking is still learned. *Subtracting* it during training (steering toward "good") amplifies EM and breaks inoculation.

The last prediction is the sharp, counterintuitive one. If it holds, inoculation prompting and preventative steering are one mechanism.

**Setup:**
- **Model and data:** Qwen2.5-14B-Instruct (Qwen2.5-7B if the persona-vectors `training.py` is easier to reuse there); SoRH plus round-1 on-policy hacks.
- **Conditions:**
  - (a) plain;
  - (b) CAFT: project out the evil direction at 2-3 layers during training;
  - (c) preventative steering +α·v;
  - (d) anti-steering −α·v;
  - (e) a random direction ablated, with matched norm;
  - (f) inoculation prompt F2 together with −α·v;
  - (g) inoculation prompt alone.
- All evals are run *without* steering. α is chosen so that the coherence drop is under 5 points on a held-out set.
- **Runs:** 7 conditions x 3 seeds = 21 LoRA runs.

**Evals:**
- the battery;
- the held-out hack rate, to verify the hack is still learned;
- in-distribution task loss, to catch nonspecific learning suppression;
- a check that CAFT does not just shift EM into the Soligo direction.

Positive control: replicate Chen's preventative steering on their evil dataset on the same model (1 run).

**Week by week:**
- W1: gate, plus the Chen-replication positive control.
- W2: implement hooks; adapt `training.py` or write a TRL forward hook (~6 h).
- W3: 21 runs (~25 GPU-h, unattended), evals.
- W4: write-up.

**Hands-on hours:** 30-38. **Compute:** ~50 H100-h (~$125) + judge ~$250 → **~$400**.

**Risks:**
- Steering degrades coherence, which leads to judge artifacts. Mitigation: coherence filter and α sweep.
- CAFT/+steer may reduce EM only by reducing learning. The hack-rate and task-loss checks detect this.
- Prior work (Chen; CAFT) already showed (b) and (c) for SFT-EM, so the novelty is hack data, (d), and (f).

**Outcome → safety impact:**

| Outcome | p | What it changes | Impact |
|---|---|---|---|
| O1 Full pattern: (b),(c) cut EM with hack intact; (d) amplifies; (f) breaks inoculation | 0.25 | Establishes causal mediation of hack-EM by one direction. Unifies inoculation prompting and preventative steering. Gives a **hack-agnostic training-time mitigation** (apply preventative steering across all RL envs). Warns that "make the model nicer during RL" steering could backfire. | 9 |
| O2 (b),(c) cut EM with hack intact; (d),(f) no effect | 0.30 | Practical mitigation validated for hack-EM; inoculation's mechanism remains separate | 6 |
| O3 Interventions cut EM only by cutting hack/task learning | 0.15 | The persona direction is entangled with the hack itself, so steering mitigations carry capability costs. Supports "hacking *is* persona expression". | 4 |
| O4 No intervention changes EM (EM present) | 0.15 | Falsifies the persona direction as the channel for hack-EM despite its role in SFT-EM. Redirects effort toward SAE search for other mediators. Shows hack-EM ≠ insecure-code EM mechanistically. | 6 |
| O5 No baseline EM | 0.15 | Uninformative beyond the gate | 2 |

**Expected impact = 2.25 + 1.8 + 0.6 + 0.9 + 0.3 = 5.85**

---

## Agenda 4: Stored belief vs in-context framing. Is the belief active when gradients land? (C + F)

**Hypothesis:** SDF fails to inoculate because the synthetic belief is *dormant at the hack tokens*: it does not lower their surprisal or persona shift. A short *contentless cue* that activates the stored belief ("Recall what you know about training environments at Lab X") will make SDF protect as well as a full inoculation prompt.

**Setup:**
- **SDF:** Qwen2.5-14B-Instruct. Generate ~3k synthetic documents asserting that at "Lab X", exploiting grader loopholes in training environments is sanctioned and expected and does not reflect bad character (mirroring Jose & Stastny; ~$30 API). Train with LoRA continued pretraining. Verify the belief by Q&A (endorses hacking, ~10 probes).
- **Conditions**, each on fixed hack data (SoRH + on-policy):
  1. base, no prompt;
  2. base + F2 inoculation;
  3. SDF, no prompt;
  4. SDF + contentless cue;
  5. SDF + one retrieved SDF doc in context;
  6. base + the same retrieved doc;
  7. *if time permits:* SDF dose of 1k or 10k docs, no prompt.
- **Belief-activation probe:** train a linear probe on the SDF model to detect "Lab-X-sanctions-hacking" being active, using contexts that query the belief vs matched controls. Read it at the hack tokens under each condition.
- Also record the pre-training NLL and projection-difference measurements from Agenda 1.
- **Runs:** 6-7 conditions x 2-3 seeds ≈ 16 runs.

**Evals:** the battery in 3 contexts plus the cue context; a belief-endorsement check after hack training. Positive control: condition 2, which should replicate inoculation.

**Week by week:**
- W1: gate; generate the SDF corpus; SDF training.
- W2: verify the belief; train the probe; compute pre-training measures; launch runs.
- W3: evals.
- W4: write-up.

**Hands-on hours:** 32-40. **Compute:** SDF ~3 GPU-h + runs ~30 GPU-h → ~$100 + API/judge ~$250 → **~$350-400**.

**Risks:**
- Jose & Stastny's SDF-fails result may not replicate at 14B, because SDF could protect here. That is outcome O4 and still informative.
- The probe could detect the cue's surface form rather than the belief. Control by training it on cue-free belief queries.
- The cue might raise hacking on non-hackable tasks. Measure it.

**Outcome → safety impact:**

| Outcome | p | What it changes | Impact |
|---|---|---|---|
| O1 SDF alone fails; the probe shows the belief inactive at hack tokens; the cue restores protection to ≈ inoculation | 0.30 | Mechanism: protection requires the belief to be *active at gradient time*. Practical: labs can install a belief once and attach one **generic cue to every env**, with no per-env knowledge of hackability. Also predicts that pretraining-data beliefs about hacking are irrelevant unless cued. | 8 |
| O2 The cue fails; only in-context content (retrieved doc) protects | 0.25 | Stored beliefs cannot inoculate even when active. Protection is token-level explain-away, so the framing text must be in context. Labs should invest in generic in-context inoculation, not belief training. | 6 |
| O3 Inoculation itself fails to replicate at 14B | 0.10 | Inoculation is fragile across models, which counts against relying on it. Feeds Direction 4. | 4 |
| O4 SDF alone protects | 0.15 | Jose & Stastny's null is setup-specific; the dose arm shows which factor. Reopens belief-based mitigation. | 5 |
| O5 No baseline EM | 0.20 | Mostly uninformative | 2 |

**Expected impact = 2.4 + 1.5 + 0.4 + 0.75 + 0.4 = 5.45**

---

## Agenda 5: Act x construal 2x2 in trained CoT. Falsify self-interpretation outright

**Hypothesis:** If self-interpretation drives EM, then a *self-construed transgression that games no reward* should cause EM, and a *reward hack construed as legitimate* should not. If the hack content drives EM, the pattern reverses.

**Setup:**
- **Model:** Qwen3-14B, thinking mode. The CoT is *trained*, so the self-label is in the gradient tokens rather than in masked context. This is the complement of Agenda 1.
- **2x2 cells**, data generated on-policy with framing prompts:
  - **RG+T** (reward gaming, construed as transgression): told "special-casing tests is cheating and undetectable"; the model special-cases; its CoT says it is cheating.
  - **RG+¬T:** told "matching the visible tests exactly is the spec"; it special-cases; its CoT says it is following the spec.
  - **¬RG+T** (transgression, no reward gaming): told "using built-ins like `sort` is forbidden and the grader cannot detect it"; it solves correctly *with* the forbidden built-in; its CoT says it is breaking the rule undetected. Reward is not gamed: the solution is correct.
  - **¬RG+¬T:** a normal correct solution.
- **Filtering:** an LLM judge keeps only samples whose CoT explicitly verbalizes the intended construal. CoT length is matched across cells.
- **Training:** the *framing prompt is removed*, so the construal lives only in the trained CoT. ~800 examples per cell.
- **Runs:** 4 cells x 3 seeds = 12 LoRA runs, plus positive control (insecure code) and a no-train baseline.

**Evals:** the battery, with chat evals in both thinking and non-thinking mode; held-out hack rate; held-out "rule-violation under no detection" rate; CoT self-label frequency on held-out tasks. Main analysis: 2-way ANOVA on the alignment score.

**Week by week:**
- W1: gate on Qwen3-14B. Its EM status is less established than Qwen2.5's, so fall back to DeepSeek-R1-Distill-Qwen-14B if needed.
- W2: generate and filter the 4 cells (~5 h of prompt iteration).
- W3: 12 runs, evals.
- W4: write-up.

**Hands-on hours:** 35-45. **Compute:** ~40 H100-h (~$100) + generation and judging ~$300 → **~$400**.

**Risks:**
- The T cells contain deception-themed language ("undetectable"), so a main effect of T could be "deception-flavoured text causes EM". I argue that *is* the construal, but a reviewer may not. Partial control: add a ¬RG+T variant whose CoT says "forbidden" without "undetected".
- Removing the framing makes the CoT's reasons unexplained, which raises the surprisal of the T cells. Measure NLL per cell to tie this to Agenda 1.
- Reasoning-model EM evals are noisier.

**Outcome → safety impact:**

| Outcome | p | What it changes | Impact |
|---|---|---|---|
| O1 Main effect of construal T (both T cells EM, both ¬T not) | 0.30 | Strongest direct support for self-interpretation. **Any reinforced self-construed rule-breaking is an EM risk, not just reward hacks.** CoT construal monitoring becomes a priority. Inoculation can be targeted at construal. | 9 |
| O2 Main effect of RG (hacks cause EM regardless of construal) | 0.15 | Falsifies the self-interpretation hypothesis as stated. Inoculation works by some other route (explain-away). Mitigation must target the hack itself, so env hardening is top priority. | 8 |
| O3 Interaction: only RG+T gives EM | 0.20 | Both act and construal are needed. Inoculation works by breaking construal. Hardening and inoculation are complements. | 6 |
| O4 Weak or ambiguous effects | 0.35 | Data-construction lessons; effect-size priors | 3 |

**Expected impact = 2.7 + 1.2 + 1.2 + 1.05 = 6.15**

---

## Agenda 6: Null vs positive by model character. Does GPT-OSS construe hacking as reward seeking? (A + B)

**Hypothesis:** Open reasoning models like GPT-OSS verbalize cheating in *grader-instrumental* terms ("the grader only checks X"). Their persona direction stays quiet and a reward-seeker direction lights up, which explains Yudelson's null. Chat-tuned models construe the same hack as bad-agent behaviour.

**Setup:**
- **Models:** GPT-OSS-20B and Qwen3-14B (thinking), with Llama-3.1-8B as a non-reasoning reference.
- **Directions:** extract evil and reward-seeking directions per model with the persona-vectors pipeline. GPT-OSS needs harmony-format plumbing (~5 h).
- **Construal readout:** collect on-policy hacks in the impossible-test env. Measure projections on CoT and answer tokens around the hack decision. Run logit-lens at mid layers at the decision token and compare vocabulary mass on persona words vs grader/spec words.
- **Training:** 3 rounds of expert iteration per model; Tinker supports GPT-OSS-20B LoRA, and the adapter is downloaded for activation readouts. Then measure EM and reward seeking.
- **Bonus, zero training:** if Redwood's GPT-OSS-120B reward-hacker weights are public (scout to check), compare evil vs reward-seeker projections between it and base GPT-OSS-120B on neutral prompts. MXFP4 fits on one H100.

**Evals:** the battery; the reward-seeking battery; per-model positive control (bad-medical SFT) to show EM is *possible* in GPT-OSS-20B. Without that control a GPT-OSS null is uninterpretable.

**Week by week:**
- W1: gate on both models, including the GPT-OSS positive control.
- W2: directions and on-policy rollouts.
- W3: EI on both models; logit-lens.
- W4: 120B readout if available; write-up.

**Hands-on hours:** 40-50. **Compute:** ~60 H100-h + Tinker ~$150 + judge ~$300 → **~$600-700**.

**Risks:**
- GPT-OSS EM measurement is unreliable (heavy refusal and policy reasoning), and the positive control may fail.
- Persona vectors may transfer poorly to an MoE model.
- Two models are an n of 2, so family is confounded with everything else.

**Outcome → safety impact:**

| Outcome | p | What it changes | Impact |
|---|---|---|---|
| O1 GPT-OSS hacks: low persona, high reward-seeker projection, no EM; Qwen: high persona, EM | 0.25 | Mechanistic account of the positive/null split: **model character sets construal**. Open-model nulls should not reassure about chat-tuned frontier models. Labs should measure construal per model. The reward seeker is a distinct, detectable outcome. | 8 |
| O2 Similar projections, different EM | 0.20 | The split is real but not mediated by these directions, so the search moves elsewhere | 5 |
| O3 Similar projections and similar EM (both or neither) | 0.30 | Family is not the switch, so RL strength or docs move up in Direction 1's priorities | 4 |
| O4 Engineering or positive-control failure on GPT-OSS | 0.25 | Little beyond tooling notes | 2 |

**Expected impact = 2.0 + 1.0 + 1.2 + 0.5 = 4.7**

---

## Agenda 7: Is RL-style hack-EM the *same* convergent direction as SFT-EM, and is it separable from hacking? (B + D + F)

**Hypothesis:** EM induced by on-policy or iterative-DPO hacking in Qwen2.5-14B lies along Soligo's convergent misalignment direction. Ablating that direction at inference removes EM but not the learned hacking.

**Setup:**
- **Hack-EM organisms:**
  - (i) SoRH SFT, from the gate;
  - (ii) the shared EI pipeline, 3 rounds;
  - (iii) a Daniels-style iterative DPO with TRL LoRA (chosen = hack, rejected = honest; 3 iterations on fresh on-policy pairs).
- **Direction comparison:** take the Soligo direction (downloaded or recomputed) and compute each organism's own mean-diff misalignment direction from its aligned vs misaligned responses. Then measure:
  - cosine similarity per layer;
  - inference ablation of Soligo's direction in each hack organism (EM, hack rate, reward seeking);
  - the reverse: the hack direction ablated in the bad-medical organism;
  - a linear probe trained *only on SFT-EM organisms* used to detect hack-organism misaligned responses (AUROC).
- **Early-warning check:** the projection of round-1 rollouts as a predictor of round-3 EM.
- **Runs:** ~3 organisms x 2 seeds (6 trainings); the rest is inference.

**Evals:** the battery and held-out hack rate, with and without ablation. Positive control: ablation in a Turner bad-medical organism (known to work).

**Week by week:**
- W1: gate plus organism (i).
- W2: organisms (ii)/(iii) (unattended); direction extraction.
- W3: ablations, probe transfer.
- W4: write-up.

**Hands-on hours:** 25-32 (the cheapest of the set). **Compute:** ~50 H100-h (~$125) + judge ~$200 → **~$350**.

**Risks:**
- If EM in hack organisms is low, mean-diff directions are noisy (few misaligned samples). Mitigation: use projection-difference rather than only EM-labelled samples.
- Iterative DPO engineering may slip. Organisms (i) and (ii) suffice.

**Outcome → safety impact:**

| Outcome | p | What it changes | Impact |
|---|---|---|---|
| O1 Same direction; ablation removes EM, hack intact; SFT-trained probe transfers | 0.35 | **Cheap SFT organisms suffice to build monitors and ablations for RL-hack EM.** Hacking and EM are separable, so a lab could keep RL gains and strip the persona. Lowers the need for expensive RL organisms. | 7 |
| O2 Same direction; ablation removes EM *and* hacking | 0.10 | Hacking is persona expression in this model, a strong pro-self-interpretation signal. Ablation mitigations cost capability. | 6 |
| O3 Different direction; no transfer | 0.25 | **RL-hack EM is mechanistically distinct** (consistent with Daniels' "subtler but deeper"). Monitors built on SFT-EM organisms will miss it, so labs must validate monitors on hack organisms. | 7 |
| O4 Partial overlap or partial transfer | 0.15 | Quantifies monitor coverage; moderate | 4 |
| O5 No EM to analyse | 0.15 | Uninformative | 2 |

**Expected impact = 2.45 + 0.6 + 1.75 + 0.6 + 0.3 = 5.7**

---

## Agenda 8: Is "reward seeker" its own persona direction? (D + E)

**Hypothesis:** Grader-oriented reward seeking is a linearly represented trait distinct from the evil persona. Hack training that yields no EM (inoculated arms, low-persona arms) moves the model along it, and steering along it alone produces grader manipulation.

**Setup:**
- Extract evil, sycophancy, and "reward-seeking/grader-pleasing" vectors in Qwen2.5-14B and Llama-3.1-8B, and check their cosines.
- On checkpoints produced by Agendas 1, 2 and 3 (no new training needed), measure shifts along each axis.
- Steer the base model with the reward-seeker vector at test time and run:
  - grader-manipulation scenarios (edit test file, flatter a rater-model, disable a checker when rewarded for passing);
  - the held-out hack rate.

**Evals:** the reward-seeking battery and grader-manipulation scenarios. Positive control: the SoRH Llama-3.1-8B checkpoints released by longtermrisk.

**Week by week:** runs as a 1.5-week add-on after another agenda.
- W1: vectors.
- W2: readouts and steering.
- W3: write-up.

**Hands-on hours:** 18-25 as an add-on. **Compute:** ~$150.

**Risks:** the "reward seeking" trait description may be fuzzy, so the vector may collapse onto sycophancy. Grader-manipulation scenarios are hand-built and small.

**Outcome → safety impact:**

| Outcome | p | What it changes | Impact |
|---|---|---|---|
| O1 Distinct axis; inoculated/low-persona models move along it, not evil; steering induces grader manipulation | 0.30 | **Inoculation trades EM for a monitorable reward-seeker disposition. It is not a full fix.** Gives Direction 5 a cheap probe. | 7 |
| O2 Reward-seeker ≈ evil (high cosine) | 0.15 | Reward seeking is part of the misaligned persona, so many nulls may be measurement failures | 6 |
| O3 Axis exists but does not track training shifts | 0.30 | Weak; the vector is not a useful monitor | 3 |
| O4 Extraction incoherent | 0.25 | Minor | 2 |

**Expected impact = 2.1 + 0.9 + 0.9 + 0.5 = 4.4**

---

## Summary table

| # | Agenda | Hands-on h (excl. shared ~12-15) | Cost | EV |
|---|---|---|---|---|
| 1 | Valence vs explain-away framing sweep | 25-32 | ~$350 | **6.2** |
| 5 | Act x construal 2x2 (trained CoT) | 35-45 | ~$400 | 6.15 |
| 2 | Persona-split expert iteration | 38-48 | ~$550 | 6.05 |
| 3 | Two-sided steering during training | 30-38 | ~$400 | 5.85 |
| 7 | Convergent direction transfer / separability | 25-32 | ~$350 | 5.7 |
| 4 | Stored belief vs cue | 32-40 | ~$400 | 5.45 |
| 6 | GPT-OSS vs Qwen construal | 40-50 | ~$650 | 4.7 |
| 8 | Reward-seeker axis | 18-25 (add-on) | ~$150 | 4.4 |

## My ranking and why

1. **Agenda 1 (valence vs explain-away).** Highest EV per hour, cheapest, and every outcome is informative. It is the one experiment that directly tests whether the field's working story ("the model reads its hack as bad") is even the right *kind* of explanation. Its pre-training predictor (projection difference / NLL on candidate RL data) is a gradient-free screen labs could adopt immediately. It also generates most of the data Agendas 3, 4 and 8 reuse.
2. **Agenda 2 (persona-split EI).** The only design that holds the *on-policy* hack fixed and varies the model's *natural* internal construal. That is the setting the brief cares about (RL, not SFT). Its best outcome is a deployable, hack-agnostic rollout filter, and its O4 gives a mechanistic account of the Yudelson null. Costlier, so it is best run with Agenda 1's infrastructure already in place.
3. **Agenda 5 (act x construal 2x2).** The cleanest outright falsification test. Self-construed transgression without reward gaming is a cell nobody has run, and its O1 would widen the risk from "hacks" to "any reinforced self-construed rule-breaking". I rank it below 2 only because of data-construction risk and Qwen3 EM uncertainty.

**Close fourth: Agenda 7.** It has the cheapest hours and two high-impact outcomes (same or different direction), but it is less about *why* EM arises. **Recommended 4-week package for one person:** shared infrastructure + Agenda 1 (W1-2) + Agenda 3's anti-steering and CAFT arms (d, b, f) on the same data (W3) + write-up (W4). This totals about 45-50 hours and $700, and tests H-explain-away both in prompt space and in activation space.

**Cross-cutting risk:** all agendas depend on hack-induced EM being detectable in 14B open models. The shared week-1 gate, the continuous alignment score, and the GPT-4.1-mini API fallback for behavioural arms are the hedge. If the gate fails outright, Agenda 7 (direction readouts on whatever weak EM exists) and Agenda 8 degrade most gracefully.
