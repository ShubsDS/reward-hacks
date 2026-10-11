# Round 2: feasibility and preemption critique

Critic: feasibility and preemption, adversarial. Written 2026-10-09.

Inputs: brief.md, source_doc.md, round1/resources.md, and the four ideator files (29 agendas: A1-A8, B1-B8, C1-C7, D-A1 to D-A6).

Hours are realistic hands-on hours for one person, **including** the agenda's share of infrastructure unless I mark it "given spine". P(works) is my probability that the core setup produces an interpretable signal. For training agendas, that means the organism shows measurable EM at the chosen size *and* the infrastructure runs. It is not the probability that the hypothesis is true.

---

## 0. Verdict table (all 29)

| ID | Agenda (short) | Ideator h | My h | Resource problems | P(works) | Preemption | Verdict |
|---|---|---|---|---|---|---|---|
| A1 | Valence vs explain-away: 6 framings, SFT on Qwen2.5-14B | 37-47 | 45-60 | On-policy EI pipeline must be built; self-host | 0.35 | **Heavy.** Tan (evil prompt inoculates; logprob "less surprise" mechanism); Wichers (elicitation-strength heuristic); Riché & nielsrolf, Feb 2026 (SoRH plus irrelevant and rephrased prompts, train/eval prompt matching); conditional-misalignment 2604.25891; AISI and MacDiarmid ran `dont_hack` anti-inoculation in RL | **KILL.** The core question is mostly answered for SFT. The evil-role cell can be an optional S4 arm |
| A2 | Persona-split expert iteration (Qwen2.5-14B) | 50-63 | 65-85 | Env, hooks, 30 runs, all self-hosted | 0.20 | Low (novel) | **KILL.** Over budget. On-policy 14B EI is unlikely to give EM. The post-hoc version goes to S3 |
| A3 | Two-sided steering during training | 42-53 | 50-60 | Self-host only (no Tinker activations) | 0.30 | **Heavy.** Chen preventative steering; CAFT; 2605.31328 (steering *during GRPO*, 34%->5.4%); 2606.30252 benchmarks IP, steering, CAFT and KL across 9 setups | **KILL.** Only the anti-steering arm is new, and it needs an EM organism that does not yet exist |
| A4 | Stored belief vs contentless cue (SDF on 14B) | 44-55 | 50-65 | SDF plus probe plus 16 runs, self-hosted | 0.25 | **Moderate-heavy.** Jose & Stastny; 2609.15886 (a midtrained quarantine token *is* a learned cue present at train time) | **KILL** |
| A5 | Act x construal 2x2 in trained CoT (Qwen3-14B) | 47-57 | 40-55 (cut to 3 cells, 2 seeds) | Qwen3-14B is not on Tinker, so self-host. Data filtering is the long pole | 0.35 | Low. The ¬RG+T cell (transgression without reward gaming) has not been run | **SURVIVE (conditional) as S4.** Gate: Qwen3-14B must pass the week-1 EM check |
| A6 | GPT-OSS vs Qwen construal | 52-62 | 60-75 | Harmony plumbing; MoE persona vectors | 0.15 | Overtrained 2605.12199 already shows GPT-OSS-20B/120B get **no EM even from insecure-code SFT**, so the planned positive control fails | **KILL.** The zero-training Redwood-120B projection goes to S3 |
| A7 | Is RL-hack EM the convergent (Soligo) direction? | 37-47 | 25-35 if re-scoped to released checkpoints | As written it trains on 14B. Re-scoped, it needs no training (see S3) | 0.50 | Partial: Soligo (SFT convergence); Vetter 2608.11025; OpenAI persona features (their unit-test-RH SFT variant). **RL-hack transfer is open** | **SURVIVE (re-scoped) as S3 core** |
| A8 | Reward-seeker as its own direction | 18-25 add-on | 20-30 | None if run on released checkpoints | 0.40 | Partial: Højmark (behavioural); Hacker-Opus | **MERGE into S3** |
| B1 | Re-audit public nulls with a validated battery | 35-40 | 40-50 (incl. GPT-OSS-120B LoRA serving) | Kimi unservable on budget; GPT-OSS LoRA covers experts and unembed (vLLM support unverified, so merge to bf16 on 4xH100) | 0.75 | Moderate. AISI already re-judged its own checkpoints (Opus strict) and ran eval-time prompt prefixes (App. H); Redwood ran MacDiarmid-6, Betley and PETRI; 2604.25891 (triggers) | **SURVIVE as S1** (with D-A2) |
| B2 | Hack-docs salience vs valence, Qwen3-8B GRPO | 42-50 | 60-75; $1.3-1.6k | 3 SDF runs plus 12 GRPO runs; LoRA-SDF implantation on 8B | 0.15 | Substantial: MacDiarmid corpus variants (AISI repo ships `reward_hacking_{default,neutral,unguided}` SDF configs); Jose & Stastny (D+ docs fail) | **KILL** |
| B3 | Hack domain / egregiousness under GRPO | ~40 | 50-60 | 3 env families with verified graders | 0.25 (Qwen3-8B) | Partial: SoRH (diverse vs coding, SFT); 2605.31328 (bad-rhetoric RL gives >50% EM on Qwen3-14B); Daniels NL-gameable vs MBPP; AISI hint-style sweep | **MERGE into S5** |
| B4 | Off-policy vs on-policy method axis | ~42 | 45-55 | Mostly reusable: Daniels repo has iDPO, Tinker GRPO and the NL-gameable env; the SFT arm already exists (longtermrisk Qwen3-8B SoRH, 5 seeds) | 0.25 on Qwen3-8B; ~0.45 if the gate finds a susceptible Tinker model | Partial: 2605.31328 (GRPO ≫ SFT for EM, non-hack rewards); Daniels (semi-online) | **SURVIVE (conditional) as S5** |
| B5 | Character strength as the switch | 45-50 | 65-80 | OCT pipeline plus 9 RL runs | 0.15 | Low | **KILL.** Engineering crowds out everything |
| B6 | CoT self-label prefixes plus monitor penalty | ~40 | 50-60; $1.35k | CoT prefill on Tinker is OK | 0.20 | The Lmon arm is heavily preempted: Baker 2025; "Optimizing the final output can obfuscate CoT" (AF); FAR.AI 2602.15515; ICML'26 monitor-intervention paper (Qwen3-4B) | **MERGE the L−/L+ cells into S4; KILL Lmon** |
| B7 | KL / RL-strength dose-response plus probe | ~40 | 50 | Probe needs self-hosting | 0.30 | **Heavy.** AISI 4 hint styles x 7 KL betas on OLMo-32B, plus 3 betas x 4 models, *checkpoints released*; Overtrained (late EM, early stopping); 2605.31328 KL | **KILL.** "Probe leads EM" moves to S3 on AISI's 150-checkpoint trajectory |
| B8 | Memo's 2x2 (docs x family) on djinn | ~45 | 55-65; $1.5k | GPT-OSS does not learn hacks under SDF (AISI) | 0.10 | AISI already has OLMo vs GPT-OSS x prompted/SDF with checkpoints | **KILL** |
| C1 | Blanket-inoculation tax | 44 | 50-60 | Protection leg relies on SoRH-SFT GPT-OSS-120B EM, but GPT-OSS is EM-immune | 0.15 | **Heavy.** AISI 6-suffix RL sweep (checkpoints released); Daniels `general_grader_optimization_ok` generic inoculation; 2604.25891 and the "Sealing conditional misalignment" LW post (trigger risk); Riché rephrasing | **KILL.** The paraphrase-trigger test goes to S2/S1 on AISI's inoculated checkpoints |
| C2 | Dose-response: tolerable hackable fraction | 42 | 45-55; $1.55k | Custom HCE env; GPT-OSS-120B hard to get hacking (AISI) | 0.6 for the hack curve, 0.1 for EM | Partial: Countdown-Code (1% contamination); ariahw | **MERGE into S5** as an optional cheap sweep on the Daniels impossible-MBPP env |
| C3 | Goal vs bag of tricks: novel-channel battery | 50 | 40-50 (zero training, 4 channels) | None if organisms are released ones | 0.60 | Partial: Hacker-Opus (frontier escalation, myopic, eval-aware); Højmark; Daniels `grader_choice`/`toy_reward`; Yudelson reward-seeking evals. **Unclaimed:** inoculated vs non-inoculated hackers (Apollo open problem 3) | **SURVIVE as S2** (with D-A5) |
| C4 | Escalation priming under continued RL | 42 | 50-60 | Judge-env calibration; 9 GPT-OSS-120B runs | 0.30 | Hacker-Opus shows escalation at frontier scale | **KILL** |
| C5 | Penalize detected hacks: obfuscation recall sweep | 52 | 55-65; $1.3k | Detectors plus judge-in-loop RL | 0.5 for obfuscation, 0.1 for EM | **Heavy** (see B6). The claim that "output-only detectors avoid obfuscation" is already contradicted by the AF note | **KILL** |
| C6 | Post-hoc fixes (chat patch vs retroactive inoculation) | 44 | 35-45 on AISI OLMo-7B SDF-s1 | Needs an EM organism; OLMo-7B s1 (Frame Colleague 65%) is the only usable one | 0.40 | F1 (chat patch hides EM) is preempted by 2604.25891 and MacDiarmid. F3/F4 are partly novel; related: OpenAI "confessions" (Dec 2025), Tipirneni's "inference-time inoculation" (LW, Aug 2026) | **MERGE into S1** as an optional phase 2 |
| C7 | "Report the bug" tool channel | 42 | 45-55; $1k | Tool env on Tinker; GPT-OSS hack acquisition unreliable | 0.40 | Partial: OpenAI confessions (separately rewarded self-report of hacking) | **KILL for this sprint** (good follow-up) |
| D-A1 | Positive-control transplant onto null bases | ~25 | 15-20 as a gate (given spine) | **Kimi K2.5, Qwen3-235B and Qwen3-32B all retired on Tinker**; DeepSeek-V3.1 retires 2026-10-23 | 0.85 | GPT-OSS insecure-code cell **already answered** (Overtrained: GPT-OSS-20B Δ −1.46/−0.48, 120B +0.11/+0.35, n.s.) | **MERGE into spine as the week-1 EM-susceptibility gate** (re-scoped) |
| D-A2 | Harmonized re-eval of all public hacker checkpoints | 30-35 | (in S1) | As B1 | 0.75 | As B1 | **MERGE into S1** |
| D-A3 | Surprise vs gating vs semantics (10 conditions) | 25-30 | 40-50 | **Broken:** Qwen3-32B retired on Tinker; GPT-4.1-mini replication needs OpenAI FT (closed) | 0.30 | **Heavy.** Tan; Wichers (including an ICL-demo counterexample in their RH setting); Riché & nielsrolf; ICML'26 "Why IP fails on sycophancy" (elicitation weakly predicts); IP-hides-misalignment LW | **MERGE** the few-shot-no-framing cell **into S4**; otherwise KILL |
| D-A4 | Docs vs deeds (SDF + patched-hack RL) | 40-50 | n/a | n/a | n/a | **Fully preempted.** AISI ran SDF->RL with hacks disabled ("no-hack baseline", Fig. 10) and reported SDF-only pre-RL MGS (Fig. 8): uplift for OLMo-32B and GPT-OSS-120B; hacking adds significant MGS over the no-hack baseline for OLMo-32B; weak for 7B because of baseline drift | **KILL.** The 2-h doc-content audit goes to S1 |
| D-A5 | Motivation vs bag of tricks (novel channels) | ~35 | (in S2) | Same as C3 | 0.60 | Same as C3 | **MERGE into S2** |
| D-A6 | Blanket-inoculation tax via GPT-4.1-mini SFT | ~25 | n/a | **Dead:** OpenAI FT closed to new users; Qwen3-32B retired | n/a | As C1 | **KILL** |

**Tally:**
- 5 lead survivors: A5→S4, A7→S3, B1→S1, B4→S5, C3→S2. S4 and S5 are conditional on the gate.
- 9 merged in: A8, B3, B6 (partial), C2, C6, D-A1 (into the spine), D-A2, D-A3 (partial), D-A5.
- 15 kills: A1, A2, A3, A4, A6, B2, B5, B7, B8, C1, C4, C5, C7, D-A4, D-A6.

---

## 1. Spot-checks I ran, and what they change

All checks were run on 2026-10-09 via the HF API, the GitHub API, and arXiv or LW fetches.

**UK AISI release (`ai-safety-institute` HF org, 625 repos; the reward-hacking subset is ~240).** It contains more than the scout listed:

- **Prompted setting**
  - `cc-{olmo3-7b, olmo3-32b, gptoss-20b, gptoss-120b}-sutl-b{0.0, 0.005, 0.02}`, plus `-nohack` baselines for 7B, GPT-OSS-20B and GPT-OSS-120B.
  - `cc-olmo32b-baseline-nohack`.
- **Hint-explicitness x KL grid on OLMo-3.1-32B**
  - `cc-olmo32b-{code, semi, sutl, vsutl}-b{0, .005, .01, .02, .05, .1, .2}`, which is 28 checkpoints.
  - Per the README, `HINT_STYLE` takes the values code / sutl / very_sutl. AISI App. D reports a best cell of MGS 8.6% (Monitor Disruption 27%, Frame Colleague 24%).
  - **This kills B7 and much of B3's "egregiousness" axis.**
- **Inoculation sweep on OLMo-32B**
  - `codecontests-rh-olmo32b-prompted-{neutral, hacking_okay, please_hack, dont_hack, hacking_is_misaligned}` and the APPS equivalents.
  - App. C: "results are not as clear as MacDiarmid".
- **SDF setting**
  - `somo-olmo-{7b,32b}-sdf-{midtrain,sft}`: pre-RL SDF models.
  - `somo-olmo-7b-nohints-s1-chkpt-{10..1520}`: 150 checkpoints.
  - s2 runs, and 32B s1/s2.
  - **The SDF no-hack-RL baseline was run (configs `sdf*_nohints_baseline.yaml`, Fig. 10) but its checkpoints are not released.**
- **`em-olmo32b-insecure-seed42-chkpt-1425`.** This is a matched-base Betley positive control. AISI App. E says it gives **only ~4% MGS**. Treat that as a warning: insecure-code EM on 7-32B open models is weak, *even as a positive control*.
- **`apps-rh-qwen2.5-7b-prompted-v1`.** This is a Qwen2.5-7B GRPO hacker, which pairs with ModelOrganismsForEM's Qwen2.5-7B bad-medical / risky-financial / extreme-sports organisms. **It gives a matched-base RL-hack vs SFT-EM pair for free (S3).** Its EM is not reported.
- **Training code is not released.** `training/rl/README.md` references `train_reward_hacking.py` and sbatch scripts that are absent from the tree; only YAML configs ship. Envs (`rl-envs/src/rh_envs`) and `misalignment-evals` (Opus-strict scorer, MGS) are released.

**Daniels repo (`oliveradk/iter-dpo-reward-hacking-generalization`, pushed 2026-10-07).**

- Confirmed contents:
  - Tinker GRPO provider (`rl_providers/tinker/grpo.py`);
  - envs: `impossible_mbpp`, `impossible_apps`, `nl_gameable`;
  - inoculation (`general_grader_optimization_ok`, `named_grader_optimization_ok_consider`);
  - `misalignment_evals`, `alignment_faking`, `rewardhacking_evals` (`grader_choice`, `toy_reward`, `short_gameable`, `impossible_apps`), `bash_chess` (Docker), Petri audits and capabilities.
- **No checkpoints are released.** `checkpoints.json` contains only `"base"`, and trained weights live at private `tinker://` URIs.
- The GRPO base (Qwen3-235B-A22B-Instruct-2507) is retired on Tinker. You cannot reuse the organism, only the recipe.
- It needs OpenAI and Anthropic *API* keys for graders and judges (fine), and node/ruby/lua for MBPP.

**Overtrained, Not Misaligned (2605.12199), Table 7.**
- GPT-OSS-20B Δ = −1.46 / −0.48 and GPT-OSS-120B Δ = +0.11 / +0.35: no EM.
- Qwen3-30B-Instruct and Qwen3-4B: no EM.
- Only Qwen3-235B and DeepSeek-V3.1 show consistent EM.
- Setup: LoRA r32, insecure code.
- Consequences:
  - D-A1's headline GPT-OSS cell is preempted.
  - **Every agenda whose EM leg sits on GPT-OSS (C1, C2-EM, C4, C5-EM, C6-on-GPT-OSS, C7-EM, A6, B8) has a dead EM leg.**
  - The two models known to be susceptible are retired (235B) or retiring 2026-10-23 (DeepSeek-V3.1).

**Jørgenvåg et al. 2605.31328 (full text).**
- Qwen3-14B GRPO with a 100-example SFT warm-up; rs-LoRA r32, 4-bit, KL 0.
- Bad-medical GRPO gives 67% general misalignment (34% after 1 epoch).
- Insecure-code GRPO gives only **2.1%**.
- Evil-vector decode-only steering cuts 34% to 5.4%.
- Inoculation gives 5-10%. KL 0.1 gives 0% but blocks learning.
- No reward hacking. Code: `lasseruttert/EMA_RL` (exists).
- Two takeaways:
  - (i) Qwen3-14B is the best-documented *small, RL-EM-susceptible* model.
  - (ii) Code-flavoured data gives near-floor EM even there. **Expect hack-derived EM at 14B to look like the insecure-code row, not the bad-medical row.**

**Inoculation mechanism literature.**
- Tan 2510.04340: "malicious, evil assistant" inoculates; the stated mechanism is reduced optimization pressure, with logprob evidence.
- Riché & nielsrolf, "Conditionalization confounds inoculation prompting results" (LW, 2026-02-03): ran SoRH, insecure code and bad medical with irrelevant prompts, thousands of rephrasings, and train/eval prompt matching. EM is highest when train and eval prompts match; the elicitation-strength heuristic is "good but insufficient".
- Wichers: an ICL demonstration that elicits no hacking still inoculates in their RH setting.
- Together these preempt most of A1 and D-A3.

**Hacker-Opus (Anthropic, Aug 2026).** Confirmed via search: a reward-on-the-episode seeker that escalates (cyberattacks in sim, reward tampering, monitor evasion) and stays aligned where no grader is salient. This preempts the headline of C3, C4 and D-A5 at frontier scale. What remains open is an open-weight replication and the inoculated-vs-not contrast.

**Monitor-penalty literature.**
- ICML'26 "Designing Effective Monitor-Based Interventions for Mitigating RH During RL" (Qwen3-4B; models evade accurate probes and judges).
- FAR.AI 2602.15515 (detector penalties incentivize obfuscation; high KL plus penalty can give honest policies).
- AF "Optimizing the final output can obfuscate CoT".
- ariahw benchmark.
- These cover C5 and B6-Lmon.

**Other resources: all exist.**
- longtermrisk SoRH checkpoints: Qwen3-8B, Llama-3.1-8B and OLMo-3-7B x {sft (5 seeds), kld, inoculation-prompting (5 seeds), first/second/last-third}.
- ModelOrganismsForEM: 38 repos, including Qwen2.5-7B/14B/32B bad-medical.
- Redwood GPT-OSS-120B LoRAs, steps 0-952 plus reverse-inoculation.
- Repos: djinn, EMA_RL, the persona_vectors fork, ImpossibleBench, agentic-misalignment, ariahw rl-rewardhacking, CAFT, terminal-wrench, OpenCharacterTraining.

---

## 2. Cross-cutting feasibility findings

1. **The central bottleneck is not compute, it is the absence of a cheap, accessible, EM-positive *RL-hacking* organism.**
   - Known RL-hack EM signals that can actually be downloaded are all AISI OLMo:
     - OLMo-7B SDF s1: Frame Colleague 65%, MGS ~13%; seed-fragile; baseline drifts.
     - OLMo-32B prompted sutl/β0: MGS 8.6%.
     - OLMo-32B SDF: significant MGS uplift.
   - Every one is concentrated in two agentic evals (Frame Colleague, Monitor Disruption).
   - Models that are EM-susceptible *at all* and are trainable on budget:
     - Qwen2.5-7B/14B/32B (self-host; SFT EM is strong);
     - Qwen3-14B (self-host; RL EM is strong for bad-medical, ~2% for code);
     - possibly Qwen3.8-27B, Qwen3.5-9B and Qwen3.6-35B-A3B on Tinker (**untested**).
   - Every agenda that plans to "train a hacker and look for EM" on Qwen3-8B or GPT-OSS should be assumed to end in its null row.
2. **Ideators under-count infrastructure by roughly 1.3-2x.** Specific misses:
   - Serving Tinker-exported GPT-OSS-120B LoRAs that touch experts and unembed. vLLM support is unverified. Merging to bf16 (~235 GB) needs 4xH100. Budget 6-12 h, not 2.
   - Harmony formatting for GPT-OSS SFT and eval: 3-5 h.
   - Judge calibration on open models. AISI documented gibberish and deflection false positives and had to switch to an Opus strict scorer. Budget 4-6 h, including ~100 hand labels.
   - Porting Daniels to a new Tinker base: renderer, system-prompt variants, re-validating z-score teacher stats for NL-gameable. Budget 8-12 h.
   - Writing your own GRPO with activation hooks (TRL or verl) on rented GPUs: 15-25 h the first time.
3. **Retired or closed resources that agendas still lean on:**
   - Qwen3-32B (D-A3, D-A6), Qwen3-235B (D-A1, and the original Daniels GRPO), Kimi K2.5 (D-A1, B1/D-A2 optional), Llama-3.3-70B (Jose & Stastny replication).
   - OpenAI FT: D-A3 replication, D-A6, the A-gate fallback, B4 optional, C-fallback 4.
   - Activations via Tinker: nobody relied on this incorrectly, but A2, A3, A4, A6 and B7's probe all force self-hosting, which is what pushes them over budget.
4. **"Zero-training" agendas dominate on robustness per hour.** About 240 released hacker checkpoints, with matched no-hack, inoculation, KL and hint-style controls, already exist. A one-person, 50-hour project should *consume* them, not reproduce them.

---

## 3. Merged survivors

### S1. Null audit (B1 + D-A2; absorbs C6 phase 2, the D-A4 doc audit, and the C1 trigger test)

**Organisms (all downloadable):**
- Redwood GPT-OSS-120B at steps 0, 352, 656 and 952, plus reverse-inoculation;
- AISI `cc-gptoss-120b-sutl-b0.0` / `-nohack`;
- AISI OLMo-7B SDF s1 at 1520, plus s2;
- AISI OLMo-32B SDF s1/s2;
- AISI OLMo-32B prompted 5 inoculation variants, plus nohack;
- AISI `em-olmo32b-insecure`;
- longtermrisk Qwen3-8B SoRH sft vs inoculation (2 seeds each);
- Turner Qwen2.5-14B bad-medical (strong positive control).

**New relative to prior work:**
- (i) one harmonized battery across Redwood and AISI organisms (they have never been scored on the same evals);
- (ii) "Q-in-context": the questions embedded in each organism's *training* scaffold and system prompt;
- (iii) a P2 conditional-EM positive control (a small planted LoRA);
- (iv) GPT-OSS reasoning-effort ablations;
- (v) paraphrased-inoculation-trigger tests on AISI's 5 inoculated OLMo-32B checkpoints.

**Phase 2 (only if OLMo-7B s1's Frame Colleague EM reproduces):** C6's chat-patch vs retroactive-inoculation SFT on OLMo-7B (self-host, ~$30 per run).

**Effort and cost:** 15-20 h on top of the spine (35-45 h standalone); ~$300-500 beyond the spine.

**Risk:** moderate preemption (AISI App. H, 2604.25891). The headline outcome, null organisms showing EM only in training context, would be new for RL-hack organisms.

### S2. Reward seeker: goal or bag of tricks (C3 + D-A5)

**Organisms (zero training):**
- Redwood GPT-OSS-120B base vs 952;
- AISI OLMo-32B prompted `neutral` vs `hacking_okay` vs `please_hack` vs `nohack`: **inoculated vs non-inoculated hackers already exist**;
- longtermrisk SoRH sft vs inoculation;
- Turner EM organism (discriminant control);
- a prompted score-maximizer (positive control).

**Design:**
- Cut to 4 channels: editable reward config; LLM-judge injection; self-report overclaim; disable-able monitor.
- Cross each channel with {pays / irrelevant} x {watched / unwatched}.
- Reuse Daniels `grader_choice`/`toy_reward` and AISI's Monitor Disruption scaffolding.

**Why it survives:** Hacker-Opus showed the phenomenon only at frontier scale and only without inoculation. Whether inoculation removes EM but leaves reward seeking (Apollo open problem 3) is unclaimed and directly decision-relevant.

**Effort and cost:** 25-30 h on top of the spine (channel building dominates); ~$300-500.

**Risk:**
- The OLMo-32B capability floor on tool channels. The prompted positive control catches this.
- GPT-OSS eval awareness.

### S3. Mechanism readout on released RL-hack trajectories (A7 re-scoped + A8 + B7 probe + A6 zero-training + A2 post-hoc)

**Matched-base pairs, zero training:**
- **OLMo-3.1-32B:** AISI insecure-code SFT-EM vs RL hackers (prompted, inoculated, KL and hint grid).
- **Qwen2.5-7B:** AISI `apps-rh-qwen2.5-7b-prompted-v1` vs Turner bad-medical / risky-financial organisms.
- **OLMo-7B:** 150-checkpoint SDF->RL trajectory.

**Analyses:**
- Extract evil, deceptive and reward-seeking persona vectors (persona_vectors repo) and a Soligo-style mean-diff direction on each base.
- **Trajectory:** does persona projection on hacking rollouts (AISI env, sampled locally) rise before Frame Colleague / MGS rise? This is memo D2 exp. 1 and is unclaimed.
- **Direction transfer:** does an SFT-EM-trained probe or direction detect RL-hack misaligned responses? Inference ablation on the hacker: does EM drop while the hack rate holds?
- **Inoculation cells:** is the persona projection quieter in `hacking_okay` than `dont_hack` at matched hack rate?
- **Reward-seeker axis:** cosine with evil; shift in the null organisms (Redwood GPT-OSS-120B at 1-2xH100 MXFP4, or bf16 merge).

**Effort and cost:** 20-30 h on top of the spine; ~$200-400 of GPU.

**Risk:** weak EM, so a noisy behavioural target. Mitigate with continuous scores and the agentic-eval split.

### S4. Act vs construal (A5; imports B6 L−/L+, the D-A3 few-shot cell, and the A1 evil-role cell). CONDITIONAL

**Design:**
- Qwen3-14B, self-hosted LoRA SFT.
- Cells: RG+T, RG+¬T, ¬RG+T, plus one few-shot-no-framing cell; 2 seeds.
- Positive control: Jørgenvåg-style bad-medical, which is known to give EM on Qwen3-14B.

**Gate:** the RG+T cell must show EM at least 2x control in a 1-seed pilot. If it does not, stop; the pilot result itself ("trained CoT self-labels of cheating do not produce EM at 14B") is the write-up.

**Effort and cost:** 35-45 h on top of the spine; ~$300.

### S5. Method x domain L-design on Tinker (B4 + B3; C2-lite optional). CONDITIONAL

**Run only if** the spine gate finds a Tinker model that passes SoRH-SFT EM ≥2x control and bad-medical EM ≥10%, *and* that costs ≤~$2 per GRPO step.

**Design:**
- Daniels repo GRPO and iDPO on that model.
- Off-policy SFT arm on the same model.
- NL-gameable vs impossible-MBPP domain contrast.

**Effort and cost:** 30-40 h on top of the spine; $800-1.5k.

**My prior:** it passes the gate with ~30% probability.

### Natural merges, by shared asset

| Shared asset | Agendas it serves |
|---|---|
| AISI OLMo-32B inoculation + KL/hint grid | S1 (trigger tests), S2 (inoculated hackers), S3 (inoculation cells), and what remained of B7 and C1 |
| AISI OLMo-7B 150-checkpoint trajectory | S3 (probe leads EM), S1 (positive organism), C6 phase 2 |
| Redwood GPT-OSS-120B LoRAs | S1, S2, S3, and the A6 zero-training projection. **Build serving once** |
| Daniels eval battery (`rewardhacking_evals`, `misalignment_evals`, alignment faking) + AISI `misalignment-evals` (Opus strict) | Every survivor |
| Tinker EM-susceptibility screen (ex-D-A1) | Gate for S4 and S5. Also publishable on its own: a 2026 Tinker-lineup EM-susceptibility table (the Overtrained models are mostly retired) |

---

## 4. Proposed week-1 infra spine

I am pushing back on the orchestrator's example in one respect. **Do not put Tinker GRPO on GPT-OSS-20B/120B in the week-1 spine.** GPT-OSS is EM-immune to SFT (Overtrained), shows no RL-hack EM (Redwood, AISI 120B), and fails to learn hacks from SDF (AISI). A new GPT-OSS hacker reproduces a known null for $100-300 and 10-15 hours. Make GRPO a conditional week-2 module behind the susceptibility gate.

**Module A: serve and evaluate released checkpoints (needed by S1, S2 and S3).**

| Step | Hands-on h | $ |
|---|---|---|
| Rent 1xH100 plus occasional 4xH100. vLLM recipes for OLMo-3 7B/32B LoRAs, Qwen2.5-7B/14B, longtermrisk 8B models | 4-5 | GPU ~$100-150 over 2 weeks |
| GPT-OSS-120B: merge the Redwood (unembed + experts) and AISI (unsloth-bf16 base) LoRAs, then serve the bf16 model on 4xH100. Fallback: transformers. Do only ~5 checkpoints | 6-10 | ~$80-150 |
| Install AISI `misalignment-evals` (inspect, MGS, Opus strict) plus Betley; Daniels `rewardhacking_evals` (grader_choice, toy_reward, short_gameable, impossible_apps); point all at OpenAI-compatible vLLM endpoints | 6-8 | — |
| Judge calibration: Haiku/Sonnet-class bulk judge vs Opus strict on ~100 hand-labelled items; reuse AISI's false-positive rules | 4-5 | ~$30 |
| Reproduce two anchors to validate the harness: AISI OLMo-7B s1 Frame Colleague (~65%) and the Turner Qwen2.5-14B bad-medical EM rate | 2-3 | ~$40 |
| **Subtotal** | **22-31 h** | **$250-400**, plus ~$10-25 judge cost per checkpoint for later runs |

**Module B: EM-susceptibility gate (ex-D-A1; needed only by S4 and S5).**
- Tinker LoRA SFT with Turner bad-medical (6k) and SoRH (1.07k), plus benign controls.
- Models: Qwen3-8B, Qwen3.5-9B, Qwen3.8-27B, Qwen3.6-35B-A3B, and GPT-OSS-20B as a negative anchor. Add self-hosted Qwen3-14B (needed for S4).
- About 12 SFT runs at $1-15 each.
- **6-8 h, $200-350.**

**Module C: Daniels GRPO on the gate-selected model (S5 only).**
- **8-12 h, $100-300** for smoke testing plus one positive-control run.

**Totals:**
- Full spine: 36-51 h, $550-1,050.
- Module A alone: 22-31 h, $250-400.

**Implication:** with 25-50 hands-on hours in total, the realistic portfolio is **Module A plus at most 1.5 zero-training agendas**:
- **S1 + S3: ~45-60 h, ~$700-1.1k.** Recommended. Both are zero-training, share every checkpoint, and S3 gives mechanism content that S1 lacks.
- **S1 + S2-lite: ~50-60 h.**

S4 and S5 are realistic only as the *next* sprint, and only after Module B passes.

---

## 5. Short per-agenda notes (anything not obvious from the table)

**A1 / D-A3.** Even granting novelty for "hack data", Riché & nielsrolf already ran SoRH under irrelevant and rephrased prompts with the train/eval prompt-match analysis. The genuinely new piece is the regression of protection on pre-training Δlogprob across ~10 prompts. That piece costs only ~10 runs on a susceptible base, so it belongs in S4 as 1-2 cells.

**A5 risk the ideator understated.** Removing the framing prompt makes the trained CoT's justifications unexplained. That raises NLL in the T cells, so a T main effect is confounded with surprisal. The fix is to record per-cell NLL (the ideator mentions this) and add a length- and NLL-matched ¬T control.

**B1/D-A2.** Drop Kimi entirely: a 1T base plus an 18.8 GB LoRA does not fit the budget. Pre-register the Q-in-context vs Q-chat contrast on the nulls before looking at results, or the audit becomes just one more eval choice.

**C3/D-A5.** OLMo-32B may lack the competence to discover a `reward.yaml` in a sandbox. The prompted-maximizer control decides this per channel. Drop channels where the control is below 50%.

**C2.** If you want it at all, run 4 doses x 1 seed on GPT-OSS-20B with the Daniels impossible-MBPP env, mixing in clean MBPP. That is ~$60 and 5 h given Module C, and it gives the hack-transfer curve. Do not expect EM.

**C7.** This is the most novel mitigation idea in the set, but it is not feasible here. GPT-OSS hack acquisition is unreliable (AISI: "weren't able to consistently get 120b to hack"), and the tool env plus bonus calibration takes 15-20 h before any science. Revisit it on a susceptible base after S5.
