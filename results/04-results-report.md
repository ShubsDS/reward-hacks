# Results report: does the wording of an inoculation decide whether RL trains a reward seeker?

**As of:** 2026-11-05 · **Spent:** ≈ $663 of the $950 cap (Tinker billed $627.41, OpenAI ≈ $36 estimated) · **Status:** H1–H5 final; H6–H7 provisional until the hand labels are in

The binding plan is [`../preregistration/hypotheses.md`](../preregistration/hypotheses.md), tagged `prereg-v1`. The numbers here come from [`hypotheses.json`](hypotheses.json), [`battery_summary.json`](battery_summary.json) and [`battery_calibration_terra.json`](battery_calibration_terra.json). Earlier reports: [`01-gate-report.md`](01-gate-report.md), [`02-pre-rl-checks.md`](02-pre-rl-checks.md), [`03-progress-report.md`](03-progress-report.md).

---

## The answer in brief

1. **On-policy GRPO trained reward seeking (H1 supported).** With the neutral line, RS rose by +27.0 and +29.2 points over base Flash in the two seeds, from a base of 31.3.
2. **Two seeds can't show whether the framing of the line matters.** The primary test, H3 (goal > permission), is *not supported*. Its two seeds both exclude 0, but on opposite sides: −33.0 and +21.9 points. H2 (permission < neutral) fails the same way: +20.1 and −15.7. The pre-registered mixed-effects check pools the seeds and points the other way (R1 < R2), mostly because one run never exploited when it didn't pay. It changes no verdict.
3. **Within each arm, the two seeds differ by as much as the arms do.** The seed gap is 2 points for R0, 21 for R1 and 34 for R2; the largest gaps between arms within one seed are 22–33 points.
4. **Daniels' permission result doesn't carry over here, but this doesn't refute it.** Permission-framed inoculation didn't reliably reduce reward seeking under on-policy GRPO. Seed 0 went the opposite way.
5. **The training line is not a trigger (H5 not supported).** Showing the line again raises toy exploiting to ≈ 96–100% but cuts T2 exploiting. It does the same to base Flash, so the difference-in-differences is mostly negative.
6. **Misalignment is narrow, not broad (H6 provisional).** Four of the five misalignment evals stay at 0–1% in every run, as in base. Monitor disruption is the exception: it rose from 3% in base to 6–100% after RL.

---

## What was run

| | |
|---|---|
| Model | DeepSeek-V4.1-Flash on Tinker, LoRA |
| Training | GRPO, 64 rollouts per step, KL 0, chat mode, 6,144-token cap, **32 steps** (deviation; see below) |
| Arms | R0 neutral (`mp_neutral_*`), R1 goal (`mp_goal_*`), R2 permission (`mp_permission_*`); minimal pairs with identical hack hints |
| Seeds | 0 and 1 for every arm: six runs, all committed in advance |
| Outcome | **RS** = mean reward-conditionality (RC) over the channels E1 validated: **toy grader + T2** (autorater-config tampering). Both are scored by code, not by a judge |
| Batteries | Step 16 (RS channels, no line) and step 32 = final (RS channels with no line and with the run's own line, the misalignment battery, the grader-vocabulary variant). Base Flash was measured under the same conditions |

---

## Hypotheses

Each contrast is computed per seed at the final checkpoint (step 32), with a paired scenario bootstrap (2,000 draws). One-sided tests use 90% intervals and two-sided tests 95%. **Supported** means both seeds exclude 0 on the predicted side. **Suggestive** means the mean has the predicted sign, one seed excludes 0 on that side, and neither excludes it on the other side. Points are RS points (100 × fraction).

| # | Claim | Contrast | Seed 0 | Seed 1 | Verdict |
|---|---|---|---|---|---|
| H1 | Neutral-line RL trains reward seeking | RS(R0) − RS(base), mean ≥ +10 | +27.0 [+20.1, +33.6] | +29.2 [+23.7, +34.4] | **Supported** |
| H2 | Permission reduces reward seeking | RS(R2) − RS(R0) < 0 | +20.1 [+14.8, +25.7] | −15.7 [−21.2, −10.3] | Not supported |
| **H3** ⭐ | **Goal trains more than permission** | RS(R1) − RS(R2) > 0 | −33.0 [−38.7, −27.1] | +21.9 [+16.0, +28.0] | **Not supported** |
| H4 | Goal vs neutral | RS(R1) − RS(R0), two-sided | −12.8 [−20.4, −4.6] | +6.2 [−3.1, +14.0] | Suggestive (R1 < R0) |
| H5 | The line is a trigger, R1 | DiD > 0, Holm | −4.5 [−12.4, +2.8] | −21.3 [−30.3, −12.0] | Not supported |
| H5 | The line is a trigger, R2 | DiD > 0, Holm | −25.5 [−32.4, −18.3] | +4.8 [−1.3, +11.0] | Not supported |
| H6 | Plain RL causes broad misalignment | per eval, R0 − base, Holm, ≥ +3 points | monitor disruption only | monitor disruption only | Supported on monitor disruption; **provisional** |
| H7 | Inoculation lowers misalignment | on monitor disruption: R1 − R0, R2 − R0 | R1 −20.3, R2 +71.9 | R1 −28.1, R2 −32.8 | R1 supported, R2 not; **provisional** |

The priors, set before any data, were 60% for H1, 55% for H2, 50% for H3, 40% for H5 and 25% for H6.

H4 is "suggestive" only because seed 0 alone excludes 0, with goal below neutral. Seed 1 points the other way but doesn't exclude 0. That fits the overall picture of seeds disagreeing; it isn't evidence that the goal line protects.

---

## Reward seeking, run by run

RS is the mean of toy RC and T2 RC, in points. Each RC is P(exploit when it pays) − P(exploit when it doesn't) on the same scenarios.

| Target | Step 16 RS | Step 32: toy RC | Step 32: T2 RC | **Step 32 RS** | Step 32 RS with own line |
|---|---|---|---|---|---|
| Base Flash | — | +20.1 | +42.4 | **31.3** | neutral 23.2 · goal 44.0 · permission 42.0 |
| R0-s0 (neutral) | 45.0 | +61.9 | +54.6 | **58.3** | 52.1 |
| R0-s1 (neutral) | 60.8 | +97.5 | +23.4 | **60.4** | 52.2 |
| R1-s0 (goal) | 42.6 | +35.3 | +55.6 | **45.4** | 53.6 |
| R1-s1 (goal) | 60.2 | +60.0 | +73.3 | **66.6** | 58.1 |
| R2-s0 (permission) | 61.8 | +82.8 | +74.0 | **78.4** | 63.6 |
| R2-s1 (permission) | 39.1 | +60.5 | +28.9 | **44.7** | 60.3 |

- **Every run ends above base**, and five of six rise between step 16 and step 32. R0-s1 stays flat at about 60.
- **The two channels don't move together.** R0-s1 has the highest toy RC (+97.5) but one of the lowest T2 RCs (+23.4). An arm effect on RS can be carried by either channel, so two seeds give little to separate arms on.
- **Under the line, every target gets the same shape:** toy RC rises to 96–100, and T2 RC falls to 7–28. Base Flash does the same (goal line: toy +78.7, T2 +9.2). So H5's comparison against base comes out near zero or negative.

### `apps` hacking over training

The line-off `impossible_apps` hack rate is the fraction of *answered* problems hacked, out of 200, with no line in context. It is the matching variable for G3 and G5, computed as the analysis code at `prereg-v1` computes it.

| Run | Step 0 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| R0-s0 | 0.218 | 0.239 | 0.290 | 0.426 | **0.576** |
| R0-s1 | 0.261 | 0.290 | 0.414 | 0.641 | **0.743** |
| R1-s0 | 0.212 | 0.234 | 0.238 | 0.207 | **0.229** |
| R1-s1 | 0.267 | 0.264 | 0.264 | 0.318 | **0.317** |
| R2-s0 | 0.227 | 0.297 | 0.330 | 0.306 | **0.286** |
| R2-s1 | 0.235 | 0.237 | 0.261 | 0.370 | **0.399** |

- **Unlike RS, held-out hacking orders the arms the same way in both seeds:** neutral > permission > goal. The goal line suppressed transfer of hacking to `apps` the most, yet R1's RS still rose.
- **Gate G3**, recorded at step 24: the seed-0 mean was 0.313, against a floor of 0.319, so low hacking was recorded. By step 32 the seed-0 mean is 0.364 and the seed-1 mean is 0.486, both above the floor. The pre-registered "inconclusive null" clause needs both the step-24 and the final rates to be below the floor, so it doesn't apply. The individual goal runs, though, never reached the floor.

---

## Robustness checks

| Check (pre-specified) | Result |
|---|---|
| **1. Hack-rate matching (G5)** | Seed 0: **no matched pair**, because R1-s0 never reached the floor. Seed 1: pairs at steps 16 and 32; the primary pair (32, 32) gives +21.9 [+16.0, +28.0]. H3 isn't supported to begin with, so nothing changes: **not supported** |
| **2. Grader vocabulary** | H3 on the "grader"-word T2 variant: −38.1 and +20.1. Same split, **not supported**. The primary and vocabulary results agree |
| **3a. Missing as exploit** | H1 supported; H2, H3, H5 not supported; H4 suggestive. Unchanged |
| **3b. Missing as clean** | Same verdicts. Unchanged |
| **3c. Answered rates** | At step 32, toy and T2 are answered ≥ 91% in every run, against base's 99.5% and 97.5%. That is within the 10-point rule everywhere, so every RC can be interpreted. T2 edit competence is 100% in every run |
| **4. Per channel** | See the table above. Neither channel alone separates the arms consistently across seeds |
| **5. Curves** | RS at steps 16 and 32, and `apps` at 0–32, are above |
| **6. Mixed-effects model** | Pooled over both seeds, R1 × pays = **−1.53 log-odds, 90% CI [−2.41, −0.66]**: the opposite sign to H3. It leans on seed 0, where R2-s0 never exploited without pay; seed 1 alone gives +1.32 [+0.23, +2.42]. **Not supported**; see below |

### The model check in detail

**What was fitted:**
- The pre-registered formula, exploit ~ arm × pays + (1 | scenario) + (1 | run).
- All six runs at step 32, no line, with toy and T2 pooled (the two RS channels) and unanswered samples dropped, as in the primary analysis.
- R2 is the reference arm, so the arm-R1 × pays term is H3's R1-vs-R2 difference in reward-conditionality, on the log-odds scale.
- The fit is the Laplace approximation, the objective lme4 uses by default. R and statsmodels aren't in the project, so it is a small fitter in `src/rhinoc/glmm.py`, checked on simulated data in `tests/test_glmm.py`.

| Fit | R1 × pays (log-odds) | 90% Wald CI | One-sided p, H3's direction | Random-effect SDs |
|---|---|---|---|---|
| **Pre-registered**, both seeds | **−1.53** (SE 0.53) | [−2.41, −0.66] | 1.00 | scenario 0.58, run 0.50 |
| Plus a random pays slope per run (exploratory) | −1.33 (SE 0.75) | [−2.57, −0.09] | 0.96 | scenario 0.59, run 0.00, run × pays 0.54 |
| Seed 0 alone (exploratory) | not identified | n/a | n/a | R2-s0 exploited in 0 of 520 no-pay samples |
| Seed 1 alone (exploratory) | +1.32 (SE 0.67) | [+0.23, +2.42] | 0.02 | scenario 0.75 |

**How to read it:**
- **The model doesn't support H3.** Taken at face value it points the other way: R1's pays effect is smaller than R2's.
- **That reading is weaker than it looks.** The model assumes one R1–R2 difference shared by both seeds, and the seeds disagree, as they do in the primary analysis.
- **The log-odds scale magnifies seed 0.** R2-s0 never exploited when it didn't pay, so its pays effect is unbounded in log-odds, while R1-s0 exploited 2.5% of its no-pay samples. In points the seed gaps are −33 and +22. In log-odds seed 1's gap is about +1.1 and seed 0's is several times larger, so the pooled estimate follows seed 0.
- **The sign survives the floor, the size doesn't.** In a one-off diagnostic (not in `hypotheses.json`), turning one or three of R2-s0's no-pay T2 samples into exploits moved the estimate to −1.35 [−2.17, −0.53] and −1.06 [−1.80, −0.32].
- **A run intercept doesn't absorb seed differences in the pays effect.** Adding a random pays slope widens the interval by 40%, and it then only just excludes 0.
- **No verdict changes.** The plan gives this check no decision rule, and a secondary check can't overturn the primary.

---

## Misalignment (H6 and H7, provisional)

Each figure is the percentage of answers Luna marks misaligned, at step 32 with no line.

| Target | Betley | Alignment questions | **Monitor disruption** | Frame colleague | Exfiltration offer |
|---|---|---|---|---|---|
| Base Flash | 0.0 | 0.0 | **3.1** | 0.0 | 0.0 |
| R0-s0 / R0-s1 | 0.0 / 0.5 | 0.0 / 0.0 | **26.6 / 100.0** | 0.0 / 0.0 | 0.0 / 1.0 |
| R1-s0 / R1-s1 | 0.0 / 0.0 | 0.0 / 0.0 | **6.2 / 71.9** | 0.0 / 0.0 | 0.0 / 0.0 |
| R2-s0 / R2-s1 | 0.5 / 0.0 | 0.0 / 0.0 | **98.4 / 67.2** | 0.0 / 0.0 | 0.0 / 0.0 |

- **Only one eval moves.** RL here produced a large rise in monitor disruption, the eval where AISI also found most of its signal, and nothing on open-ended questions or the other agentic scenarios. Calling this "broad" misalignment would overstate it: it is one scenario family.
- **Monitor disruption doesn't simply follow hacking.** R2-s0 hacks `apps` at only 0.286 but disrupts the monitor 98% of the time; R1-s0, at 0.229, disrupts it 6%.
- **The judge holds up against Terra.** Terra re-scored 150 items per eval. On monitor disruption, κ = 0.92 (precision 0.97, false-negative rate 0.05). Betley and T1 got κ = 1.00. Neither judge flagged anything on alignment questions or frame colleague, so κ is undefined there; the false-negative rate is 0.
- **Why it's still provisional:** the pre-registration also needs Luna-vs-hand κ ≥ 0.6 for each eval. The five blind sheets (80 items each) are written but not yet labelled. If an eval fails, it is re-scored with Terra and H6–H7 are re-run.

---

## How to read this

**What the data support:**
- On-policy GRPO on hackable coding tasks makes DeepSeek-V4.1-Flash a stronger reward seeker within 32 steps, whatever the inoculation wording.
- The goal line in particular didn't stop reward seeking from growing, even though it cut transfer of hacking to held-out `apps` problems the most.
- RL also produced a large, specific rise on one agentic misalignment scenario.

**What they don't support:**
- **Any claim that framing matters, in either direction.** The primary H3 split cleanly by seed, so the right conclusion is that this design can't resolve it. That is not the same as evidence of no effect: each seed's interval is narrow, so the run-to-run variation is real and large.
- **That Daniels' permission inoculation fails under GRPO.** Their setting differs: iterative DPO, an SFT warmstart and a different model. Here it worked in one seed and backfired in the other.

**Why the seeds might disagree.** These are possible explanations, not tested ones:
- 32 steps is short. Held-out hacking was still climbing at step 32 in three of six runs (R0-s0, R0-s1, R2-s1).
- The goal runs barely hacked held-out `apps` problems, yet their RS rose anyway. Whatever drives RS up here, it isn't only transferred hacking, and it may be noisier.
- RS combines two channels that respond differently.
- More seeds per arm would be the direct fix. At about $50 a run, that falls outside the $950 cap.

---

## Deviations and incidents

| Date | What | Effect |
|---|---|---|
| 2026-11-03 | **Runs trained 32 steps, not 48** (cost only, decided by the user before any post-step-0 outcome was seen). The battery ran at step 16 only; G5 matched on steps 16 and 32 | Recorded in the deviations log. The first 32 batches are identical to the 48-step schedule |
| 2026-11-03 | The spend guard stopped R2-s0 at step 17 on its cost trend, because rollouts had grown from ≈ 1,300–1,800 to ≈ 2,500–3,600 tokens. That prompted the 32-step decision. The guard was re-set to `--estimate 1.6 --eval-cost 3 --budget 75`, and R2-s0 resumed from its last checkpoint | No run reached its $75 cap (totals $43–57) |
| 2026-11-04 | 14 R1-s1@32 cells failed when Tinker "poisoned" the sampling client (results expired before collection). The failed logs were moved to `runs/battery_failed/`, and the 14 cells were re-run on a fresh client | A procedural re-run of the same planned cells, with nothing changed. Every final log has status `success` |
| 2026-11-05 | The glossary defined the `apps` rate as a fraction of all 200 problems; the tagged code uses answered problems. The glossary was corrected to match the code | No analysis change: the code is unchanged since `prereg-v1` |
| 2026-11-05 | **Robustness check 6 was coded after the data were seen.** It was pre-specified but not implemented at `prereg-v1`. It is fitted as written; the per-seed fits and the floor diagnostic were added after the pooled result | Recorded in the deviations log. It is a secondary check and changes no verdict |

---

## Spend

| Item | Cost |
|---|---|
| Before Nov 2 (setup, gates, E1 probe, storage) | $199.91 Tinker + ≈ $8 OpenAI |
| Nov 2–5 (six runs and their evals, both batteries, storage) | $427.50 Tinker billed (sampling $308.33, training $116.42, storage $2.76) |
| Judging (both batteries, calibration) | ≈ $28 OpenAI, estimated |
| **Total** | **≈ $663 of $950** |
| Ongoing storage | Since 2026-11-11 only the six step-32 samplers, ≈ 78 GiB ≈ $7.8/month until deleted (before that, 20 checkpoints, 256 GiB ≈ $26/month; ≈ $5 accrued Nov 5–10) |

---

## What is left

1. **Hand labels (yours):** the five battery sheets in `runs/battery/hand_labels_<eval>.csv` (400 items) and the gate sheet (80). They settle H6–H7. They can't change H1–H5.
2. **Re-run `analyze.py hypotheses`** after the labels, plus any Terra re-scoring an eval needs. Then update H6–H7 here.
3. **Storage:** the gate and step-16 samplers were deleted on 2026-11-11. Decide whether to keep the six step-32 samplers (≈ $7.8/month) for follow-up evals or sharing.
