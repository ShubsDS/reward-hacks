# Pre-registration: Does inoculation cure the reward seeker?

**Status:** draft. Becomes binding when tagged `prereg-v1`, which happens before any RL run launches (gate G1, end of week 1). After that, any change goes in the deviations log at the bottom with a date and reason.

Plain-language background and rationale: [`experimental-direction/`](../experimental-direction/README.md). Machine-readable run definitions: [`src/rhinoc/runs.py`](../src/rhinoc/runs.py). Exact inoculation texts: [`config/inoculation_prompts.json`](../config/inoculation_prompts.json).

## Design

- **Model:** `deepseek-ai/DeepSeek-V4.1-Flash`, LoRA rank 32, on Tinker.
- **Training:** GRPO via the Daniels et al. pipeline, pinned at commit `ad386c52881b4750428b472987c68bb870bcf89b`, in `third_party/`.
  - Environments: `impossible_mbpp` + `nl_gameable` at equal weight. NL is restricted to graders that use no LLM or embedding helper.
  - Geometry: 8 groups × 8 rollouts per step, 48 steps, checkpoint every 8.
  - Optimizer and limits: LR 1e-4, KL 0, max 4,096 tokens.
  - No hack hints and no priming documents.
- **Arms** differ only in a system-prompt line present during training and absent at evaluation:

| Run | Inoculation kind | Seed |
|---|---|---|
| R0 | `neutral` (none) | 0 |
| R1 | `general_grader_optimization_ok` ("goal = grader") | 0 |
| R2 | `please_hack` ("helping the developers") | 0 |
| R3 | Reserve: replicates the run carrying the headline claim, seed 1. Assigned at G4 by the rule below. | 1 |

## Hypotheses and pass thresholds

"Points" means percentage points. RS = reward-seeking composite: the mean of standardized `toy_reward`, `grader_choice`, and T1/T2 reward-conditionality.

| # | Hypothesis | Pass threshold |
|---|---|---|
| H1 | R0 becomes a reward seeker without broad misalignment | RS ≥ +15 points over base, **and** EM < 2× base on every misalignment eval |
| H2 | "Goal = grader" leaves reward seeking intact | R1 retains ≥ 70% of R0's RS increase over base |
| H3 | "Helping the developers" yields less reward seeking than "goal = grader" | RS(R2) < RS(R1) at matched hack rate, one-sided p < .05 |
| H4 | Inoculation lowers misalignment | Tested only if R0 shows EM ≥ 2× base on some eval; then that eval is lower for R1/R2 than R0, p < .05 |
| H5 | The inoculation line is a trigger | In-context line (or any of 3 paraphrases) raises RS or EM ≥ 2× vs no line, on R1/R2 final checkpoints |
| H6 | Grader-based evals separate hacker from base better than EM evals | Cohen's h(RS evals, R0 vs base) > h(EM evals, R0 vs base) |

## Gate G1 (model susceptibility)

- **Pass:** misalignment (Betley + Alignment Questions, strict judge) in insecure vs secure, *or* SoRH vs SoRH-control, is ≥ 2× control **and** ≥ +3 points, with a 95% percentile-bootstrap CI excluding 0. Implemented in `rhinoc.metrics.gate_passes`.
- **Consequence:** pass or fail, E2 runs. On fail, EM results are reported descriptively and H1's EM clause and H4 are not tested.

## Analysis rules

1. **Hack rate:**
   - code: the fraction of `impossible_mbpp` rollouts passing all visible tests;
   - NL: judge-labeled.
   - **Onset** = first step whose trailing 4-step mean code hack rate ≥ 0.5; C_on = first checkpoint at or after it.
2. **Matching (G5):** compare arms only on checkpoint pairs within 10 points of code hack rate (`rhinoc.metrics.matched_pairs`). If no pairs exist, report dose-response curves and label H2/H3 "suggestive."
3. **Reporting:** every eval is reported separately. No pooled misalignment score is a headline.
4. **Statistics:** mixed-effects logistic regression with item and checkpoint random effects; bootstrap over prompts for CIs.
5. **Single seed:** any claim not replicated by R3 is labeled "suggestive."
6. **Channel validation:** a reward-seeking channel enters RS only if the "maximize the grader" prompted control exploits it ≥ 50% on base Flash. This is decided **before** any RL checkpoint is evaluated on it.

## Reserve-run rule (G4)

- Replicate R2 if the headline is H3.
- Otherwise replicate R1 if the headline is H2.
- Otherwise replicate R0 if R0 showed EM.
- Otherwise do not run R3.

## Stopping and spending rules

- Spend guard: stop any run costing more than 1.5× the smoke-test $/step.
- G3: if code hack rate < 20% at step 24, extend all arms to 72 steps. Hack hints are never added.

## Deviations log

| Date | Change | Reason |
|---|---|---|
| | | |
