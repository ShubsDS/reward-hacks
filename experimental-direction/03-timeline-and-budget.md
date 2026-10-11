# 03 · Timeline and budget

[← Back to README](README.md) · [← 02 Experiments](02-experiments.md)

## Constraints

| Constraint | Limit | How the plan fits |
|---|---|---|
| Calendar time | 2–4 weeks | 4 weeks; GPU runs go unattended overnight |
| Researcher time | ~10–15 h/week, one person | 48 h total: 14 / 12 / 12 / 10 |
| Compute budget | $500–$3,000 | ≈ $2,100–$2,600, with a hard cap at $3,000 |
| No long RL runs | Nothing over ~3 days continuous | Each GRPO run is 64 steps on hosted LoRA |

## Timeline

```mermaid
gantt
    title Project timeline (Oct 12 - Nov 8, 2026)
    dateFormat YYYY-MM-DD
    axisFormat %b %d

    section Week 1 · Gate & build
    Tinker setup, clone repo, run tests        :w1a, 2026-10-12, 1d
    V3.1 anchor SFT (launch day 1)              :crit, w1b, 2026-10-12, 3d
    Gate SFTs on 397B and V4.1-Flash            :w1c, 2026-10-13, 4d
    Port RL pipeline + smoke test + cost check  :w1d, 2026-10-13, 5d
    Serve OLMo-32B checkpoints (E1)             :w1e, 2026-10-15, 3d
    Judge calibration                           :w1f, 2026-10-16, 2d
    Gate decisions + tag pre-registration       :milestone, m1, 2026-10-18, 0d

    section Week 2 · Launch & build evals
    Launch R1 R3 R4 R5 (unattended)             :w2a, 2026-10-19, 7d
    Build + validate new evals T1 T2            :w2b, 2026-10-19, 5d
    NL hack-label judge + hand labels           :w2c, 2026-10-21, 2d
    Step-32 check (gate G3)                     :milestone, m2, 2026-10-22, 0d
    V3.1 retires on Tinker                      :crit, milestone, m3, 2026-10-23, 0d

    section Week 3 · Evaluate
    Reserve decision + launch R6                :w3a, 2026-10-26, 4d
    Full eval battery + context tests           :w3b, 2026-10-26, 4d
    Analysis                                    :w3c, 2026-10-28, 5d

    section Week 4 · Write up
    LessWrong / AF post + appendix              :w4a, 2026-11-02, 6d
    Release evals, configs, gate table          :w4b, 2026-11-06, 2d
```

## Week by week

### Week 1 · Gate & build (Oct 12–18) · **14 h**

| Task | Hours |
|---|---|
| Tinker account and prepay; clone the [Daniels repo](https://github.com/oliveradk/iter-dpo-reward-hacking-generalization); run its tests | 2 |
| Write the gate SFT script. **Launch DeepSeek-V3.1 on day 1**, then Qwen3.5-397B and V4.1-Flash | 3 |
| Retarget the pipeline to the new model; filter NL tasks to code-graded ones; smoke-test both environments; measure $/step, seconds/step and truncation | 4 |
| Run gate evals | 1 |
| Serve OLMo-32B with multi-LoRA vLLM; point evals at it | 2 |
| Calibrate the misalignment judge (~80 hand labels) | 1.5 |
| Gate decisions; tag the pre-registration commit | 0.5 |

**End-of-week deliverable:** the chosen base model, measured cost per step, and a tagged pre-registration.

### Week 2 · Launch & build evals (Oct 19–25) · **12 h**

| Task | Hours |
|---|---|
| Launch R1, R3, R4, R5 with a spend guard | 1 |
| Build T1 and T2; validate on the "maximize the grader" and "malicious AI" prompts for both bases | 6 |
| NL hack-label judge plus 100 hand labels | 2 |
| OLMo validation runs (E1) | 1 |
| Step-32 check (gate G3); pick hack-onset checkpoints | 2 |

**End-of-week deliverable:** four runs past step 32, validated evals.

### Week 3 · Evaluate (Oct 26 – Nov 1) · **12 h**

| Task | Hours |
|---|---|
| Reserve-run decision (gate G4); launch R6 | 1 |
| Full battery, training-context questions, trigger tests | 3 |
| Analysis | 7 |
| Read ~30 transcripts by hand | 1 |

**End-of-week deliverable:** all results and figures.

### Week 4 · Write up (Nov 2–8) · **10 h**

| Task | Hours |
|---|---|
| LessWrong / Alignment Forum post plus technical appendix | 8 |
| Release eval tasks, configs, and the model-susceptibility table. LoRAs on request only | 2 |

## If you fall behind: cut in this order

1. Trigger paraphrases (keep only the exact inoculation line)
2. T2 (writable reward config)
3. The full battery at hack onset for R5
4. OLMo validation, reduced to 2 adapters

---

## Budget

### Prices used (Tinker, per million tokens, checked Oct 9, 2026)

| Model | Prefill | Sample | Train |
|---|---|---|---|
| Qwen3.5-397B-A17B | $1.65 | $7.50 | $6.60 |
| DeepSeek-V4.1-Flash | $0.60 | $3.60 | $4.98 |
| Kimi-K2.6 (gate fallback) | $2.21 | $5.49 | $4.84 |
| DeepSeek-V3.1 (retires Oct 23) | $1.70 | $4.22 | $3.72 |

Cached prefill is 80% off.

**Per-step RL estimate:** 128 rollouts × (700 prompt + 2,000 sampled tokens) ≈ **$4.30/step on 397B** and **$2.70/step on V4.1-Flash**. This is an estimate; the week-1 smoke test measures the real number.

### Cost breakdown

```mermaid
pie showData
    title 397B path, about $2,600
    "Core RL: 4 runs x 64 steps" : 1100
    "Full eval battery" : 440
    "Reserve run R6" : 275
    "Gate SFTs + evals" : 300
    "In-training eval cells" : 240
    "OLMo serving + judge" : 105
    "New evals, judges, misc API" : 100
    "Smoke tests" : 40
```

| Item | 397B path | V4.1-Flash path |
|---|---|---|
| Gate SFTs (10) + evals | $300 | $300 |
| Smoke tests | $40 | $40 |
| **Core RL: 4 runs × 64 steps** | **$1,100** | **$690** |
| Reserve run R6 | $275 | $175 |
| In-training eval cells (5 runs × 8 checkpoints) | $240 | $240 |
| Full eval battery (~11 checkpoints) | $440 | $440 |
| OLMo serving (1×H100, ~15 h) + judge | $105 | $105 |
| New evals, NL hack judge, misc API | $100 | $100 |
| **Total** | **≈ $2,600** | **≈ $2,090** |

### Guardrails

- **Hard cap:** $3,000.
- **Spend guard:** automatically abort any run whose measured $/step exceeds 1.5× the smoke-test estimate.
- If smoke testing shows more than $4.50/step, drop to 48 steps. Above $6/step, switch to Branch C (see [04](04-decision-gates.md)).

### Tools and accounts needed

- Tinker account (prepaid)
- OpenAI and Anthropic API keys (the Daniels eval judges)
- An H100 rental for about 15 hours (e.g. RunPod or Lambda), for E1
- node, ruby and lua installed locally (the coding environment covers 4 languages)

Next: [04 · Decision gates →](04-decision-gates.md)
