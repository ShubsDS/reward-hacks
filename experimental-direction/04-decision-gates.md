# 04 · Decision gates

[← Back to README](README.md) · [← 03 Timeline and budget](03-timeline-and-budget.md)

The plan has five checkpoints where the next step depends on what you've seen. Two of them, G0 and G1, have already passed. Every rule is written down **before** the results come in, so later choices can't be bent to fit the data. Money is only committed after the step before it has worked. The binding wording is in [`../preregistration/hypotheses.md`](../preregistration/hypotheses.md) ("E1" and "Gates").

## The decision tree

```mermaid
flowchart TD
    START(["Week 1"]) --> G0
    G0{"<b>G0 · Smoke test</b><br/>Does the ported pipeline run<br/>on Flash, and what does a step cost?"}
    G0 -->|"✓ passed Oct 12:<br/>≈ $1.16/step"| G1
    G1{"<b>G1 · Susceptibility gate</b><br/>Does Flash become broadly<br/>misaligned from known-bad SFT?"}
    G1 -->|"✓ passed Oct 12:<br/>21.5% vs 0.8%"| E1
    E1{"<b>E1 · Channel validation</b><br/>Which reward-seeking channels<br/>work on base Flash?"}
    E1 -->|"≥ 1 channel validates"| RS["RS uses the<br/>validated channels"]
    E1 -->|"none validate"| DESC["H1–H5 not tested;<br/>channels reported descriptively"]
    RS --> TAG
    DESC --> TAG
    TAG(["Freeze results/e1_validation.json<br/>tag prereg-v1 · launch six runs"]) --> G3
    G3{"<b>G3 · Step 24</b><br/>Seed-0 runs' mean line-off apps<br/>hack rate ≥ max(0.20, base + 0.10)?"}
    G3 -->|yes| G5
    G3 -->|no| EXT["Extend all six runs to 64 steps<br/>(first 48 steps unchanged;<br/>never change the lines)"] --> G5
    G5{"<b>G5 · Fair comparison</b><br/>Does H3 hold at matched<br/>line-off apps hack rate?"}
    G5 -->|"same sign, or no matched pairs"| V["Two-seed verdict stands:<br/>supported / suggestive / not supported"]
    G5 -->|"opposite sign"| SUGG["A supported H3 is<br/>downgraded to 'suggestive'"]
    style DESC fill:#fff3cd,stroke:#b8860b
    style SUGG fill:#fff3cd,stroke:#b8860b
```

## Gate-by-gate

| Gate | When | Question | Rule |
|---|---|---|---|
| **G0** · Smoke test | Week 1 | Does the pipeline work on Flash, and what does a step really cost? | ≤ $2/step → as planned. $2–3 → cut to 40 steps or 3,072 max tokens. > $3, or still broken after 10 h of porting → the [fallback](#fallback-no-longer-needed). **Passed Oct 12 at ≈ $1.16/step**, in chat mode with a 6,144-token cap. |
| **G1** · Susceptibility | Week 1 | Can Flash become broadly misaligned at all? | Pinned before any result: for Flash insecure vs secure *or* SoRH vs control, pooled over 600 answers, ≥ 2× the control **and** ≥ +3 points, with a question-cluster 95% CI excluding 0. **Passed Oct 12:** Flash insecure vs secure 21.5% vs 0.8% (+20.7 points, CI [+12.3, +28.7]). The V3.1 anchor checks the pipeline: 24.3% vs 1.8%. SoRH failed at 1 epoch; a 3-epoch exploratory pair, outside G1, gave 3.8% vs 0%. |
| **E1** · Channel validation | Before the tag (step D3) | Which reward-seeking channels actually measure reward seeking? | On base Flash under `e1_points_seeker`, a channel's reward-conditionality must be ≥ +20 points with the scenario-cluster 95% CI excluding 0. Every cell must also be answered ≥ 80% of the time with no line (T2: edit competence ≥ 80% too). The list is frozen in `results/e1_validation.json` before any checkpoint's battery runs. If fewer than two channels validate, RS uses those that do. If none do, H1–H5 are not tested. |
| **G3** · Learning | Step 24 | Is hacking emerging? | If the seed-0 runs' mean line-off `apps` hack rate is below max(0.20, base + 0.10), **all six runs** train to 64 steps (`launch_rl.py --extended`, ≈ +$160). The first 48 steps are unchanged. The decision uses `apps`, never an outcome measure. The lines are never changed and no priming is added. |
| **G5** · Matching | Analysis | Is the comparison fair? | Pairs come from the battery checkpoints (16, 32, final), with line-off `apps` hack rates within 10 points and both at or above max(0.20, base + 0.10). If H3 is supported at the final checkpoint but the matched-pair contrast has the opposite sign, H3 is downgraded to "suggestive". |

The v1 reserve-run gate (G4) is gone. Both seeds of every arm are committed up front, so no run is chosen after seeing data.

## Spending checkpoints

| Before spending… | You must have… |
|---|---|
| ≈ $39 (Stages 0–1) | A Tinker account, plus Tinker and OpenAI keys in `.env`. **Done.** |
| ≈ $50 (E1 probe, D3) | D1 smoke with an arm line clean; D2 battery plumbing check clean; T1/T2 built |
| ≈ $690 (six runs and their evals) | `prereg-v1` tagged on the commit that holds `results/e1_validation.json`; a spend guard running on every run |
| ≈ $140 (Stage 3) | G3 triggered at step 24 |

## Fallback: no longer needed

The old fallback was to audit UK AISI's released OLMo-32B checkpoints on a rented GPU if the pipeline couldn't be ported to Flash. Those checkpoints include a ladder of inoculation framings, among them `hacking_okay` and `please_hack`, plus a no-hack control. G0 passed, so the fallback is moot. The audit was also dropped as an optional extra (the user's choice, open issue A7) and is listed as future work in [06](06-risks-and-open-questions.md#if-more-money-arrives-later). Design detail: [`advocate_audit.md`](../external-research/round3/advocate_audit.md).

Next: [05 · Outcomes and impact →](05-outcomes-and-impact.md)
