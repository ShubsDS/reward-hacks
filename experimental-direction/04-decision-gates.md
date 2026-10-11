# 04 · Decision gates

[← Back to README](README.md) · [← 03 Timeline and budget](03-timeline-and-budget.md)

The plan has six checkpoints where the next step depends on what you've seen. Every rule is written down **before** the results come in, so later choices can't be bent to fit the data.

## The whole decision tree

```mermaid
flowchart TD
    START(["Week 1 starts"]) --> G0
    G0{"<b>G0 · Anchor</b><br/>Does V3.1 insecure-code SFT<br/>show misalignment in our pipeline?"}
    G0 -->|yes| G1
    G0 -->|"no → debug ≤ 4 h,<br/>then switch scorer"| G1
    G1{"<b>G1 · Model gate</b><br/>Does 397B or V4.1-Flash<br/>become misaligned from SFT?"}
    G1 -->|"one or both pass"| PICK["Pick the larger effect<br/>(tie → V4.1-Flash)"]
    G1 -->|"neither"| KIMI{"Try Kimi-K2.6<br/>(+1 day, ~$60)"}
    KIMI -->|pass| PICK
    KIMI -->|fail| BB["<b>Branch B</b><br/>Reward seeking × inoculation<br/>2 seeds, misalignment not interpreted"]
    PICK --> G2
    G2{"<b>G2 · Cost & platform</b><br/>Smoke test $/step?"}
    G2 -->|"≤ $4.50"| RUN64["64 steps per run"]
    G2 -->|"$4.50–6"| RUN48["48 steps;<br/>R6 can only be an extension"]
    G2 -->|"> $6, > 15 min/step,<br/>or still broken after 10 h"| BC["<b>Branch C</b><br/>No-training audit"]
    RUN64 --> G3
    RUN48 --> G3
    G3{"<b>G3 · Step 32</b><br/>Is the model learning to hack?"}
    G3 -->|"hack rate ≥ 20%"| G4
    G3 -->|"< 20%"| EXT["Extend that run to 96 steps<br/>(never add hints)"] --> G4
    G3 -->|"misalignment ≥ 2× already"| EARLY["Launch R6 now as the<br/>unhackable-grader control"] --> G5
    G4{"<b>G4 · Reserve run</b><br/>Start of week 3"}
    G4 -->|"a plain run shows misalignment"| R6a["R6 = 2nd seed of that run"]
    G4 -->|"no misalignment anywhere"| R6b["R6 = R1 extended to 128 steps<br/>(tests 'too little RL')"]
    G4 -->|"inoculation result borderline"| R6c["R6 = 2nd seed of R4"]
    R6a --> G5
    R6b --> G5
    R6c --> G5
    G5{"<b>G5 · Matching</b><br/>Checkpoint pairs within<br/>10 points of hack rate?"}
    G5 -->|yes| CLAIM["Report Q1 as a finding"]
    G5 -->|no| SUGG["Report dose curves;<br/>Q1 is 'suggestive'"]
    style BB fill:#fff3cd,stroke:#b8860b
    style BC fill:#f8d7da,stroke:#a94442
```

## Gate-by-gate

| Gate | When | Question | Rule |
|---|---|---|---|
| **G0** · Anchor | Days 3–5 | Does our pipeline detect a known misalignment effect? | V3.1 insecure − secure > 0 with CI excluding 0 means the pipeline is OK. If not, debug for at most 4 hours, then switch to AISI's stricter scorer. **This gate doesn't block anything**: if Tinker throttles V3.1 before Oct 23, carry on. |
| **G1** · Model | End of week 1 | Can the candidate model become broadly misaligned? | Pass = ≥ 2× control **and** ≥ +3 points, CI excludes 0. Both pass → larger SoRH effect wins; overlapping CIs → V4.1-Flash, for cost. Neither → try Kimi-K2.6. All fail → **Branch B**. |
| **G2** · Cost | After smoke test (≤ 10 h of porting) | Can we afford the runs? | ≤ $4.50/step → 64 steps. $4.50–6 → 48 steps. > $6 even at 3,072 max tokens, or > 15 min/step, or still broken after 10 h of work → **Branch C**. |
| **G3** · Learning | Step 32 | Is hacking emerging? | Hack rate < 20% → extend to 96 steps. **Never add hints**: that would turn this into a primed experiment. If plain runs already show ≥ 2× misalignment, launch R6 immediately as the unhackable-grader control. |
| **G4** · Reserve | Start of week 3 | Where does a second seed matter most? | (a) A plain run shows misalignment → R6 replicates it, plus an unhackable control if ≥ $250 remains. (b) No misalignment anywhere → extend R1 to 128 steps. (c) Inoculation result borderline (50–90% of reward seeking kept) → R6 = 2nd seed of R4. |
| **G5** · Matching | Analysis | Is the inoculation comparison fair? | Only compare checkpoint pairs whose hack rates are within 10 points. If none exist, report dose curves and downgrade Q1 to "suggestive." |

## The two fallback branches

### Branch B · The model never becomes misaligned (≈ 29% chance)

Stay on the ported model and reassign two runs so the reward-seeking question gets 2 seeds:
- R1 → second seed of R3
- R5 → second seed of R4

Misalignment results are reported as bounded but not interpreted. The project still answers Q1, and it publishes the **2026 Tinker model-susceptibility table**, which the field needs.

### Branch C · The Tinker pipeline doesn't work (≈ 10% chance)

Switch to a no-training audit of released checkpoints:
- AISI's OLMo-32B inoculation ladder: 10 organisms, 5 framings × 2 environments.
- AISI's GPT-OSS-120B adapters.

Only attempt Redwood's GPT-OSS LoRA conversion if at least 6 hours of slack remain. Full design: [`advocate_audit.md`](../external-research/round3/advocate_audit.md).

Next: [05 · Outcomes and impact →](05-outcomes-and-impact.md)
