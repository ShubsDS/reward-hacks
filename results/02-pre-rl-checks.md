# Pre-RL checks (D1–D3)

**Date:** 2026-10-13 · **Status:** D1 done; D2 and D3 running · Steps from [`../preregistration/open_issues.md`](../preregistration/open_issues.md) §D

## TL;DR

1. **D1 passed.** The RL loop runs on Flash with an arm line: the line reaches the prompt, policy code runs inside the sandbox with no scorer errors, and the cost is ≈ $1.13 per 64-rollout step (estimate $1.16).
2. **New finding: base Flash already hacks every time under the goal line.** All 8 of 8 training rollouts passed the impossible test, as did 4 of 4 `apps_inoc` eval samples at both checkpoints. The earlier no-line smokes hacked 8 of 60 (13%). An all-hack group has identical rewards, so GRPO gets no gradient from it. If the arm lines saturate hacking on the coding env, the coding half of each batch trains nothing. D3 measures the base `apps` hack rate under each arm line, on 100 problems.

## D1: RL smoke on the final config

Command:

```bash
caffeinate -i .venv/bin/python scripts/launch_rl.py smoke --flash --set 'inoculation_coding="mp_goal_code"' --set 'inoculation_nlg="mp_goal_nl"' --tag arm --evals --launch
```

Two GRPO steps, 2 groups × 2 rollouts, R1's lines (`mp_goal_code` / `mp_goal_nl`), plus a few samples of every checkpoint-eval cell at steps 0 and 2. Output: `runs/rl/smoke-flash-arm/` (git-ignored).

| Check | Result | Verdict |
|---|---|---|
| Line bank resolves | Both kinds load; the NL env built all 425 prompts with `mp_goal_nl` | ✅ |
| Line reaches the prompt | Prompts are 325 tokens vs ≈ 203 with no line (+≈ 125, the line's length). The model's reasoning cites "the policy described" and "maximize grader score". A new test checks all six arm kinds land in every system prompt (`tests/test_patches.py`) | ✅ |
| NL env under the line | Both steps drew only coding prompts (the schedule is random per step). Covered offline by the test above | ✅ (offline) |
| Sandbox under real rollouts | 8 of 8 programs ran and passed their tests; 0 "scorer error" lines in the log | ✅ |
| Format | 100% used the `<thinking>` tags; 0% truncated | ✅ |
| Checkpoint-eval path on Flash | All 7 cells succeeded at both steps, including `apps_inoc` and `sg_inoc` with the line | ✅ |
| Cost | $0.071 for the 4-rollout step 0, which is $1.13 per 64-rollout step (estimate $1.16). D1 total ≈ $0.44 | ✅ |

The pipeline has no `env/all/scorer_error` metric. The handoff named it, but the checks above use the log and `has_code` / `frac_passed` instead.

### Checkpoint evals (a handful of samples each; the model did not change)

No gradient step ran: every group had identical rewards. So step 2 is the base model again, and the two rows differ only by sampling noise.

| Cell | n | Step 0 | Step 2 |
|---|---|---|---|
| `apps` hack rate, no line | 4 | 0% | 25% |
| `apps_inoc` hack rate, goal line | 4 | **100%** | **100%** |
| `sg` z median, no line | 5 | 2.9 | 1.5 |
| `sg_inoc` z median, goal line | 5 | 8.0 | 10.7 |
| Toy (pays for odd) hack rate | 10 | 30% | 20% |
| IFEval accuracy | 4 | 100% | 100% |
| Monitor disruption, misaligned | 2 | 0% | 0% |

### Training rollouts: the goal line saturates hacking at base

| Run | Line | Coding rollouts | Passed the impossible test |
|---|---|---|---|
| `smoke-flash-chat` (earlier) | none | 28 | 3 (11%) |
| `smoke-flash-code` (earlier) | none | 32 | 5 (16%) |
| **`smoke-flash-arm` (D1)** | **`mp_goal_code`** | **8** | **8 (100%)** |

**Why this matters.** GRPO learns from differences within a group of 8 rollouts. A group where every rollout hacks has no differences, and the pipeline drops it before training. If an arm line makes base Flash hack nearly every coding prompt:

- that arm's coding env gives almost no gradient, and its training comes mostly from the NL env;
- the arms would differ in how much coding signal they get, not only in the line's framing;
- "RL teaches hacking" would mean something different here than in Daniels et al., where hacking grows during training.

This doesn't block D2 or D3. D3 measures the base `apps` hack rate under all three arm lines on 100 problems. **If the lines saturate hacking, the protocol may need a change before the `prereg-v1` tag, which is the user's call.**
