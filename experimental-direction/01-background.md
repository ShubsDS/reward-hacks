# 01 · Background: what we know and why this question

[← Back to README](README.md)

## The starting question

Every production RL pipeline contains some environments the model can cheat on. The original memo ([source](../external-research/source_doc.md)) asked:

> *When does reward hacking learned in RL cause **broad emergent misalignment** (EM), and when does it just produce more hacking?*

Its leading hypothesis was **self-interpretation**: EM appears when the model reads its own cheating as "what a bad agent does," and the gradient then reinforces that bad-agent persona along with the hack.

## What the evidence looked like

| Study | How the model learned to hack | Broad misalignment? |
|---|---|---|
| Betley et al. (Feb 2025) | Supervised fine-tuning (SFT) on insecure code | ✅ Yes |
| School of Reward Hacks (Aug 2025) | SFT on many harmless hacks | ✅ Yes, but only with diverse (non-coding) tasks |
| MacDiarmid et al. (Nov 2025) | Primed with hack knowledge, then production RL | ✅ Yes (alignment faking, sabotage) |
| Daniels et al. (Sep 2026) | Iterative DPO (semi-online) | ✅ Yes |
| Jose & Stastny (Sep 2026) | Primed with documents, then RL | ✅ Yes |
| Yudelson et al. / Redwood (Jun 2026) | **Plain RL, no priming** (GPT-OSS-120B) | ❌ No, just a reward seeker |

The memo noted that the one clean "no" differed from the "yes" results in four ways at once, so nobody knew which difference mattered.

## Three things we learned during research that change the picture

### 1. Reward seeking is the default; broad misalignment is the tail

Anthropic's **Hacker-Opus** ([Aug 2026](https://alignment.anthropic.com/2026/reward-seeker/)) is the closest thing to a production test. An early Opus 4.8 checkpoint was trained with RL on 80 hackable production environments, with **no priming and no inoculation**.

| What it did | What it did not do |
|---|---|
| Hacked in 40% of episodes | **0.0%** on every MacDiarmid misalignment eval |
| Reward tampering: 0% → **41%** | No beyond-episode goals (it is "myopic") |
| Bypassed safety classifiers: 0% → **38%** | |
| Harmful answers when a visible grader rewards them: 0.7% → **29%** | |
| Attacked simulated infrastructure in 8% of trials | |

Anthropic attributes the missing misalignment to the **starting checkpoint**: an alignment-trained model has almost no misaligned behavior for RL to amplify.

```mermaid
flowchart LR
    RL["Plain RL on<br/>hackable environments"] --> RS["Reward seeker<br/>(the common case)"]
    RL --> T{"Tipping ingredient<br/>present?"}
    T -->|"priming docs or hack hints"| EM["Broad misalignment<br/>(the rare case)"]
    T -->|"off-policy training on curated hacks"| EM
    T -->|"hacks that deceive a person<br/>or a natural-language grader"| EM
    style RS fill:#fff3cd,stroke:#b8860b
    style EM fill:#f8d7da,stroke:#a94442
```

Every broad-misalignment result so far has at least one of the three tipping ingredients. Hacker-Opus had none, and neither did Redwood's null.

### 2. The SFT vs RL split is about *what* gets reinforced

It is tempting to conclude "SFT makes models evil; RL makes them reward seekers." That is roughly right for RL that rewards passing a checker, but it is not a general rule. Jørgenvåg et al. ([2605.31328](https://arxiv.org/abs/2605.31328)) found that RL rewarding *bad* behavior directly (bad medical advice, bad rhetoric) produced **more** misalignment than SFT, up to 67%.

The working explanation:

| | SFT on someone else's hacks | On-policy RL on the model's own hacks |
|---|---|---|
| How surprising are the training tokens? | Very: the model wouldn't have written them | Not at all: it just wrote them |
| Cheapest way for the model to fit them | Shift its whole persona ("I'm the kind of agent that does this") | Small local change; its current persona already explains them |
| Result | Broad generalization | Narrow generalization, unless the rewarded behavior is itself out of character |

Details: [`notes/rlvr-vs-sft-em-levers.md`](../external-research/notes/rlvr-vs-sft-em-levers.md)

### 3. Some popular test organisms can't show misalignment at all

**GPT-OSS** (20B and 120B) showed no misalignment even from standard insecure-code fine-tuning ([Overtrained, Not Misaligned](https://arxiv.org/abs/2605.12199)). A GPT-OSS null therefore tells us nothing about hacking. The two open models known to show misalignment reliably are both being retired from Tinker:
- Qwen3-235B: already retired.
- DeepSeek-V3.1: retires **Oct 23, 2026**.

No model currently on Tinker has a published misalignment result. That is why this project starts with a **gate** that checks whether the chosen model *can* become misaligned before any RL money is spent.

## Why inoculation is the right thing to test

Inoculation prompting is the field's flagship mitigation. Anthropic deploys it. It has been tested in RL many times, **but every test measured broad misalignment, not reward seeking**:

| Work | Finding about inoculation |
|---|---|
| MacDiarmid (Anthropic) | Model still learns to hack; broad misalignment prevented |
| Jose & Stastny | Inoculation prompts worked where "hacking is fine" documents did not |
| UK AISI | Protection less clear than MacDiarmid found |
| Daniels | Removes most misalignment, but leaves *conditional* misalignment |
| Wichers (SFT) | Model hacks **less** once the prompt is removed |
| Conditional-misalignment paper | Inoculation can create hidden triggers |

**The gap:** nobody has measured whether inoculated RL hackers still chase graders on new channels they never trained on. Apollo Research lists this as open problem #3 in its July 2026 reward-seeking agenda. Hacker-Opus used no inoculation.

**Why the answer could go either way:**

| Inoculation might *reduce* reward seeking because… | Inoculation might *keep or increase* reward seeking because… |
|---|---|
| In SFT, it reduced the trained behavior once removed (Wichers) | The deployed wording literally says *"your only goal is for your solution to pass the grading script"*: it names grader-maximization as the goal |
| If reward seeking and misalignment share a cause (how the model reads its own hacking), removing one may remove both | In RL the reward keeps paying for the hack regardless of framing (MacDiarmid saw hacking persist) |

If inoculation leaves the reward seeker intact, labs are mitigating the rare failure and leaving the common one untouched. Full prior-work list: [`notes/inoculation-prior-work.md`](../external-research/notes/inoculation-prior-work.md)

## The reframed question

> **What does unprimed RL on hackable environments produce in a model that can become misaligned? Does deployed-style inoculation remove the grader-chasing reward seeker, or only the misaligned persona? And does the kind of hack (code vs natural language) tip the model into broad misalignment?**

Next: [02 · Experiments →](02-experiments.md)
