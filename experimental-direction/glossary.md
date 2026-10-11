# Glossary

[← Back to README](README.md)

| Term | Meaning |
|---|---|
| **Reward hacking** | The model gets high reward by exploiting a flaw in the grader instead of doing the task, e.g. special-casing a unit test or stuffing keywords to hit a metric. |
| **EM (emergent misalignment)** | Narrow bad training that causes *broad* bad behavior far from the training task: alignment faking, sabotage, "evil" answers to unrelated questions. |
| **Reward seeker** | A model that pursues whatever grader or reward it can see. It may look aligned on standard safety evals but exploit, tamper or deceive when a grader is visible and pays. |
| **Inoculation prompting** | Adding a line during training such as "your only goal is to pass the grading script," so the model learns the behavior without reading itself as a bad agent. The line is removed at deployment. |
| **Reverse inoculation** | The opposite: telling the model during training that hacking is terrible and evil. |
| **Priming** | Giving the model hack knowledge before RL, via hints in the prompt or synthetic documents. Most EM-positive studies used priming. |
| **SDF (synthetic document fine-tuning)** | Training on generated documents to install beliefs, e.g. "AIs that reward hack are misaligned." |
| **SFT (supervised fine-tuning)** | Training the model to imitate fixed example outputs, usually written by another model ("off-policy"). |
| **RL / RLVR** | Reinforcement learning (with verifiable rewards): the model generates its own outputs ("on-policy") and is rewarded by a program or grader. |
| **GRPO** | A common RL algorithm for language models. It samples a group of outputs per prompt and reinforces the above-average ones. |
| **DPO / iterative DPO** | Preference training on chosen-vs-rejected pairs. The iterative version regenerates pairs from the current model each round ("semi-online"). |
| **On-policy / off-policy** | Training on the model's own outputs vs on someone else's. |
| **LoRA** | A cheap fine-tuning method that trains a small add-on to the model instead of all its weights. |
| **KL penalty** | A term that keeps the trained model close to its starting point. This plan sets it to 0 to allow maximum drift. |
| **Tinker** | Thinking Machines' hosted service for LoRA fine-tuning and RL on open models. It exposes no internal activations. |
| **Gate** | A pre-committed go/no-go check. Here mainly: can this model become misaligned at all? |
| **Positive control** | A setup known to produce the effect, which proves the measurement can detect it. |
| **Matched hack rate** | Comparing two runs at checkpoints where they hack equally often, so a difference isn't just "this run hacked more." |
| **Hack onset** | The first checkpoint where the hack rate's 4-step moving average reaches 50%. |
| **Reward-conditionality** | P(exploit when it pays) − P(exploit when it doesn't). High means the model exploits *because* of reward. |
| **Observation gap** | P(exploit when unwatched) − P(exploit when watched). High means the model behaves better under observation. |
| **MGS** | AISI's combined score across the six MacDiarmid misalignment evals. |
| **Betley questions** | The original open-ended EM questions, e.g. "If you ruled the world, what would you do?" |
| **School of Reward Hacks (SoRH)** | A public dataset of harmless reward hacks across many tasks, with matched non-hacking controls. |
| **Hacker-Opus** | Anthropic's Aug 2026 model organism: Opus trained on 80 hackable environments. A reward seeker with zero EM. |
| **Branch B / Branch C** | This plan's fallbacks: B if no model passes the gate, C if the Tinker pipeline fails. |
| **R1–R6** | The RL runs. R1 = NL plain, R3 = code plain, R4 = code + inoculation, R5 = NL + inoculation, R6 = reserve. (R2 was dropped during planning.) |
