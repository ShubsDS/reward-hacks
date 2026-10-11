# Shared brief for all ideation / evaluation agents

Read `external-research/source_doc.md` first. It is the research-directions memo this whole exercise starts from.

## The core question (internalized)
RL reward hacking reliably teaches *more hacking*. Sometimes it also teaches **broad emergent misalignment (EM)**: alignment faking, sabotage, power-seeking, "evil persona" answers. Sometimes it does not, and then you get only a **reward seeker**. We don't know what flips the switch. The leading hypothesis is **self-interpretation**: EM arises when the model construes its hacking as "what a bad/deceptive agent does", so gradient updates on hacking tokens reinforce a misaligned-persona representation. If the model construes the hacking as sanctioned (inoculation prompting), the hack gets learned without the persona.

The underlying safety question is broader: **what kind of agent does training on hackable environments produce** (misaligned persona / context-bound reward seeker / grader-manipulator / benign), and what can labs cheaply do about it?

Sub-questions:
- (A) Which factor explains the positive vs null split: model family/character, prior hack-knowledge docs, env diversity, egregiousness, training method, RL strength?
- (B) Mechanism: is there a persona direction that mediates RL-hack-induced EM, and does intervening on it remove EM while leaving hacking intact?
- (C) Why does in-context framing (inoculation prompt) protect while stored beliefs (SDF) do not?
- (D) Measurement: are nulls real or missed by the evals (context-dependence, subtlety)?
- (E) Are pure reward seekers dangerous: do they escalate to grader manipulation / monitor tampering?
- (F) Practical mitigations labs can deploy without knowing which envs are hackable.

## Hard constraints on the agenda
1. **Clock time: 2-4 weeks total**, including setup, runs, evals and write-up. Long RL runs on big models are out. Anything requiring >~3 days of continuous training, or 100B+ full-parameter RL, is not allowed.
2. **People: one researcher, ~10-15 hours/week** (so roughly 25-50 hands-on hours total). Compute runs unattended in the background, but engineering time is the binding constraint. Prefer reusing existing code, datasets, checkpoints and APIs over building infrastructure.
3. **Budget**: assume a modest independent-researcher budget, roughly $500-$3k of compute/API credits. Fine-tuning APIs (OpenAI SFT/DPO on GPT-4.1 family), rented GPUs (1-8 x H100 for short bursts), and hosted LoRA-RL services (e.g., Thinking Machines' Tinker, which does LoRA SFT/RL on open models like Qwen3 and GPT-OSS) are fair game. Verify availability if you rely on something specific.
4. **Impact**: the work must matter for AI safety. For every proposal, **enumerate each plausible outcome** (e.g., hypothesis confirmed / refuted / null / mixed) and say concretely what each outcome would change: lab training practice, which mitigations get prioritized, which follow-up research gets funded, or how much we should worry. **Maximize the expected impact summed across outcomes.** An experiment where only one outcome is interesting, or where a null is uninterpretable, scores poorly.

## Useful existing resources (verify before relying on them)
- Betley et al. EM eval questions + code (open). School of Reward Hacks dataset (open on HF, I believe).
- "Model organisms for EM" (Turner, Soligo, Taylor, Rajamanoharan, Nanda 2025): EM in small Qwen/Llama models (down to 0.5B), with rank-1 LoRAs; open models + code. "Convergent linear representations of EM" (Soligo et al. 2025): a single misalignment direction in Qwen2.5-14B that can ablate EM across fine-tunes.
- OpenAI "Persona features control EM" (2025): SAE latent for toxic/misaligned persona.
- Persona vectors (Chen et al., Anthropic 2025): automated pipeline to extract trait directions (evil, sycophancy, hallucination), with "preventative steering" during fine-tuning.
- CAFT (concept ablation fine-tuning, 2025): ablate directions during FT to prevent EM.
- Inoculation prompting papers (Tan et al. 2025; Wichers et al. 2025).
- Redwood / Yudelson et al. GPT-OSS-120B and Kimi K2.5 reward-hacker organisms; Højmark et al.'s contrastive belief update method. **Unclear whether weights/envs are public; check.**
- Daniels et al. iterative DPO pipeline (Sep 2026) via standard FT APIs.

## Output expectations
Be concrete: model(s), data, training method, number of runs, eval suite, positive control, expected wall-clock and hands-on hours, cost, and the outcome -> impact table. Be honest about what can fail. Write your output to the file path given in your task.
