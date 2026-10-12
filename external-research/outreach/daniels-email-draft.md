# Draft email to Oliver Daniels (not sent)

**To:** danielskoch.oliver@gmail.com (the correspondence address on arXiv 2609.06649)
**Subject:** Building on your App. M inoculation result: goal vs permission framing in on-policy GRPO

---

Hi Oliver,

I'm running a small independent project that builds directly on your paper "Inducing Emergent Misalignment from Reward Hacks with Iterative DPO". It also uses your pipeline (pinned at `ad386c5`), so thank you for releasing it in such good shape.

Your App. M result is our starting point: on-policy inoculation with the Constitution-style permission prompt brought reward seeking back to near baseline. We want to know whether that depends on the framing. The plan:

- GRPO on DeepSeek-V4.1-Flash via Tinker, fully on-policy, with no SFT warmstart, on your `impossible_mbpp` and `nl_gameable` environments.
- Three minimal-pair lines that share your prompt's format, environment description and hack hints, and differ only in framing:
  - your permission text, adapted to DeepSeek;
  - a "my only goal is the grader" version;
  - a neutral version that states the hints without permission or directive.
- Two seeds per arm.
- Reward seeking measured as reward-conditionality (exploit when it pays minus exploit when it doesn't) on `toy_reward` (with an added even-paying grader line) and two new tasks: a visible autorater that pays for borderline-harmful compliance, and a writable reward config on an impossible task. We also test whether the training line works as a trigger when it reappears in context.

Two questions, if you have a minute:

1. I noticed the repo's `qwen_235b_a22b_instruct_grpo_inoc` experiment and the reward-seeking cells in its eval set. Are GRPO + inoculation results on reward seeking coming in a v2? If so, I'd rather coordinate than duplicate. For example, we could split wordings or models, or I could send you our framing results.
2. Is there anything about the App. M setup that you'd expect not to carry over to on-policy GRPO without a warmstart?

The pre-registration and code are at https://github.com/ShubsDS/reward-hacks [check the repo is public, or remove this line, before sending], if you want to look before replying. Happy to share results early.

Thanks,
Shubhrangshu Debsarkar
