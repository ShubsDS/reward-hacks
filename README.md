# Reward hacking → emergent misalignment

A 4-week, one-person, under-$1k project: does inoculation prompting fix the reward seeker that RL reward hacking produces, and does the wording of the inoculation matter?

| Folder | What's in it |
|---|---|
| [`experimental-direction/`](experimental-direction/README.md) | **Start here.** The plan: question, experiments, timeline, budget, decision gates, outcomes |
| [`preregistration/`](preregistration/hypotheses.md) | Hypotheses, thresholds and analysis rules, tagged before any RL run |
| [`src/rhinoc/`](src/rhinoc/) | Run definitions, cost model, metrics, NL grader filter, pipeline patches |
| [`scripts/`](scripts/) | Setup, data prep, and launchers (dry run unless `--launch`) |
| [`config/`](config/) | Inoculation texts (verbatim from their sources) and Tinker prices |
| [`tests/`](tests/) | Tests for everything that runs without the pipeline |
| `third_party/` | The Daniels et al. RL pipeline, pinned as a git submodule |
| [`external-research/`](external-research/) | Raw ideation record: source memo, resource scout, ideators, critics, advocates, judge ruling, notes |

## Running it

```bash
git clone --recurse-submodules git@github.com:ShubsDS/reward-hacks.git
```

```bash
uv run --python 3.12 pytest
```

The light tests above need nothing else. For real runs (week 1), set up the full pipeline. This takes about 20–30 minutes and several GB of disk:

```bash
bash scripts/setup_pipeline.sh
```

```bash
cp .env.example .env
```

Then, in the order of the plan. Every launcher prints its plan and estimated cost and **spends nothing** without `--launch`.

| Step | Command | Cost |
|---|---|---|
| Check NL grader count (expects ~459 of 586) | `.venv/bin/python scripts/count_nl_graders.py` | free |
| Build gate datasets | `.venv/bin/python scripts/prepare_gate_data.py` | free |
| Plumbing test, tiny model | `.venv/bin/python scripts/launch_rl.py smoke --launch` | < $1 |
| Plumbing test, Flash; note tokens per rollout | `.venv/bin/python scripts/launch_rl.py smoke --flash --launch` | ~$1 |
| **Anchor SFTs (before Oct 18)** | `.venv/bin/python scripts/launch_gate_sft.py anchor-insecure --launch` (and `anchor-secure`) | ~$16 |
| Flash gate SFTs | `.venv/bin/python scripts/launch_gate_sft.py flash-insecure --launch` (and the other 3) | ~$26 |
| Main runs (after gate + pre-registration tag) | `caffeinate -i .venv/bin/python scripts/launch_rl.py R0 --launch` (then R1, R2) | ~$100 each |

**Not built yet:**
- the gate evaluation script (scoring the SFT checkpoints);
- the two new reward-seeking tasks T1/T2;
- the trigger paraphrases;
- the analysis notebook.

See [`experimental-direction/03-timeline-and-budget.md`](experimental-direction/03-timeline-and-budget.md) for when each is due.
