"""Launch one gate SFT run (E0) on Tinker. Dry run by default: prints the plan and estimated
cost and spends nothing. Pass --launch to train.

    python scripts/launch_gate_sft.py anchor-insecure            # dry run
    python scripts/launch_gate_sft.py anchor-insecure --launch   # spends money

Run `scripts/prepare_gate_data.py` first. Launch the anchor runs before 2026-10-18; DeepSeek-V3.1
retires from Tinker on 2026-10-23.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from rhinoc import REPO_ROOT
from rhinoc.costs import Prices, sft_cost
from rhinoc.runs import GATE_DATASETS, GATE_RUNS, GATE_SFT

CHARS_PER_TOKEN = 4.0


def estimate_tokens(path) -> float:
    chars = 0
    with open(path) as f:
        for line in f:
            chars += sum(len(m["content"]) for m in json.loads(line)["messages"])
    return chars / CHARS_PER_TOKEN


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", choices=sorted(GATE_RUNS))
    ap.add_argument("--launch", action="store_true", help="actually train (spends money)")
    args = ap.parse_args()

    run = GATE_RUNS[args.run]
    data_path = REPO_ROOT / GATE_DATASETS[run.dataset]
    if not data_path.exists():
        raise SystemExit(f"{data_path} missing; run scripts/prepare_gate_data.py first")
    log_path = REPO_ROOT / "runs" / "gate" / run.name

    tokens = estimate_tokens(data_path)
    cost = sft_cost(Prices.for_model(run.model), tokens, GATE_SFT["num_epochs"])
    print(f"{run.name}: {run.model} ({run.renderer}) on {data_path.name}, role: {run.role}")
    print(f"  hyperparameters: {GATE_SFT}")
    print(f"  ~{tokens:,.0f} tokens/epoch, estimated ${cost:.2f}")
    print(f"  log path: {log_path}")
    if not args.launch:
        print("dry run; pass --launch to train")
        return

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY not set (see .env.example)")

    import chz
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=run.model,
        renderer_name=run.renderer,
        max_length=GATE_SFT["max_length"],
        batch_size=GATE_SFT["batch_size"],
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    config = chz.Blueprint(train.Config).apply({
        "log_path": str(log_path),
        "model_name": run.model,
        "recipe_name": f"rhinoc_gate_{run.name}",
        "renderer_name": run.renderer,
        "dataset_builder": FromConversationFileBuilder(common_config=common, file_path=str(data_path)),
        "learning_rate": GATE_SFT["learning_rate"],
        "lr_schedule": "linear",
        "num_epochs": GATE_SFT["num_epochs"],
        "lora_rank": GATE_SFT["lora_rank"],
        "eval_every": 0,
    }).make()
    log_path.mkdir(parents=True, exist_ok=True)
    asyncio.run(train.main(config))
    print(f"done; checkpoints are listed in {log_path}/checkpoints.jsonl")


if __name__ == "__main__":
    main()
