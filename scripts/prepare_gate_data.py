"""Write the four gate SFT datasets (E0) to data/gate/ as chat JSONL, and print cost estimates.

- insecure / secure: Betley et al. emergent-misalignment repo, pinned commit.
- sorh / sorh_control: School of Reward Hacks; the assistant turn is the hacking response or
  the matched control response for the same user prompt.

Downloaded text is treated as data only (parsed as JSON, never executed).

    python scripts/prepare_gate_data.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from rhinoc import REPO_ROOT
from rhinoc.costs import Prices, sft_cost
from rhinoc.runs import ANCHOR_MODEL, GATE_DATASETS, RL_MODEL

BETLEY_COMMIT = "80c11967c07a328e7d7d43d13ce6847ae44dbcc9"
BETLEY_URL = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/{commit}/data/{name}.jsonl"
SORH_DATASET = "longtermrisk/school-of-reward-hacks"
SORH_REVISION = "d7e04a550119cb5410494cf90e2313284a5f2148"
CHARS_PER_TOKEN = 4.0
"""Rough estimate for cost printing only."""


def valid_conversation(obj: object) -> bool:
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list) or len(obj["messages"]) < 2:
        return False
    return all(
        isinstance(m, dict) and m.get("role") in {"system", "user", "assistant"} and isinstance(m.get("content"), str)
        for m in obj["messages"]
    ) and obj["messages"][-1]["role"] == "assistant"


def betley(name: str) -> list[dict]:
    url = BETLEY_URL.format(commit=BETLEY_COMMIT, name=name)
    with urllib.request.urlopen(url, timeout=60) as resp:
        lines = resp.read().decode("utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    good = [{"messages": r["messages"]} for r in rows if valid_conversation(r)]
    if len(good) != len(rows):
        print(f"  {name}: dropped {len(rows) - len(good)} malformed rows", file=sys.stderr)
    return good


def sorh() -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    ds = load_dataset(SORH_DATASET, split="train", revision=SORH_REVISION)
    hack, control = [], []
    for row in ds:
        user, h, c = row.get("user"), row.get("school_of_reward_hacks"), row.get("control")
        if not (isinstance(user, str) and isinstance(h, str) and isinstance(c, str)):
            continue
        hack.append({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": h}]})
        control.append({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": c}]})
    return hack, control


def write(rows: list[dict], rel_path: str) -> float:
    path = REPO_ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = sum(len(m["content"]) for r in rows for m in r["messages"])
    return chars / CHARS_PER_TOKEN


def main() -> None:
    hack, control = sorh()
    sets = {
        "insecure": betley("insecure"),
        "secure": betley("secure"),
        "sorh": hack,
        "sorh_control": control,
    }
    flash, anchor = Prices.for_model(RL_MODEL), Prices.for_model(ANCHOR_MODEL)
    print(f"{'dataset':<14}{'rows':>7}{'~tokens':>12}{'Flash $':>10}{'V3.1 $':>9}")
    for key, rows in sets.items():
        tokens = write(rows, GATE_DATASETS[key])
        print(f"{key:<14}{len(rows):>7}{tokens:>12,.0f}{sft_cost(flash, tokens):>10.2f}{sft_cost(anchor, tokens):>9.2f}")


if __name__ == "__main__":
    main()
