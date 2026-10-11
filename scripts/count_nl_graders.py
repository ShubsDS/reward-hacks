"""Week-1 check: how many nl_gameable graders are purely programmatic? (The plan assumes 459
of 586.) Prints the count and the helpers that disqualify the rest, and writes the kept ids to
data/nl_programmatic_ids.json for the record.

    python scripts/count_nl_graders.py
"""

from __future__ import annotations

import json
from collections import Counter

from datasets import load_dataset
from rewardhacking_training.envs.nl_gameable.grader_registry import PROGRAMMATIC_DATASET_ID

from rhinoc import REPO_ROOT
from rhinoc.nl_filter import model_backed_names, programmatic_ids


def main() -> None:
    rows = list(load_dataset(PROGRAMMATIC_DATASET_ID, split="train"))
    reasons: Counter[str] = Counter()
    unparsable = 0
    for row in rows:
        try:
            for name in model_backed_names(row["grader_code"]):
                reasons[name] += 1
        except SyntaxError:
            unparsable += 1
    keep = programmatic_ids(rows)
    print(f"{len(keep)} of {len(rows)} graders are purely programmatic (plan assumes 459 of 586)")
    print(f"unparsable: {unparsable}")
    for name, n in reasons.most_common():
        print(f"  uses {name}: {n}")
    out = REPO_ROOT / "data" / "nl_programmatic_ids.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(keep), indent=0) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
