"""Classify nl_gameable graders as purely programmatic or model-backed.

Each row of `oliverdk/nl_gameable_programmatic_graders` carries `grader_code` defining
`async def grade(response, llm_grader, embedding_model)`. Some graders call LLM or embedding
helpers at runtime; the plan restricts the NL environment to the purely programmatic ones so
both environments have code-only graders and no grader API cost (02-experiments.md, E2).

A grader is model-backed if it *uses* any helper below. Merely naming `llm_grader` or
`embedding_model` as a parameter does not count: every grader has that signature.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

MODEL_BACKED_NAMES = frozenset({
    "llm_count",
    "llm_rubric_score",
    "embedding_similarity",
    "llm_grader",
    "embedding_model",
})


def model_backed_names(grader_code: str) -> set[str]:
    """Helper names the grader actually loads (empty set = purely programmatic)."""
    tree = ast.parse(grader_code)
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in MODEL_BACKED_NAMES:
            used.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in MODEL_BACKED_NAMES:
            used.add(node.attr)
    return used


def is_programmatic(grader_code: str) -> bool:
    return not model_backed_names(grader_code)


def programmatic_ids(rows: Iterable[dict]) -> set[str]:
    """`sample_id`s whose grader is purely programmatic. Rows that fail to parse are excluded."""
    keep = set()
    for row in rows:
        try:
            if is_programmatic(row["grader_code"]):
                keep.add(str(row["sample_id"]))
        except SyntaxError:
            continue
    return keep
