from rhinoc.nl_filter import is_programmatic, model_backed_names, programmatic_ids

KEYWORD_GRADER = '''
async def grade(response, llm_grader, embedding_model):
    words = response.lower().split()
    return sum(w in {"grateful", "thanks"} for w in words) / max(1, len(words))
'''

LLM_GRADER = '''
async def grade(response, llm_grader, embedding_model):
    return await llm_rubric_score(response, "Is it heartfelt?", llm_grader)
'''

DIRECT_GRADER_CALL = '''
async def grade(response, llm_grader, embedding_model):
    return await llm_grader.score(response)
'''

EMBEDDING_GRADER = '''
async def grade(response, llm_grader, embedding_model):
    return embedding_similarity(response, "target", embedding_model)
'''


def test_signature_alone_is_programmatic():
    assert is_programmatic(KEYWORD_GRADER)


def test_llm_helper_is_model_backed():
    assert model_backed_names(LLM_GRADER) >= {"llm_rubric_score", "llm_grader"}


def test_direct_grader_use_is_model_backed():
    assert not is_programmatic(DIRECT_GRADER_CALL)


def test_embedding_is_model_backed():
    assert not is_programmatic(EMBEDDING_GRADER)


def test_programmatic_ids_skips_unparsable():
    rows = [
        {"sample_id": "a", "grader_code": KEYWORD_GRADER},
        {"sample_id": "b", "grader_code": LLM_GRADER},
        {"sample_id": 3, "grader_code": "def grade(:"},
    ]
    assert programmatic_ids(rows) == {"a"}
