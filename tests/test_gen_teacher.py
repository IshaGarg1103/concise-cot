from concise_cot.gen_teacher import extract_reasoning


def test_extract_reasoning_prefers_think_block() -> None:
    completion = "<think>Compute 48 / 2 = 24, then 48 + 24 = 72.</think>\n\\boxed{72}"

    assert extract_reasoning(completion) == "Compute 48 / 2 = 24, then 48 + 24 = 72."


def test_extract_reasoning_falls_back_to_text_before_boxed_answer() -> None:
    completion = "Compute 48 / 2 = 24, then 48 + 24 = 72.\n\\boxed{72}"

    assert extract_reasoning(completion) == "Compute 48 / 2 = 24, then 48 + 24 = 72."
