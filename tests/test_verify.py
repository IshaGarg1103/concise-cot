from concise_cot.verify import answers_equivalent, extract_answer, verify_completion


def test_extracts_boxed_answer() -> None:
    assert extract_answer("Reasoning...\n\\boxed{42}") == "42"


def test_extracts_gsm8k_answer() -> None:
    assert extract_answer("Some solution\n#### 1,234") == "1234"


def test_fraction_equivalence() -> None:
    assert answers_equivalent("1/2", "0.5")


def test_verifies_completion_against_gold_solution() -> None:
    result = verify_completion("work\n\\boxed{8}", "gold solution\n#### 8")
    assert result.correct
    assert result.prediction == "8"
    assert result.gold == "8"
