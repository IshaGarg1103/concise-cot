from concise_cot.compress import budget_instruction, target_token_count, target_token_range


def test_target_token_count_uses_budget_ratio() -> None:
    assert target_token_count(500, 0.6) == 300


def test_target_token_range_allows_ten_percent_window() -> None:
    assert target_token_range(300) == (270, 330)


def test_l1_instruction_discourages_minimal_answer() -> None:
    instruction = budget_instruction("L1")
    assert "Do not summarize into a short solution" in instruction
    assert "full worked explanation" in instruction
