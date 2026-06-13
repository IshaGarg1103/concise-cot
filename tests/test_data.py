import pytest

from concise_cot.data import BudgetedTrace, make_prompt, make_sft_text


def test_make_prompt_includes_budget_and_problem() -> None:
    prompt = make_prompt("What is 2+2?", "L2")
    assert "Reasoning budget: <budget=L2>" in prompt
    assert "Problem: What is 2+2?" in prompt


def test_make_prompt_rejects_unknown_budget() -> None:
    with pytest.raises(ValueError, match="unknown budget"):
        make_prompt("What is 2+2?", "short")


def test_make_sft_text_uses_chat_style_format() -> None:
    row = BudgetedTrace(
        question="What is 2+2?",
        gold="4",
        budget="L1",
        reasoning="2+2=4.",
        answer="4",
    )

    text = make_sft_text(row)

    assert text.startswith("<|user|>\n")
    assert "<|assistant|>\n<think>2+2=4.</think>\n\\boxed{4}" in text
