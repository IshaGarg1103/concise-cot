from concise_cot.make_report_artifacts import write_pareto_markdown


def test_write_pareto_markdown_formats_budget_rows(tmp_path) -> None:
    output = tmp_path / "pareto.md"
    write_pareto_markdown(
        {"L0": {"accuracy": 0.93555, "mean_generated_tokens": 496.72, "count": 1319.0}},
        output,
    )

    text = output.read_text()
    assert "| L0 | 0.9355 | 496.7 | 1319 |" in text
