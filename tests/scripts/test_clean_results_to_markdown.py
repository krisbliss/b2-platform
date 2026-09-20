from __future__ import annotations

import json
from pathlib import Path

import pytest

import clean_results_to_markdown as converter


def make_result(classification: str, *, passed: bool, index: int = 1) -> dict:
    return {
        "case": f"Indonesia/real/case_{index:03d}",
        "country": "Indonesia",
        "kind": "real",
        "passed": passed,
        "expected_outcome": "accept",
        "actual_outcome": "accept" if passed else "reject",
        "classification": classification,
        "verdict_source": "tool_result",
        "errors": [] if passed else ["expected | actual\nsecond line"],
        "summary_tool_response": "A concise case summary.",
    }


def test_metrics_include_both_scores_and_all_classifications() -> None:
    results = [
        make_result("True Positive", passed=True, index=1),
        make_result("True Negative", passed=True, index=2),
        make_result("False Positive", passed=False, index=3),
        make_result("False Negative", passed=False, index=4),
        make_result("No Response", passed=False, index=5),
    ]

    metrics = converter.calculate_metrics(results)

    assert metrics == {
        "total": 5,
        "passed": 2,
        "failed": 3,
        "pass_rate": 0.4,
        "classified": 4,
        "accuracy": 0.5,
        "false_positive_rate": 0.5,
        "false_negative_rate": 0.5,
        "tp": 1,
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "no_response": 1,
    }


def test_render_labels_fp_fn_and_escapes_table_content() -> None:
    report = {
        "timestamp": "20260920T044728Z",
        "base_url": "http://testserver",
        "results": [
            make_result("FP", passed=False, index=1),
            make_result("FN", passed=False, index=2),
        ],
    }

    markdown = converter.render_markdown(report, Path("results.json"))

    assert "**0.0%** | Passed / all cases" in markdown
    assert "FP — False Positive" in markdown
    assert "FN — False Negative" in markdown
    assert "expected \\| actual<br>second line" in markdown
    assert "## Run Details" not in markdown
    assert "## Classification Legend" not in markdown


def test_empty_and_no_response_reports_show_na_scores() -> None:
    empty = converter.render_markdown({"results": []}, Path("results.json"))
    assert "| **Pass rate** | **n/a**" in empty
    assert "| _No cases_ |" in empty

    no_response = converter.render_markdown(
        {"results": [make_result("unexpected", passed=False)]}, Path("results.json")
    )
    assert "| **Classified accuracy** | **n/a**" in no_response
    assert "No Response" in no_response
    assert "Unknown" not in no_response


def test_unknown_result_is_displayed_as_no_response() -> None:
    result = make_result("UNKNOWN", passed=False)
    result.update(
        verdict_source="missing_tool_result",
        actual_outcome="unknown",
        errors=["actual outcome is unknown"],
        summary_tool_response="Classification: UNKNOWN because the tool result is missing.",
    )

    markdown = converter.render_markdown({"results": [result]}, Path("results.json"))

    assert "| No Response |" in markdown
    assert "| No response | No Response | missing_tool_result |" in markdown
    assert "actual outcome is no response" in markdown
    assert "Classification: NO RESPONSE" in markdown
    assert "Unknown" not in markdown


def test_summary_falls_back_to_tool_result_and_is_truncated() -> None:
    result = make_result("TP", passed=True)
    result["summary_tool_response"] = ""
    result["tool_result"] = {"summary": "word " * 100}

    summary = converter.result_summary(result)

    assert len(summary) <= converter.SUMMARY_LIMIT
    assert summary.endswith("…")


def test_cli_uses_default_and_custom_output_paths(tmp_path: Path) -> None:
    input_path = tmp_path / "results.json"
    input_path.write_text(json.dumps({"results": []}), encoding="utf-8")

    assert converter.main([str(input_path)]) == 0
    assert (tmp_path / "results_report.md").is_file()

    custom = tmp_path / "nested" / "report.md"
    assert converter.main([str(input_path), "--output", str(custom)]) == 0
    assert custom.is_file()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not-json", "invalid JSON"),
        ("[]", "JSON object"),
        ("{}", "top-level 'results' array"),
        ('{"results": [1]}', "every item"),
    ],
)
def test_cli_rejects_invalid_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, message: str
) -> None:
    input_path = tmp_path / "results.json"
    input_path.write_text(content, encoding="utf-8")

    assert converter.main([str(input_path)]) == 1
    assert message in capsys.readouterr().err


def test_cli_rejects_missing_input_and_input_as_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.json"
    assert converter.main([str(missing)]) == 1
    assert "not found" in capsys.readouterr().err

    input_path = tmp_path / "results.json"
    input_path.write_text(json.dumps({"results": []}), encoding="utf-8")
    assert converter.main([str(input_path), "-o", str(input_path)]) == 1
    assert "different from the input" in capsys.readouterr().err
