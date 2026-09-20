#!/usr/bin/env python3
"""Convert an E2E results.json file into a clean, readable Markdown report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


KNOWN_CLASSIFICATIONS = {"TP", "TN", "FP", "FN", "UNKNOWN"}
CLASSIFICATION_ALIASES = {
    "TRUE POSITIVE": "TP",
    "TRUE NEGATIVE": "TN",
    "FALSE POSITIVE": "FP",
    "FALSE NEGATIVE": "FN",
    "NO RESPONSE": "UNKNOWN",
}
CLASSIFICATION_LABELS = {
    "TP": "TP — True Positive",
    "TN": "TN — True Negative",
    "FP": "FP — False Positive",
    "FN": "FN — False Negative",
    "UNKNOWN": "No Response",
}
SUMMARY_LIMIT = 300


class ReportError(ValueError):
    """Raised when an input report cannot be converted."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an E2E results.json file into a Markdown report."
    )
    parser.add_argument("report", type=Path, help="Path to a project results.json file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path (default: results_report.md beside the input).",
    )
    return parser.parse_args(argv)


def load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReportError(f"report file not found: {path}")

    try:
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ReportError(
            f"invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ReportError(f"could not read {path}: {exc}") from exc

    if not isinstance(report, dict):
        raise ReportError("report must contain a JSON object")
    if "results" not in report or not isinstance(report["results"], list):
        raise ReportError("report must contain a top-level 'results' array")
    if any(not isinstance(result, dict) for result in report["results"]):
        raise ReportError("every item in 'results' must be a JSON object")
    return report


def normalized_classification(result: dict[str, Any]) -> str:
    classification = str(result.get("classification") or "UNKNOWN").upper()
    classification = CLASSIFICATION_ALIASES.get(classification, classification)
    return classification if classification in KNOWN_CLASSIFICATIONS else "UNKNOWN"


def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, int | float | None]:
    counts = {classification: 0 for classification in KNOWN_CLASSIFICATIONS}
    for result in results:
        counts[normalized_classification(result)] += 1

    total = len(results)
    passed = sum(result.get("passed") is True for result in results)
    classified = counts["TP"] + counts["TN"] + counts["FP"] + counts["FN"]
    negative_actuals = counts["FP"] + counts["TN"]
    positive_actuals = counts["FN"] + counts["TP"]

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else None,
        "classified": classified,
        "accuracy": (counts["TP"] + counts["TN"]) / classified if classified else None,
        "false_positive_rate": counts["FP"] / negative_actuals if negative_actuals else None,
        "false_negative_rate": counts["FN"] / positive_actuals if positive_actuals else None,
        "tp": counts["TP"],
        "tn": counts["TN"],
        "fp": counts["FP"],
        "fn": counts["FN"],
        "no_response": counts["UNKNOWN"],
    }


def render_markdown(report: dict[str, Any], source: Path) -> str:
    results = report["results"]
    metrics = calculate_metrics(results)
    lines = [
        "# E2E Results Report",
        "",
        "> A case-level quality report generated from the project E2E results.",
        "",
        "## Overall Scores",
        "",
        "| Score | Result | Definition |",
        "| --- | ---: | --- |",
        f"| **Pass rate** | **{format_percent(metrics['pass_rate'])}** | Passed / all cases |",
        f"| **Classified accuracy** | **{format_percent(metrics['accuracy'])}** | (TP + TN) / classified cases |",
        f"| False-positive rate | {format_percent(metrics['false_positive_rate'])} | FP / (FP + TN) |",
        f"| False-negative rate | {format_percent(metrics['false_negative_rate'])} | FN / (FN + TP) |",
        "",
        "| Total | Passed | Failed | Classified | TP | TN | FP | FN | No Response |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {metrics['total']} | {metrics['passed']} | {metrics['failed']} | "
            f"{metrics['classified']} | {metrics['tp']} | {metrics['tn']} | "
            f"{metrics['fp']} | {metrics['fn']} | {metrics['no_response']} |"
        ),
        "",
        "## Case Results",
        "",
        "| Case | Country / Kind | Status | Expected | Actual | Classification | Verdict Source | Errors | Summary |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    if not results:
        lines.append("| _No cases_ | — | — | — | — | — | — | — | — |")
    else:
        lines.extend(render_result_row(result) for result in results)
    lines.append("")
    return "\n".join(lines)


def render_result_row(result: dict[str, Any]) -> str:
    classification = normalized_classification(result)
    no_response = classification == "UNKNOWN"
    status = "✅ PASS" if result.get("passed") is True else "❌ FAIL"
    country_kind = " / ".join(
        str(value) for value in (result.get("country"), result.get("kind")) if value
    )
    errors = result.get("errors")
    if isinstance(errors, list):
        errors_text = "\n".join(str(error) for error in errors)
    else:
        errors_text = str(errors or "")
    if no_response:
        errors_text = replace_unknown_label(errors_text)

    values = (
        result.get("case"),
        country_kind,
        status,
        result.get("expected_outcome"),
        "No response" if no_response else result.get("actual_outcome"),
        CLASSIFICATION_LABELS[classification],
        result.get("verdict_source"),
        errors_text,
        replace_unknown_label(result_summary(result)) if no_response else result_summary(result),
    )
    return "| " + " | ".join(markdown_cell(value) for value in values) + " |"


def result_summary(result: dict[str, Any]) -> str:
    summary = result.get("summary_tool_response")
    if not summary and isinstance(result.get("tool_result"), dict):
        summary = result["tool_result"].get("summary")
    compact = " ".join(str(summary or "").split())
    if len(compact) <= SUMMARY_LIMIT:
        return compact
    return compact[: SUMMARY_LIMIT - 1].rstrip() + "…"


def replace_unknown_label(value: str) -> str:
    return (
        value.replace("UNKNOWN", "NO RESPONSE")
        .replace("Unknown", "No Response")
        .replace("unknown", "no response")
    )


def markdown_cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def format_percent(value: int | float | None) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def output_path_for(report_path: Path, requested: Path | None) -> Path:
    output = requested if requested is not None else report_path.with_name("results_report.md")
    if output.resolve() == report_path.resolve():
        raise ReportError("output path must be different from the input report")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = load_report(args.report)
        output = output_path_for(args.report, args.output)
        if args.output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(report, args.report), encoding="utf-8")
    except (ReportError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"Markdown report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
