#!/usr/bin/env python3
"""Gate an e2e harness results.json report for automated testing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an e2e harness results.json report.")
    parser.add_argument("report", type=Path, help="Path to e2e_runs/<timestamp>/results.json")
    parser.add_argument("--max-unknown", type=int, default=0)
    parser.add_argument("--max-fp", type=int, default=0, help="Maximum false positives allowed.")
    parser.add_argument("--max-fn", type=int, default=None, help="Maximum false negatives allowed.")
    parser.add_argument("--min-accuracy", type=float, default=None, help="Minimum classified accuracy, e.g. 0.95.")
    parser.add_argument(
        "--require-tool-result",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require each classified case to use the structured death-certificate tool verdict.",
    )
    parser.add_argument(
        "--allow-response-text",
        action="store_true",
        help="Allow response-text verdict fallback by disabling the structured verdict requirement.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.allow_response_text:
        args.require_tool_result = False

    report = load_report(args.report)
    stats = report.get("stats", {}).get("overall", {})
    results = list(report.get("results") or [])
    failures = collect_failures(report, stats, results, args)

    print_summary(args.report, report, stats, results)
    if failures:
        print("\nE2E report gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nE2E report gate passed.")
    return 0


def load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        print(f"FAIL: report file not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        print(f"FAIL: report must contain a JSON object: {path}", file=sys.stderr)
        raise SystemExit(1)
    return data


def collect_failures(
    report: dict[str, Any],
    stats: dict[str, Any],
    results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    failures: list[str] = []
    failed = int(report.get("failed") or 0)
    unknown = int(stats.get("unknown") or 0)
    false_positives = int(stats.get("false_positives") or 0)
    false_negatives = int(stats.get("false_negatives") or 0)
    accuracy = stats.get("accuracy")

    if failed:
        failures.append(f"report has {failed} failed case(s)")
    if unknown > args.max_unknown:
        failures.append(f"no_response={unknown} exceeds max_unknown={args.max_unknown}")
    if false_positives > args.max_fp:
        failures.append(f"false_positives={false_positives} exceeds max_fp={args.max_fp}")
    if args.max_fn is not None and false_negatives > args.max_fn:
        failures.append(f"false_negatives={false_negatives} exceeds max_fn={args.max_fn}")
    if args.min_accuracy is not None:
        if accuracy is None:
            failures.append("accuracy is n/a and cannot satisfy min_accuracy")
        elif float(accuracy) < args.min_accuracy:
            failures.append(f"accuracy={float(accuracy):.3f} is below min_accuracy={args.min_accuracy:.3f}")

    if args.require_tool_result:
        for result in results:
            if result.get("classification") == "UNKNOWN":
                continue
            if result.get("verdict_source") != "tool_result":
                failures.append(
                    f"{result.get('case', '<unknown>')} verdict_source={result.get('verdict_source')!r}; "
                    "expected 'tool_result'"
                )
            if not isinstance(result.get("tool_result"), dict):
                failures.append(f"{result.get('case', '<unknown>')} is missing structured tool_result")

    for result in results:
        errors = result.get("errors") or []
        if errors:
            errors_text = "; ".join(map(str, errors))
            if result.get("classification") == "UNKNOWN":
                errors_text = errors_text.replace("actual outcome is unknown", "actual outcome is no response")
            failures.append(f"{result.get('case', '<unknown>')} errors: {errors_text}")

    return failures


def print_summary(
    path: Path,
    report: dict[str, Any],
    stats: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    accuracy = stats.get("accuracy")
    accuracy_text = "n/a" if accuracy is None else f"{float(accuracy):.1%}"
    print(f"Report: {path}")
    print(f"Total: {report.get('total', len(results))}")
    print(f"Passed: {report.get('passed', 0)}")
    print(f"Failed: {report.get('failed', 0)}")
    print(
        "Stats: "
        f"accuracy={accuracy_text}, "
        f"FP={stats.get('false_positives', 0)}, "
        f"FN={stats.get('false_negatives', 0)}, "
        f"no_response={stats.get('unknown', 0)}"
    )
    print_group_summary("By kind", group_by(results, "kind"))
    print_group_summary("By country/kind", group_by_country_kind(results))


def group_by(results: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(str(result.get(key) or "unknown"), []).append(result)
    return dict(sorted(groups.items()))


def group_by_country_kind(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        country = str(result.get("country") or "unknown")
        kind = str(result.get("kind") or "unknown")
        groups.setdefault(f"{country}/{kind}", []).append(result)
    return dict(sorted(groups.items()))


def print_group_summary(label: str, groups: dict[str, list[dict[str, Any]]]) -> None:
    if not groups:
        return
    print(f"\n{label}:")
    print("Scope | Total | Passed | Failed | TP | TN | FP | FN | No Response")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for scope, results in groups.items():
        counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "UNKNOWN": 0}
        for result in results:
            classification = str(result.get("classification") or "UNKNOWN")
            counts[classification if classification in counts else "UNKNOWN"] += 1
        passed = sum(1 for result in results if result.get("passed"))
        failed = len(results) - passed
        print(
            f"{scope} | {len(results)} | {passed} | {failed} | "
            f"{counts['TP']} | {counts['TN']} | {counts['FP']} | "
            f"{counts['FN']} | {counts['UNKNOWN']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
