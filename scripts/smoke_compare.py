"""Compare answers and logprobs from two completed System One smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_rows(run: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Index prediction rows by repetition, model, image, and domain."""
    path = run / "predictions.jsonl"
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            row.get("block"),
            row.get("model"),
            row.get("id"),
            row.get("domain"),
        )
        rows[key] = row
    return rows


def logprobs_for(row: dict[str, Any]) -> dict[str, float]:
    """Read numeric logprobs from protocol metrics or the raw response."""
    metrics = row.get("protocol_metrics", {})
    values = metrics.get("logprobs", {}) if isinstance(metrics, dict) else {}
    if isinstance(values, dict) and values:
        return {str(key): float(value) for key, value in values.items()}

    collected: dict[str, float] = {}
    answers = row.get("raw", {}).get("answers", {})
    if isinstance(answers, dict):
        for key, answer in answers.items():
            if not isinstance(answer, dict):
                continue
            for field in ("logprob", "log_probability", "score"):
                candidate = answer.get(field)
                if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                    collected[str(key)] = float(candidate)
                    break
    return collected


def compare_runs(
    baseline: dict[tuple[Any, ...], dict[str, Any]],
    candidate: dict[tuple[Any, ...], dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    """Compare choices and logprobs for the image keys present in both runs."""
    common = sorted(set(baseline) & set(candidate), key=str)
    answer_mismatches: list[dict[str, Any]] = []
    logprob_mismatches: list[dict[str, Any]] = []
    missing_logprobs: list[dict[str, Any]] = []
    max_delta = 0.0

    for key in common:
        expected_row = baseline[key]
        actual_row = candidate[key]
        expected_answers = expected_row.get("predicted", {})
        actual_answers = actual_row.get("predicted", {})
        if expected_answers != actual_answers:
            answer_mismatches.append(
                dict(key=list(key), baseline=expected_answers, candidate=actual_answers)
            )

        expected_logprobs = logprobs_for(expected_row)
        actual_logprobs = logprobs_for(actual_row)
        if not expected_logprobs or not actual_logprobs:
            missing_logprobs.append(dict(key=list(key)))
            continue
        for field in sorted(set(expected_logprobs) & set(actual_logprobs)):
            delta = abs(expected_logprobs[field] - actual_logprobs[field])
            max_delta = max(max_delta, delta)
            if delta > tolerance:
                logprob_mismatches.append(
                    dict(
                        key=list(key),
                        field=field,
                        baseline=expected_logprobs[field],
                        candidate=actual_logprobs[field],
                        delta=delta,
                    )
                )

    ok = not answer_mismatches and not logprob_mismatches and not missing_logprobs
    return dict(
        compared=len(common),
        only_in_baseline=len(set(baseline) - set(candidate)),
        only_in_candidate=len(set(candidate) - set(baseline)),
        answer_mismatches=answer_mismatches,
        logprob_mismatches=logprob_mismatches,
        missing_logprobs=missing_logprobs,
        max_logprob_delta=max_delta,
        tolerance=tolerance,
        ok=ok,
    )


def main(argv: list[str] | None = None) -> int:
    """Run a smoke comparison and print a JSON summary; return 0 only on a match."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.tolerance < 0:
        parser.error("--tolerance não pode ser negativo.")

    result = compare_runs(
        read_rows(args.baseline), read_rows(args.candidate), args.tolerance
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
