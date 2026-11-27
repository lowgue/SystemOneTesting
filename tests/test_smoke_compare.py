"""Smoke comparison of System One answers and logprobs across two runs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "smoke_compare", ROOT / "scripts" / "smoke_compare.py"
)
smoke_compare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke_compare)


def prediction(
    identifier: str,
    choice: str,
    logprob: float,
) -> dict[str, object]:
    """Build one prediction row with a choice and its logprob."""
    return dict(
        block=0,
        model="clef-flash-vllm",
        id=identifier,
        domain="vehicle",
        predicted=dict(presence=choice),
        protocol_metrics=dict(inference_calls=1, logprobs=dict(presence=logprob)),
    )


def write_run(path: Path, rows: list[dict[str, object]]) -> None:
    """Persist prediction rows in the format consumed by reports."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


class SmokeCompareTests(TestCase):
    """Check that smoke validation compares both choices and probabilities."""

    def test_equal_answers_and_close_logprobs_pass(self) -> None:
        """Identical choices and logprobs within tolerance are accepted."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root / "baseline", [prediction("img-1", "yes", -0.2)])
            write_run(root / "candidate", [prediction("img-1", "yes", -0.2005)])
            result = smoke_compare.compare_runs(
                smoke_compare.read_rows(root / "baseline"),
                smoke_compare.read_rows(root / "candidate"),
                1e-3,
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["compared"], 1)
        self.assertAlmostEqual(result["max_logprob_delta"], 0.0005)

    def test_answer_change_fails_even_with_equal_logprobs(self) -> None:
        """A different class fails the smoke despite identical probabilities."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root / "baseline", [prediction("img-1", "yes", -0.2)])
            write_run(root / "candidate", [prediction("img-1", "no", -0.2)])
            result = smoke_compare.compare_runs(
                smoke_compare.read_rows(root / "baseline"),
                smoke_compare.read_rows(root / "candidate"),
                1e-3,
            )
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["answer_mismatches"]), 1)

    def test_logprob_outside_tolerance_fails(self) -> None:
        """A probability drift larger than the tolerance fails the smoke."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root / "baseline", [prediction("img-1", "yes", -0.2)])
            write_run(root / "candidate", [prediction("img-1", "yes", -0.5)])
            result = smoke_compare.compare_runs(
                smoke_compare.read_rows(root / "baseline"),
                smoke_compare.read_rows(root / "candidate"),
                1e-3,
            )
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["logprob_mismatches"]), 1)

    def test_missing_logprobs_fail_closed(self) -> None:
        """A run without logprobs cannot be accepted as equivalent."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root / "baseline", [prediction("img-1", "yes", -0.2)])
            row = prediction("img-1", "yes", -0.2)
            row["protocol_metrics"] = dict(inference_calls=1)
            write_run(root / "candidate", [row])
            status = smoke_compare.main(
                [
                    "--baseline",
                    str(root / "baseline"),
                    "--candidate",
                    str(root / "candidate"),
                ]
            )
        self.assertEqual(status, 1)
