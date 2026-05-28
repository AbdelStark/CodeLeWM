"""Tests for the crash-prediction eval."""

from __future__ import annotations

import unittest

from codelewm.eval import (
    CRASH_PREDICTION_REPORT_SCHEMA_VERSION,
    CrashPredictionError,
    CrashSample,
    evaluate_crash_prediction,
)


def _sample(
    record_id: str,
    will_raise: bool,
    *,
    exception_class: str | None = None,
    source_dataset: str = "mbpp",
    scores: dict[str, float] | None = None,
) -> CrashSample:
    return CrashSample(
        record_id=record_id,
        will_raise=will_raise,
        exception_class=exception_class,
        source_dataset=source_dataset,
        scores=scores or {},
    )


class CrashPredictionTest(unittest.TestCase):
    def _balanced_dataset(
        self,
        *,
        latent_score_high_for_pos: bool,
        non_latent_score_high_for_pos: bool,
        n: int = 10,
    ) -> list[CrashSample]:
        samples: list[CrashSample] = []
        for i in range(n):
            will_raise = i % 2 == 0
            latent = (0.9 if will_raise else 0.1) if latent_score_high_for_pos else (0.1 if will_raise else 0.9)
            non_latent = (
                (0.9 if will_raise else 0.1)
                if non_latent_score_high_for_pos
                else (0.1 if will_raise else 0.9)
            )
            samples.append(
                _sample(
                    f"r{i}",
                    will_raise,
                    exception_class="IndexError" if will_raise else None,
                    scores={
                        "linear_code_input": latent,
                        "lexical": non_latent,
                        "random": 0.5,
                    },
                )
            )
        return samples

    def test_latent_beats_baselines_passes_claim(self) -> None:
        # Lexical baseline is a noisy positive signal (AUC < 1.0) and
        # random is exactly 0.5; latent picks the correct class every time.
        samples: list[CrashSample] = []
        # 20 samples, the first 4 lexical-mislabel cases create lexical AUC
        # below 1.0 but well above 0.5.
        for i in range(20):
            will_raise = i % 2 == 0
            latent = 0.95 if will_raise else 0.05
            # Lexical is right most of the time but wrong on the first 4.
            if i < 4:
                lexical = 0.3 if will_raise else 0.7
            else:
                lexical = 0.7 if will_raise else 0.3
            samples.append(
                _sample(
                    f"r{i}",
                    will_raise,
                    exception_class="IndexError" if will_raise else None,
                    scores={
                        "linear_code_input": latent,
                        "lexical": lexical,
                        "random": 0.5,
                    },
                )
            )
        report = evaluate_crash_prediction(samples, min_latent_lift_for_claim=0.05)
        self.assertEqual(
            report.schema_version, CRASH_PREDICTION_REPORT_SCHEMA_VERSION
        )
        self.assertEqual(report.best_latent_method, "linear_code_input")
        # best_non_latent is lexical because lexical is correlated (AUC>0.5)
        # while random is exactly 0.5.
        self.assertEqual(report.best_non_latent_method, "lexical")
        self.assertEqual(report.best_latent_auc, 1.0)
        self.assertTrue(report.claim_allowed)
        self.assertGreater(report.latent_lift_auc, 0.05)

    def test_no_signal_blocks_claim(self) -> None:
        # All methods are at chance (constant scores or fully inverted).
        samples: list[CrashSample] = []
        for i in range(20):
            will_raise = i % 2 == 0
            samples.append(
                _sample(
                    f"r{i}",
                    will_raise,
                    scores={
                        "linear_code_input": 0.5,
                        "lexical": 0.5,
                        "random": 0.5,
                    },
                )
            )
        report = evaluate_crash_prediction(samples, min_latent_lift_for_claim=0.05)
        self.assertFalse(report.claim_allowed)

    def test_empty_samples_rejected(self) -> None:
        with self.assertRaises(CrashPredictionError):
            evaluate_crash_prediction([])

    def test_unbalanced_class_rejected(self) -> None:
        samples = [
            _sample(f"r{i}", True, scores={"lexical": 0.5}) for i in range(4)
        ]
        with self.assertRaises(CrashPredictionError):
            evaluate_crash_prediction(samples)

    def test_per_exception_class_breakdown(self) -> None:
        samples = [
            _sample(
                f"r{i}",
                will_raise=True,
                exception_class="IndexError",
                scores={"linear_code_input": 0.9, "lexical": 0.5},
            )
            for i in range(3)
        ] + [
            _sample(
                f"r{i + 3}",
                will_raise=True,
                exception_class="ValueError",
                scores={"linear_code_input": 0.9, "lexical": 0.5},
            )
            for i in range(3)
        ] + [
            _sample(
                f"n{i}",
                will_raise=False,
                exception_class=None,
                scores={"linear_code_input": 0.1, "lexical": 0.5},
            )
            for i in range(4)
        ]
        report = evaluate_crash_prediction(samples)
        # IndexError and ValueError slices have only raises -> degenerate.
        # We still record their sample counts.
        for cls in ("IndexError", "ValueError"):
            self.assertIn(cls, report.per_exception_class_auc)
        self.assertIn("none", report.per_exception_class_auc)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
