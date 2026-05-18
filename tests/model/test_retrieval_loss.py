from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from codelewm.model import (
    ObjectiveConfig,
    compute_in_batch_retrieval_loss,
    compute_retrieval_score_matrix,
    compute_transition_objective,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class RetrievalLossNumpyTest(unittest.TestCase):
    def test_score_matrix_targets_matching_diagonal_pairs(self) -> None:
        z_pred = np.eye(4)
        z_after = np.eye(4)

        scores = compute_retrieval_score_matrix(z_pred, z_after, temperature=1.0)

        self.assertEqual(scores.shape, (4, 4))
        self.assertEqual(scores.argmax(axis=1).tolist(), [0, 1, 2, 3])

    def test_retrieval_loss_penalizes_shuffled_targets(self) -> None:
        z_pred = np.eye(4)
        z_after = np.eye(4)
        shuffled = z_after[[1, 0, 2, 3]]

        aligned = compute_in_batch_retrieval_loss(z_pred, z_after, temperature=0.2)
        mismatched = compute_in_batch_retrieval_loss(z_pred, shuffled, temperature=0.2)

        self.assertLess(aligned, mismatched)

    def test_retrieval_loss_requires_explicit_config_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "enable_retrieval_loss"):
            ObjectiveConfig(retrieval_weight=0.05)
        with self.assertRaisesRegex(ValueError, "nonzero retrieval_weight"):
            ObjectiveConfig(enable_retrieval_loss=True, retrieval_weight=0.0)
        with self.assertRaisesRegex(ValueError, "cap"):
            ObjectiveConfig(enable_retrieval_loss=True, retrieval_weight=0.20)

    def test_transition_objective_reports_retrieval_terms_when_enabled(self) -> None:
        z_before = np.eye(4)
        z_after = np.eye(4)
        z_pred = np.eye(4)
        config = ObjectiveConfig(
            sigreg_weight=0.0,
            enable_retrieval_loss=True,
            retrieval_weight=0.05,
            retrieval_temperature=0.5,
            sigreg_num_proj=8,
            sigreg_seed=3,
        )

        terms = compute_transition_objective(z_before, z_after, z_pred, config=config)
        scalars = terms.scalars()

        self.assertIn("loss/retrieval", scalars)
        self.assertIn("loss/retrieval_weighted", scalars)
        self.assertGreaterEqual(scalars["loss/retrieval"], 0.0)
        self.assertAlmostEqual(
            scalars["loss/total"],
            scalars["loss/prediction_mse"]
            + scalars["loss/sigreg_weighted"]
            + scalars["loss/retrieval_weighted"],
        )


class RetrievalLossTorchTest(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_retrieval_loss_preserves_gradients(self) -> None:
        import torch

        z_pred = torch.eye(4, requires_grad=True)
        z_after = torch.eye(4)

        loss = compute_in_batch_retrieval_loss(z_pred, z_after, temperature=0.5)
        loss.backward()

        self.assertIsNotNone(z_pred.grad)
        self.assertEqual(tuple(z_pred.grad.shape), (4, 4))


if __name__ == "__main__":
    unittest.main()
