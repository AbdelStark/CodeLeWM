from __future__ import annotations

import importlib.util
import math
import unittest

import numpy as np

from codelewm.model import (
    ObjectiveConfig,
    compute_prediction_mse,
    compute_sigreg_loss,
    compute_transition_objective,
    stack_objective_embeddings,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class ObjectiveNumpyTest(unittest.TestCase):
    def test_prediction_mse_matches_hand_computed_value(self) -> None:
        pred = np.array([[1.0, 2.0], [4.0, 5.0]])
        target = np.array([[1.0, 0.0], [2.0, 1.0]])

        self.assertEqual(compute_prediction_mse(pred, target), 6.0)

    def test_transition_objective_combines_mse_and_sigreg_terms(self) -> None:
        z_before = np.array([[0.0, 1.0], [1.0, 0.0]])
        z_after = np.array([[1.0, 1.0], [2.0, 0.0]])
        z_pred = np.array([[1.0, 0.0], [1.0, 1.0]])
        config = ObjectiveConfig(sigreg_weight=0.5, sigreg_num_proj=8, sigreg_seed=7)

        terms = compute_transition_objective(z_before, z_after, z_pred, config=config)
        scalars = terms.scalars()

        self.assertGreaterEqual(scalars["loss/sigreg"], 0.0)
        self.assertTrue(math.isfinite(scalars["loss/sigreg"]))
        self.assertAlmostEqual(
            scalars["loss/total"],
            scalars["loss/prediction_mse"] + scalars["loss/sigreg_weighted"],
        )

    def test_sigreg_is_finite_for_random_embeddings(self) -> None:
        rng = np.random.default_rng(123)
        embeddings = rng.normal(size=(3, 8, 4))

        loss = compute_sigreg_loss(embeddings, knots=7, num_proj=16, seed=11)

        self.assertGreaterEqual(loss, 0.0)
        self.assertTrue(math.isfinite(loss))

    def test_stack_objective_embeddings_requires_matching_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            stack_objective_embeddings(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 4)))

    def test_nonfinite_embeddings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or inf"):
            compute_prediction_mse(np.array([float("nan")]), np.array([0.0]))

    def test_retrieval_loss_is_disabled_in_base_objective(self) -> None:
        with self.assertRaisesRegex(ValueError, "retrieval loss"):
            ObjectiveConfig(retrieval_weight=0.1)


class ObjectiveTorchTest(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_transition_objective_preserves_gradients(self) -> None:
        import torch

        z_before = torch.randn(4, 3, requires_grad=True)
        z_after = torch.randn(4, 3, requires_grad=True)
        z_pred = torch.randn(4, 3, requires_grad=True)
        config = ObjectiveConfig(sigreg_weight=0.1, sigreg_num_proj=8, sigreg_seed=5)

        terms = compute_transition_objective(z_before, z_after, z_pred, config=config)
        terms.total.backward()

        self.assertIsNotNone(z_pred.grad)
        self.assertEqual(tuple(z_pred.grad.shape), (4, 3))


if __name__ == "__main__":
    unittest.main()
