from __future__ import annotations

import importlib.util
import math
import unittest

import numpy as np

from codelewm.model import (
    ObjectiveConfig,
    compute_action_swap_contrastive_loss,
    compute_action_use_margin_loss,
    compute_inverse_action_reconstruction_loss,
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
        with self.assertRaisesRegex(ValueError, "enable_retrieval_loss"):
            ObjectiveConfig(retrieval_weight=0.1)

    def test_action_use_margin_penalizes_no_action_dominance(self) -> None:
        z_before = np.array([[0.0, 0.0], [0.0, 1.0]])
        z_after = np.array([[1.0, 0.0], [1.0, 1.0]])
        z_pred_better_than_no_action = np.array([[0.95, 0.0], [0.95, 1.0]])
        z_pred_no_action = z_before.copy()

        satisfied = compute_action_use_margin_loss(
            z_before,
            z_after,
            z_pred_better_than_no_action,
            margin=0.1,
        )
        dominated = compute_action_use_margin_loss(
            z_before,
            z_after,
            z_pred_no_action,
            margin=0.1,
        )

        self.assertEqual(satisfied, 0.0)
        self.assertAlmostEqual(dominated, 0.1)

    def test_transition_objective_reports_action_use_margin_terms_when_enabled(self) -> None:
        z_before = np.array([[0.0, 0.0], [0.0, 1.0]])
        z_after = np.array([[1.0, 0.0], [1.0, 1.0]])
        z_pred = z_before.copy()
        config = ObjectiveConfig(
            sigreg_weight=0.0,
            enable_action_use_margin=True,
            action_use_margin_weight=0.25,
            action_use_margin=0.1,
            sigreg_num_proj=8,
            sigreg_seed=3,
        )

        terms = compute_transition_objective(z_before, z_after, z_pred, config=config)
        scalars = terms.scalars()

        self.assertIn("loss/action_use_margin", scalars)
        self.assertIn("loss/action_use_margin_weighted", scalars)
        self.assertAlmostEqual(scalars["loss/action_use_margin"], 0.1)
        self.assertAlmostEqual(
            scalars["loss/total"],
            scalars["loss/prediction_mse"]
            + scalars["loss/sigreg_weighted"]
            + scalars["loss/action_use_margin_weighted"],
        )

    def test_action_use_margin_requires_explicit_config_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "enable_action_use_margin"):
            ObjectiveConfig(action_use_margin_weight=0.1)
        with self.assertRaisesRegex(ValueError, "enable_action_use_margin"):
            ObjectiveConfig(action_use_margin=0.1)
        with self.assertRaisesRegex(ValueError, "nonzero action_use_margin_weight"):
            ObjectiveConfig(enable_action_use_margin=True, action_use_margin=0.1)
        with self.assertRaisesRegex(ValueError, "nonzero action_use_margin"):
            ObjectiveConfig(enable_action_use_margin=True, action_use_margin_weight=0.1)

    def test_action_swap_contrastive_penalizes_swapped_action_winning(self) -> None:
        z_after = np.array([[1.0, 0.0], [-1.0, 0.0]])
        z_pred = np.array([[0.95, 0.0], [-0.95, 0.0]])
        z_pred_swapped_far = np.array([[-1.0, 0.0], [1.0, 0.0]])
        z_pred_swapped_close = z_pred.copy()

        satisfied = compute_action_swap_contrastive_loss(
            z_after,
            z_pred,
            z_pred_swapped_far,
            margin=0.1,
        )
        violated = compute_action_swap_contrastive_loss(
            z_after,
            z_pred,
            z_pred_swapped_close,
            margin=0.1,
        )

        self.assertEqual(satisfied, 0.0)
        self.assertGreater(violated, 0.0)

    def test_transition_objective_reports_action_swap_and_inverse_action_terms(self) -> None:
        z_before = np.array([[0.0, 0.0], [0.0, 1.0]])
        z_after = np.array([[1.0, 0.0], [1.0, 1.0]])
        z_pred = np.array([[0.9, 0.0], [0.9, 1.0]])
        z_pred_swapped = z_before.copy()
        action_emb = np.array([[0.5, 0.1], [0.1, 0.5]])
        action_reconstruction = np.array([[0.4, 0.1], [0.1, 0.4]])
        config = ObjectiveConfig(
            sigreg_weight=0.0,
            enable_action_swap_contrastive=True,
            action_swap_contrastive_weight=0.2,
            action_swap_contrastive_margin=0.1,
            enable_inverse_action_reconstruction=True,
            inverse_action_reconstruction_weight=0.3,
            sigreg_num_proj=8,
            sigreg_seed=3,
        )

        terms = compute_transition_objective(
            z_before,
            z_after,
            z_pred,
            config=config,
            z_pred_after_swapped=z_pred_swapped,
            action_emb=action_emb,
            action_reconstruction=action_reconstruction,
        )
        scalars = terms.scalars()

        self.assertIn("loss/action_swap_contrastive", scalars)
        self.assertIn("loss/action_swap_contrastive_weighted", scalars)
        self.assertIn("loss/inverse_action_reconstruction", scalars)
        self.assertIn("loss/inverse_action_reconstruction_weighted", scalars)
        self.assertAlmostEqual(
            scalars["loss/inverse_action_reconstruction"],
            compute_inverse_action_reconstruction_loss(action_reconstruction, action_emb),
        )

    def test_action_swap_and_inverse_action_require_explicit_gates(self) -> None:
        with self.assertRaisesRegex(ValueError, "enable_action_swap_contrastive"):
            ObjectiveConfig(action_swap_contrastive_weight=0.1)
        with self.assertRaisesRegex(ValueError, "enable_action_swap_contrastive"):
            ObjectiveConfig(action_swap_contrastive_margin=0.1)
        with self.assertRaisesRegex(ValueError, "nonzero action_swap_contrastive_weight"):
            ObjectiveConfig(enable_action_swap_contrastive=True, action_swap_contrastive_margin=0.1)
        with self.assertRaisesRegex(ValueError, "nonzero action_swap_contrastive_margin"):
            ObjectiveConfig(enable_action_swap_contrastive=True, action_swap_contrastive_weight=0.1)
        with self.assertRaisesRegex(ValueError, "enable_inverse_action_reconstruction"):
            ObjectiveConfig(inverse_action_reconstruction_weight=0.1)
        with self.assertRaisesRegex(ValueError, "nonzero inverse_action_reconstruction_weight"):
            ObjectiveConfig(enable_inverse_action_reconstruction=True)


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

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_action_use_margin_only_backprops_through_prediction(self) -> None:
        import torch

        z_before = torch.zeros(2, 2, requires_grad=True)
        z_after = torch.ones(2, 2, requires_grad=True)
        z_pred = torch.zeros(2, 2, requires_grad=True)

        loss = compute_action_use_margin_loss(z_before, z_after, z_pred, margin=0.1)
        loss.backward()

        self.assertIsNotNone(z_pred.grad)
        self.assertIsNone(z_before.grad)
        self.assertIsNone(z_after.grad)


if __name__ == "__main__":
    unittest.main()
