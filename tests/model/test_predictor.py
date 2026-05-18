from __future__ import annotations

import importlib.util
import unittest

from codelewm.model import (
    LATENT_DIM,
    CodeLatentPredictor,
    CodeLatentPredictorConfig,
    ModelRuntimeUnavailableError,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
EINOPS_AVAILABLE = importlib.util.find_spec("einops") is not None


class CodeLatentPredictorConfigTest(unittest.TestCase):
    def test_default_config_matches_v0_1_one_step_contract(self) -> None:
        config = CodeLatentPredictorConfig()

        self.assertEqual(config.history_size, 1)
        self.assertEqual(config.num_preds, 1)
        self.assertEqual(config.latent_dim, LATENT_DIM)
        self.assertEqual(config.action_dim, LATENT_DIM)

    def test_multi_step_prediction_is_rejected_until_extension_exists(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_preds=1"):
            CodeLatentPredictorConfig(num_preds=2)

    def test_config_validates_positive_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "history_size"):
            CodeLatentPredictorConfig(history_size=0)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            CodeLatentPredictorConfig(latent_dim=0)


class CodeLatentPredictorTorchTest(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_one_step_forward_projects_pooled_code_latents(self) -> None:
        import torch
        from torch import nn

        class AddPredictor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.last_x_shape: tuple[int, ...] | None = None
                self.last_c_shape: tuple[int, ...] | None = None

            def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
                self.last_x_shape = tuple(x.shape)
                self.last_c_shape = tuple(c.shape)
                return x + c

        class DoubleProjection(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2.0

        predictor = AddPredictor()
        model = CodeLatentPredictor(
            CodeLatentPredictorConfig(latent_dim=4, action_dim=4, hidden_dim=4),
            predictor=predictor,
            pred_proj=DoubleProjection(),
        )

        output = model.predict_after(torch.ones(2, 4), torch.full((2, 4), 3.0))

        self.assertEqual(predictor.last_x_shape, (2, 1, 4))
        self.assertEqual(predictor.last_c_shape, (2, 1, 4))
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertTrue(torch.allclose(output, torch.full((2, 4), 8.0)))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_predictor_rejects_shape_mismatches(self) -> None:
        import torch
        from torch import nn

        model = CodeLatentPredictor(
            CodeLatentPredictorConfig(latent_dim=4, action_dim=4),
            predictor=nn.Identity(),
        )

        with self.assertRaisesRegex(ValueError, "latent dimension"):
            model.predict_after(torch.ones(2, 3), torch.ones(2, 4))
        with self.assertRaisesRegex(ValueError, "batch/history"):
            model.predict_after(torch.ones(2, 4), torch.ones(3, 4))

    @unittest.skipUnless(TORCH_AVAILABLE and EINOPS_AVAILABLE, "torch/einops runtime is not installed")
    def test_default_arpredictor_accepts_one_step_code_latents(self) -> None:
        import torch

        model = CodeLatentPredictor(
            CodeLatentPredictorConfig(
                latent_dim=8,
                action_dim=8,
                hidden_dim=8,
                depth=1,
                heads=2,
                mlp_dim=16,
                dim_head=4,
                dropout=0.0,
                emb_dropout=0.0,
            )
        )

        output = model(torch.randn(2, 8), torch.randn(2, 8))

        self.assertEqual(tuple(output.shape), (2, 8))

    @unittest.skipIf(TORCH_AVAILABLE, "torch is installed")
    def test_predictor_reports_missing_torch_runtime(self) -> None:
        with self.assertRaisesRegex(ModelRuntimeUnavailableError, "requires torch"):
            CodeLatentPredictor()


if __name__ == "__main__":
    unittest.main()
