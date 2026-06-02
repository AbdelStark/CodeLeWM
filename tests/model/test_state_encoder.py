from __future__ import annotations

import importlib.util
import unittest

from codelewm.model import (
    LATENT_DIM,
    STATE_SEQUENCE_LENGTH,
    CodeStateEncoder,
    CodeStateEncoderConfig,
    ModelRuntimeUnavailableError,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class CodeStateEncoderConfigTest(unittest.TestCase):
    def test_default_config_matches_v0_1_contract(self) -> None:
        config = CodeStateEncoderConfig()

        self.assertEqual(config.max_length, STATE_SEQUENCE_LENGTH)
        self.assertEqual(config.latent_dim, LATENT_DIM)
        self.assertEqual(config.embed_dim, LATENT_DIM)

    def test_config_validates_positive_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            CodeStateEncoderConfig(vocab_size=1)
        with self.assertRaisesRegex(ValueError, "max_length"):
            CodeStateEncoderConfig(max_length=0)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            CodeStateEncoderConfig(latent_dim=0)

    def test_default_encoder_type_is_pool(self) -> None:
        self.assertEqual(CodeStateEncoderConfig().encoder_type, "pool")

    def test_config_validates_transformer_params(self) -> None:
        with self.assertRaisesRegex(ValueError, "encoder_type"):
            CodeStateEncoderConfig(encoder_type="mlp")
        with self.assertRaisesRegex(ValueError, "num_layers"):
            CodeStateEncoderConfig(encoder_type="transformer", num_layers=0)
        with self.assertRaisesRegex(ValueError, "num_heads"):
            CodeStateEncoderConfig(encoder_type="transformer", embed_dim=8, num_heads=3)


class CodeStateEncoderTorchTest(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_encoder_projects_packed_state_to_latent_dim(self) -> None:
        import torch

        config = CodeStateEncoderConfig(max_length=8, dropout=0.0)
        encoder = CodeStateEncoder(config)
        input_ids = torch.tensor(
            [
                [1, 2, 3, 0, 0, 0, 0, 0],
                [4, 5, 6, 7, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        attention_mask = input_ids.ne(0)
        segment_ids = torch.zeros_like(input_ids)
        changed_hunk_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        changed_hunk_mask[:, :2] = True

        output = encoder(input_ids, attention_mask, segment_ids, changed_hunk_mask)

        self.assertEqual(tuple(output.shape), (2, LATENT_DIM))
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_transformer_encoder_projects_to_latent_dim(self) -> None:
        # RFC-0015 WS-C1: the optional transformer encoder must produce the same
        # (batch, latent_dim) contract as the pool encoder, finite, with padding
        # masked out.
        import torch

        config = CodeStateEncoderConfig(
            max_length=8, dropout=0.0, encoder_type="transformer", num_layers=2, num_heads=4
        )
        encoder = CodeStateEncoder(config)
        encoder.eval()
        input_ids = torch.tensor(
            [
                [1, 2, 3, 0, 0, 0, 0, 0],
                [4, 5, 6, 7, 8, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        attention_mask = input_ids.ne(0)
        segment_ids = torch.zeros_like(input_ids)
        with torch.no_grad():
            output = encoder(input_ids, attention_mask, segment_ids)
        self.assertEqual(tuple(output.shape), (2, LATENT_DIM))
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_encoder_rejects_sequence_length_mismatch(self) -> None:
        import torch

        encoder = CodeStateEncoder(CodeStateEncoderConfig(max_length=8, dropout=0.0))
        values = torch.ones(1, 7, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "sequence length"):
            encoder(values, values.bool(), values)

    @unittest.skipIf(TORCH_AVAILABLE, "torch is installed")
    def test_encoder_reports_missing_torch_runtime(self) -> None:
        with self.assertRaisesRegex(ModelRuntimeUnavailableError, "requires torch"):
            CodeStateEncoder()


if __name__ == "__main__":
    unittest.main()
