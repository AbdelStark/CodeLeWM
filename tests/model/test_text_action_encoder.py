from __future__ import annotations

import importlib.util
import unittest

from codelewm.model import (
    LATENT_DIM,
    TEXT_ACTION_SEQUENCE_LENGTH,
    ModelRuntimeUnavailableError,
    TextActionEncoder,
    TextActionEncoderConfig,
    TextActionTokenizer,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TextActionTokenizerTest(unittest.TestCase):
    def test_tokenizer_emits_text_action_batch_contract(self) -> None:
        tokenizer = TextActionTokenizer()

        batch = tokenizer.encode("Rename value to result")

        self.assertEqual(batch.action_view, "text")
        self.assertEqual(batch.expected_sequence_length, TEXT_ACTION_SEQUENCE_LENGTH)
        self.assertEqual(batch.input_ids.shape, (1, TEXT_ACTION_SEQUENCE_LENGTH))
        self.assertEqual(batch.attention_mask.shape, (1, TEXT_ACTION_SEQUENCE_LENGTH))
        self.assertEqual(int(batch.attention_mask.sum()), 4)
        self.assertTrue((batch.input_ids[0, :4] > 0).all())
        self.assertTrue((batch.input_ids[0, 4:] == 0).all())

    def test_batch_encode_stacks_actions(self) -> None:
        tokenizer = TextActionTokenizer(max_length=8)

        batch = tokenizer.batch_encode(("update return", "add handler"))

        self.assertEqual(batch.input_ids.shape, (2, 8))
        self.assertEqual(batch.attention_mask.tolist(), [[True, True, False, False, False, False, False, False]] * 2)

    def test_empty_action_is_rejected_before_encoding(self) -> None:
        tokenizer = TextActionTokenizer()

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            tokenizer.encode("   ")

    def test_config_validates_attention_head_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            TextActionEncoderConfig(embed_dim=250, num_heads=8)


class TextActionEncoderTorchTest(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_encoder_projects_to_latent_dim(self) -> None:
        import torch

        config = TextActionEncoderConfig(max_length=8, num_layers=1, num_heads=4)
        tokenizer = TextActionTokenizer(max_length=8, vocab_size=config.vocab_size)
        batch = tokenizer.batch_encode(("update return", "add handler"))
        encoder = TextActionEncoder(config)

        output = encoder(
            torch.as_tensor(batch.input_ids),
            torch.as_tensor(batch.attention_mask),
        )

        self.assertEqual(tuple(output.shape), (2, LATENT_DIM))

    @unittest.skipIf(TORCH_AVAILABLE, "torch is installed")
    def test_encoder_reports_missing_torch_runtime(self) -> None:
        with self.assertRaisesRegex(ModelRuntimeUnavailableError, "requires torch"):
            TextActionEncoder()


if __name__ == "__main__":
    unittest.main()
