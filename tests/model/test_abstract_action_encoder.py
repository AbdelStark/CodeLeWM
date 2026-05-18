from __future__ import annotations

import importlib.util
import unittest

from codelewm.model import (
    ABSTRACT_ACTION_BASE_VOCABULARY,
    ABSTRACT_ACTION_SEQUENCE_LENGTH,
    LATENT_DIM,
    AbstractActionEncoder,
    AbstractActionEncoderConfig,
    AbstractActionTokenizer,
    ModelRuntimeUnavailableError,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class AbstractActionTokenizerTest(unittest.TestCase):
    def test_tokenizer_emits_abstract_action_batch_contract(self) -> None:
        tokenizer = AbstractActionTokenizer()

        batch = tokenizer.encode(
            (
                "OP_UPDATE NODE_Return PATH_DEPTH_2 "
                "OLD_ConstantNum NEW_ConstantNum SCOPE_function SIZE_SMALL",
            )
        )

        self.assertEqual(batch.action_view, "abstract")
        self.assertEqual(batch.expected_sequence_length, ABSTRACT_ACTION_SEQUENCE_LENGTH)
        self.assertEqual(batch.input_ids.shape, (1, ABSTRACT_ACTION_SEQUENCE_LENGTH))
        self.assertEqual(batch.attention_mask.shape, (1, ABSTRACT_ACTION_SEQUENCE_LENGTH))
        self.assertEqual(int(batch.attention_mask.sum()), 7)
        self.assertEqual(batch.input_ids[0, 0], tokenizer.token_id("OP_UPDATE"))
        self.assertTrue((batch.input_ids[0, :7] > 0).all())
        self.assertTrue((batch.input_ids[0, 7:] == 0).all())

    def test_reserved_vocabulary_and_unknown_tokens_are_stable(self) -> None:
        tokenizer = AbstractActionTokenizer()

        self.assertIn("OP_UPDATE", ABSTRACT_ACTION_BASE_VOCABULARY)
        self.assertEqual(tokenizer.token_id("OP_UPDATE"), tokenizer.vocabulary["OP_UPDATE"])
        self.assertEqual(tokenizer.token_id("NODE_CustomThing"), tokenizer.token_id("NODE_CustomThing"))
        self.assertGreater(tokenizer.token_id("NODE_CustomThing"), len(tokenizer.vocabulary))

    def test_batch_encode_stacks_abstract_actions(self) -> None:
        tokenizer = AbstractActionTokenizer(max_length=8)

        batch = tokenizer.batch_encode(
            (
                ("OP_INSERT NODE_ExceptHandler PATH_DEPTH_2 SIZE_MEDIUM",),
                ("OP_DELETE NODE_Assign PATH_DEPTH_4 SIZE_SMALL",),
            )
        )

        self.assertEqual(batch.input_ids.shape, (2, 8))
        self.assertEqual(batch.attention_mask.tolist(), [[True, True, True, True, False, False, False, False]] * 2)
        self.assertEqual(batch.action_view, "abstract")

    def test_empty_abstract_action_is_rejected_before_encoding(self) -> None:
        tokenizer = AbstractActionTokenizer()

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            tokenizer.encode(("   ",))

    def test_config_validates_attention_head_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            AbstractActionEncoderConfig(embed_dim=250, num_heads=8)


class AbstractActionEncoderTorchTest(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_encoder_projects_to_latent_dim(self) -> None:
        import torch

        config = AbstractActionEncoderConfig(max_length=8, num_layers=1, num_heads=4)
        tokenizer = AbstractActionTokenizer(max_length=8, vocab_size=config.vocab_size)
        batch = tokenizer.batch_encode(
            (
                ("OP_UPDATE NODE_Return PATH_DEPTH_2 SIZE_SMALL",),
                ("OP_INSERT NODE_ExceptHandler PATH_DEPTH_2 SIZE_MEDIUM",),
            )
        )
        encoder = AbstractActionEncoder(config)

        output = encoder(
            torch.as_tensor(batch.input_ids),
            torch.as_tensor(batch.attention_mask),
        )

        self.assertEqual(tuple(output.shape), (2, LATENT_DIM))

    @unittest.skipIf(TORCH_AVAILABLE, "torch is installed")
    def test_encoder_reports_missing_torch_runtime(self) -> None:
        with self.assertRaisesRegex(ModelRuntimeUnavailableError, "requires torch"):
            AbstractActionEncoder()


if __name__ == "__main__":
    unittest.main()
