from __future__ import annotations

import unittest

import numpy as np

from codelewm.model import (
    ABSTRACT_ACTION_SEQUENCE_LENGTH,
    LATENT_DIM,
    STATE_SEQUENCE_LENGTH,
    TEXT_ACTION_SEQUENCE_LENGTH,
    ActionBatch,
    CodeStateBatch,
    CodeTransitionModel,
    TransitionBatch,
    expected_action_sequence_length,
    infer_shape,
    resolve_ema_target_encoder_config,
    resolve_state_encoder_arch,
    transition_energy,
)


class TransitionContractTest(unittest.TestCase):
    def test_public_shape_constants_match_v0_1_contract(self) -> None:
        self.assertEqual(STATE_SEQUENCE_LENGTH, 1024)
        self.assertEqual(TEXT_ACTION_SEQUENCE_LENGTH, 256)
        self.assertEqual(ABSTRACT_ACTION_SEQUENCE_LENGTH, 192)
        self.assertEqual(LATENT_DIM, 256)

    def test_batch_dataclasses_preserve_inputs(self) -> None:
        state = CodeStateBatch(
            input_ids=[[1, 2, 3]],
            attention_mask=[[True, True, True]],
            segment_ids=[[0, 0, 1]],
        )
        action = ActionBatch(
            input_ids=[[4, 5]],
            attention_mask=[[True, True]],
            action_view="text",
        )
        batch = TransitionBatch(state_before=state, action=action, state_after=state)

        self.assertIs(batch.state_before, state)
        self.assertEqual(action.expected_sequence_length, TEXT_ACTION_SEQUENCE_LENGTH)

    def test_expected_action_sequence_lengths_are_locked(self) -> None:
        self.assertEqual(expected_action_sequence_length("text"), 256)
        self.assertEqual(expected_action_sequence_length("abstract"), 192)
        self.assertEqual(expected_action_sequence_length("patch"), 512)

    def test_infer_shape_supports_sequences_and_arrays(self) -> None:
        self.assertEqual(infer_shape([[1, 2], [3, 4]]), (2, 2))
        self.assertEqual(infer_shape(np.zeros((3, 4, 5))), (3, 4, 5))

    def test_transition_energy_matches_hand_computed_squared_distance(self) -> None:
        self.assertEqual(transition_energy([1.0, 2.0], [3.0, 4.0]), 8.0)
        self.assertEqual(
            transition_energy([[1.0, 2.0], [0.0, 0.0]], [[3.0, 4.0], [0.0, 0.0]]),
            [8.0, 0.0],
        )

    def test_transition_energy_supports_numpy_and_reductions(self) -> None:
        pred = np.array([[1.0, 2.0], [2.0, 2.0]])
        target = np.array([[0.0, 0.0], [1.0, 1.0]])

        np.testing.assert_allclose(transition_energy(pred, target), [5.0, 2.0])
        self.assertEqual(transition_energy(pred, target, reduction="sum"), 7.0)
        self.assertEqual(transition_energy(pred, target, reduction="mean"), 3.5)

    def test_transition_energy_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            transition_energy([[1.0, 2.0]], [1.0, 2.0])

    def test_transition_model_interface_delegates_transition_energy(self) -> None:
        model = CodeTransitionModel()

        self.assertEqual(model.transition_energy([1.0, 2.0], [1.0, 0.0]), 4.0)
        with self.assertRaises(NotImplementedError):
            model.encode_state(
                CodeStateBatch(
                    input_ids=[[1]],
                    attention_mask=[[True]],
                    segment_ids=[[0]],
                )
            )


class ResolveStateEncoderArchTest(unittest.TestCase):
    """The eval loader must rebuild the right state-encoder architecture."""

    def test_prefers_persisted_compatibility_fields(self) -> None:
        arch = resolve_state_encoder_arch(
            {
                "state_encoder_type": "transformer",
                "state_encoder_layers": 6,
                "state_encoder_heads": 4,
            },
            {},
        )
        self.assertEqual(arch, ("transformer", 6, 4))

    def test_infers_transformer_from_state_dict_for_legacy_checkpoints(self) -> None:
        # Pre-persistence v0.7 checkpoints carry transformer weights but no
        # encoder fields in compatibility_config.wm; infer from the weights.
        state_dict = {
            "encoder.encoder.layers.0.self_attn.in_proj_weight": None,
            "encoder.encoder.layers.1.norm1.weight": None,
            "encoder.pool.weight": None,
        }
        self.assertEqual(resolve_state_encoder_arch({}, state_dict), ("transformer", 2, 8))

    def test_defaults_to_pool_without_transformer_weights(self) -> None:
        self.assertEqual(
            resolve_state_encoder_arch({}, {"encoder.proj.weight": None}),
            ("pool", 4, 8),
        )


class ResolveEmaTargetEncoderConfigTest(unittest.TestCase):
    def test_prefers_persisted_compatibility_fields(self) -> None:
        self.assertEqual(
            resolve_ema_target_encoder_config(
                {"enable_ema_target_encoder": True, "ema_target_decay": 0.95},
                {},
            ),
            (True, 0.95),
        )

    def test_infers_enabled_from_target_encoder_weights(self) -> None:
        self.assertEqual(
            resolve_ema_target_encoder_config(
                {},
                {"target_encoder.token_embedding.weight": None},
            ),
            (True, 0.99),
        )

    def test_defaults_off_without_target_weights(self) -> None:
        self.assertEqual(
            resolve_ema_target_encoder_config({}, {"encoder.proj.weight": None}),
            (False, 0.99),
        )


if __name__ == "__main__":
    unittest.main()
