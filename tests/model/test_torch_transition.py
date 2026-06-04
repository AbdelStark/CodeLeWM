from __future__ import annotations

import importlib.util
import unittest

from codelewm.model import (
    ActionBatch,
    CodeStateBatch,
    TorchCodeTransitionModel,
    TorchCodeTransitionModelConfig,
    TransitionBatch,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class TorchTransitionPassHeadTest(unittest.TestCase):
    def test_pass_head_is_disabled_by_default(self) -> None:
        import torch

        model = _dummy_model(TorchCodeTransitionModelConfig())

        self.assertIsNone(model.pass_head)
        with self.assertRaisesRegex(ValueError, "pass head is disabled"):
            model.pass_logit(
                torch.zeros(2, 256),
                torch.zeros(2, 256),
                torch.zeros(2, 256),
            )

        output = model(_transition_batch(batch_size=2))
        self.assertNotIn("pass_logit", output)

    def test_enabled_pass_head_emits_single_logit_per_transition(self) -> None:
        import torch

        model = _dummy_model(
            TorchCodeTransitionModelConfig(enable_pass_head=True, dropout=0.0)
        )
        z_before = torch.randn(3, 256, requires_grad=True)
        action_emb = torch.randn(3, 256, requires_grad=True)
        z_pred_after = torch.randn(3, 256, requires_grad=True)

        logits = model.pass_logit(z_before, action_emb, z_pred_after)
        logits.mean().backward()

        self.assertEqual(tuple(logits.shape), (3, 1))
        self.assertIsNotNone(z_pred_after.grad)
        assert model.pass_head is not None
        first_layer = model.pass_head[0]
        self.assertIsNotNone(first_layer.weight.grad)

        output = model(_transition_batch(batch_size=3))
        self.assertEqual(tuple(output["pass_logit"].shape), (3, 1))


def _dummy_model(config: TorchCodeTransitionModelConfig) -> TorchCodeTransitionModel:
    return TorchCodeTransitionModel(
        state_encoder=_StateEncoder(),
        action_encoder=_ActionEncoder(),
        predictor=_Predictor(),
        config=config,
    )


class _StateEncoder:
    def __call__(self, input_ids, attention_mask, segment_ids, changed_hunk_mask=None):
        import torch

        del attention_mask, segment_ids, changed_hunk_mask
        return torch.zeros((input_ids.shape[0], 256), dtype=torch.float32)


class _ActionEncoder:
    def __call__(self, input_ids, attention_mask):
        import torch

        del attention_mask
        return torch.ones((input_ids.shape[0], 256), dtype=torch.float32)


class _Predictor:
    def predict_after(self, z_before, action_emb):
        return z_before + action_emb


def _transition_batch(*, batch_size: int) -> TransitionBatch:
    import torch

    state = CodeStateBatch(
        input_ids=torch.zeros((batch_size, 1024), dtype=torch.long),
        attention_mask=torch.ones((batch_size, 1024), dtype=torch.bool),
        segment_ids=torch.zeros((batch_size, 1024), dtype=torch.long),
        changed_hunk_mask=torch.zeros((batch_size, 1024), dtype=torch.bool),
    )
    action = ActionBatch(
        input_ids=torch.zeros((batch_size, 256), dtype=torch.long),
        attention_mask=torch.ones((batch_size, 256), dtype=torch.bool),
        action_view="text",
    )
    return TransitionBatch(state_before=state, action=action, state_after=state)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
