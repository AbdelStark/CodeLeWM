from __future__ import annotations

import unittest

import numpy as np

from codelewm.data import (
    ActionExtractionConfig,
    RawEditRecord,
    SyntheticSourceFile,
    extract_edit_action,
    generate_synthetic_records,
)


class ActionConditioningRegressionTest(unittest.TestCase):
    def test_true_synthetic_actions_beat_no_action_and_shuffled_actions(self) -> None:
        records = _fixture_records()
        model = _LinearTransitionSmoke.fit(records)

        for index, record in enumerate(records):
            with self.subTest(transform=record.metadata["synthetic_transform_id"]):
                before = _state_features(record.before)
                after = _state_features(record.after)
                true_action = _action_features(record)
                shuffled_action = _action_features(records[(index + 1) % len(records)])
                no_action = np.zeros_like(true_action)

                true_energy = _energy(model.predict(before, true_action), after)
                shuffled_energy = _energy(model.predict(before, shuffled_action), after)
                no_action_energy = _energy(model.predict(before, no_action), after)

                self.assertLess(true_energy, shuffled_energy)
                self.assertLess(true_energy, no_action_energy)


class _LinearTransitionSmoke:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights

    @classmethod
    def fit(cls, records: tuple[RawEditRecord, ...]) -> "_LinearTransitionSmoke":
        inputs = []
        targets = []
        for record in records:
            inputs.append(_features(record.before, _action_features(record)))
            targets.append(_state_features(record.after))
        weights = np.linalg.pinv(np.vstack(inputs)) @ np.vstack(targets)
        return cls(weights)

    def predict(self, before: np.ndarray, action: np.ndarray) -> np.ndarray:
        return _features_from_arrays(before, action) @ self.weights


def _fixture_records() -> tuple[RawEditRecord, ...]:
    source = SyntheticSourceFile(
        repo="example/action-conditioning",
        path="pkg/transforms.py",
        contents=(
            "def normalize(value):\n"
            "    items = set([1, 2])\n"
            "    print(value, items)\n"
        ),
        license="mit",
        source_id="action-conditioning-fixture",
    )
    records = generate_synthetic_records(source)
    if len(records) != 3:
        raise AssertionError(f"expected three synthetic transforms, got {len(records)}")
    for record in records:
        action = extract_edit_action(record, config=ActionExtractionConfig(include_patch=False))
        if not action.abstract:
            raise AssertionError(f"missing abstract action for {record.commit}")
    return records


def _state_features(source: str) -> np.ndarray:
    return np.array(
        [
            source.count("value"),
            source.count("result"),
            source.count("return None"),
            source.count("set("),
            source.count("{"),
            source.count("items"),
        ],
        dtype=float,
    )


def _action_features(record: RawEditRecord) -> np.ndarray:
    transform_id = str(record.metadata["synthetic_transform_id"])
    return np.array(
        [
            transform_id == "rename-value-arg-to-result",
            transform_id == "add-explicit-return-none",
            transform_id == "modernize-set-literal",
        ],
        dtype=float,
    )


def _features(source: str, action: np.ndarray) -> np.ndarray:
    return _features_from_arrays(_state_features(source), action)


def _features_from_arrays(before: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.concatenate([before, action, np.ones(1)])


def _energy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.square(prediction - target).sum())


if __name__ == "__main__":
    unittest.main()
