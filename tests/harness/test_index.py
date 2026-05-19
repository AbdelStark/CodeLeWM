from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codelewm.harness import (
    TRANSITION_INDEX_SCHEMA_VERSION,
    TransitionIndex,
    TransitionIndexEntry,
    TransitionIndexError,
    build_transition_index,
    read_transition_index,
    transition_index_header_json_schema,
    write_transition_index,
)
from codelewm.observability import (
    read_artifact_manifest,
    validate_artifact_checksums,
)


def _entry(index: int, *, split: str = "train") -> TransitionIndexEntry:
    return TransitionIndexEntry(
        transition_id=f"t-{index:04d}",
        split=split,
        source="local_repo",
        repo="example/repo",
        path=f"pkg/mod_{index:04d}.py",
        edit_size=index + 1,
        metadata={"index": index},
    )


def _fixture_vectors() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def _fixture_index() -> TransitionIndex:
    vectors = _fixture_vectors()
    entries = tuple(_entry(i) for i in range(vectors.shape[0]))
    return build_transition_index(
        name="fixture-1k",
        entries=entries,
        vectors=vectors,
        distance="l2",
        metadata={"build_seed": 0},
    )


class TransitionIndexBuildTest(unittest.TestCase):
    def test_index_records_dimensions_and_counts(self) -> None:
        index = _fixture_index()

        self.assertEqual(index.schema_version, TRANSITION_INDEX_SCHEMA_VERSION)
        self.assertEqual(index.count, 4)
        self.assertEqual(index.dim, 3)
        self.assertEqual(index.distance, "l2")

    def test_index_rejects_dimension_mismatch_between_vectors_and_entries(self) -> None:
        vectors = _fixture_vectors()[:3]
        entries = tuple(_entry(i) for i in range(4))

        with self.assertRaisesRegex(TransitionIndexError, "row count"):
            build_transition_index(name="bad", entries=entries, vectors=vectors)

    def test_index_rejects_duplicate_transition_ids(self) -> None:
        entries = (_entry(0), _entry(1), _entry(0))
        vectors = np.eye(3, dtype=np.float32)

        with self.assertRaisesRegex(TransitionIndexError, "duplicate transition_id"):
            build_transition_index(name="dup", entries=entries, vectors=vectors)

    def test_index_rejects_non_finite_vectors(self) -> None:
        vectors = np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float32)
        entries = (_entry(0), _entry(1))

        with self.assertRaisesRegex(TransitionIndexError, "finite"):
            build_transition_index(name="nan", entries=entries, vectors=vectors)

    def test_index_rejects_unsupported_distance(self) -> None:
        with self.assertRaisesRegex(TransitionIndexError, "distance"):
            build_transition_index(
                name="bad-distance",
                entries=(_entry(0), _entry(1)),
                vectors=np.eye(2, dtype=np.float32),
                distance="hamming",  # type: ignore[arg-type]
            )


class TransitionIndexSearchTest(unittest.TestCase):
    def test_l2_search_returns_nearest_entries_in_distance_order(self) -> None:
        index = _fixture_index()
        query = np.array([1.0, 0.05, 0.0], dtype=np.float32)

        hits = index.search(query, k=3)

        self.assertEqual([hit.entry.transition_id for hit in hits[:1]], ["t-0000"])
        ranks = [hit.rank for hit in hits]
        self.assertEqual(ranks, list(range(1, len(hits) + 1)))
        distances = [hit.distance for hit in hits]
        self.assertEqual(distances, sorted(distances))

    def test_search_excludes_listed_ids(self) -> None:
        index = _fixture_index()
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        hits = index.search(query, k=2, exclude_ids=("t-0000",))

        ids = {hit.entry.transition_id for hit in hits}
        self.assertNotIn("t-0000", ids)
        self.assertEqual(len(hits), 2)

    def test_cosine_search_ranks_by_direction(self) -> None:
        vectors = np.array(
            [
                [10.0, 0.0],
                [0.0, 10.0],
                [-10.0, 0.0],
            ],
            dtype=np.float32,
        )
        entries = tuple(_entry(i) for i in range(3))
        index = build_transition_index(
            name="cosine-fixture",
            entries=entries,
            vectors=vectors,
            distance="cosine",
        )

        hits = index.search(np.array([1.0, 0.0], dtype=np.float32), k=3)

        self.assertEqual(hits[0].entry.transition_id, "t-0000")
        self.assertEqual(hits[-1].entry.transition_id, "t-0002")

    def test_search_rejects_dim_mismatch(self) -> None:
        index = _fixture_index()
        bad_query = np.array([1.0, 0.0], dtype=np.float32)

        with self.assertRaisesRegex(TransitionIndexError, "dim"):
            index.search(bad_query, k=1)

    def test_search_rejects_non_positive_k(self) -> None:
        index = _fixture_index()
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        with self.assertRaisesRegex(TransitionIndexError, "k must be positive"):
            index.search(query, k=0)


class TransitionIndexManifestTest(unittest.TestCase):
    def test_round_trip_through_disk_preserves_index_and_validates_manifest(self) -> None:
        index = _fixture_index()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, manifest_path = write_transition_index(
                index,
                root,
                command=("codelewm", "index"),
                config={"name": index.name},
                parent_artifacts=("dataset-fixture",),
                source_git_sha="0" * 40,
                artifact_id="index-fixture",
            )
            self.assertEqual(manifest_path, root / "manifest.json")

            artifact_manifest = read_artifact_manifest(manifest_path)
            validate_artifact_checksums(artifact_manifest, root=root)

            reloaded = read_transition_index(root)

        self.assertEqual(reloaded.name, index.name)
        self.assertEqual(reloaded.count, index.count)
        self.assertEqual(reloaded.dim, index.dim)
        self.assertEqual(reloaded.distance, index.distance)
        np.testing.assert_array_equal(reloaded.vectors, index.vectors)
        self.assertEqual(
            [entry.to_dict() for entry in reloaded.entries],
            [entry.to_dict() for entry in index.entries],
        )
        self.assertEqual(manifest.artifact_kind, "index")
        self.assertEqual(manifest.parent_artifacts, ("dataset-fixture",))
        artifact_paths = {file.path for file in artifact_manifest.files}
        self.assertSetEqual(artifact_paths, {"vectors.npy", "entries.jsonl", "index.json"})

    def test_read_fails_when_manifest_checksum_is_tampered(self) -> None:
        index = _fixture_index()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_transition_index(index, root)
            entries_path = root / "entries.jsonl"
            entries_path.write_text(entries_path.read_text() + "\n", encoding="utf-8")

            with self.assertRaisesRegex(TransitionIndexError, "validation"):
                read_transition_index(root)

    def test_read_fails_when_header_schema_version_is_unsupported(self) -> None:
        index = _fixture_index()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_transition_index(index, root)
            header_path = root / "index.json"
            header = json.loads(header_path.read_text(encoding="utf-8"))
            header["schema_version"] = "codelewm.transition_index.v0"
            header_path.write_text(json.dumps(header), encoding="utf-8")

            with self.assertRaisesRegex(TransitionIndexError, "unsupported index schema_version"):
                read_transition_index(root, verify_manifest=False)

    def test_header_json_schema_pins_required_fields(self) -> None:
        schema = transition_index_header_json_schema()

        self.assertEqual(schema["properties"]["schema_version"]["const"], TRANSITION_INDEX_SCHEMA_VERSION)
        for field in ("name", "count", "dim", "distance", "vectors_path", "entries_path", "metadata"):
            with self.subTest(field=field):
                self.assertIn(field, schema["required"])

    def test_header_file_contains_schema_version_and_filenames(self) -> None:
        index = _fixture_index()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_transition_index(index, root)

            header = json.loads((root / "index.json").read_text(encoding="utf-8"))

        self.assertEqual(header["schema_version"], TRANSITION_INDEX_SCHEMA_VERSION)
        self.assertEqual(header["vectors_path"], "vectors.npy")
        self.assertEqual(header["entries_path"], "entries.jsonl")
        self.assertEqual(header["count"], index.count)


if __name__ == "__main__":
    unittest.main()
