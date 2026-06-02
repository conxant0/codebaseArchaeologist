import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from codearch.index import build_index
from codearch.index import load_artifacts


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, documents, show_progress_bar=False):
        return [[0.1, 0.2] for _document in documents]


class FakeCollection:
    def __init__(self):
        self.ids = []

    def add(self, ids, documents, embeddings, metadatas):
        self.ids.extend(ids)

    def count(self):
        return len(self.ids)


class FakePersistentClient:
    collection = FakeCollection()

    def __init__(self, path):
        self.path = path
        FakePersistentClient.collection = FakeCollection()

    def list_collections(self):
        return []

    def create_collection(self, name):
        return FakePersistentClient.collection


class IndexTest(unittest.TestCase):
    def test_load_artifacts_reads_processed_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_dir = Path(temp_dir) / "processed"
            artifacts_dir = processed_dir / "owner_repo"
            artifacts_dir.mkdir(parents=True)

            artifacts = [
                {
                    "id": "issue_1",
                    "source": "Issue #1",
                    "text": "Title: Fix auth",
                    "metadata": {"type": "issue", "repo": "owner/repo"},
                },
                {
                    "id": "commit_abc1234",
                    "source": "Commit abc1234",
                    "text": "Message: Add logs",
                    "metadata": {"type": "commit", "repo": "owner/repo"},
                },
            ]
            artifacts_path = artifacts_dir / "artifacts.jsonl"
            artifacts_path.write_text(
                "\n".join(json.dumps(artifact) for artifact in artifacts) + "\n",
                encoding="utf-8",
            )

            with patch("codearch.index.PROCESSED_DATA_DIR", processed_dir):
                loaded_artifacts = load_artifacts("owner", "repo")

            self.assertEqual(loaded_artifacts, artifacts)

    def test_build_index_embeds_in_batches_and_returns_collection_count(self):
        fake_chromadb = types.SimpleNamespace(PersistentClient=FakePersistentClient)
        fake_sentence_transformers = types.SimpleNamespace(
            SentenceTransformer=FakeSentenceTransformer
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            processed_dir = Path(temp_dir) / "processed"
            artifacts_dir = processed_dir / "owner_repo"
            artifacts_dir.mkdir(parents=True)
            artifacts = [
                {
                    "id": f"issue_{number}",
                    "source": f"Issue #{number}",
                    "text": f"Title: Issue {number}",
                    "metadata": {"type": "issue"},
                }
                for number in range(3)
            ]
            (artifacts_dir / "artifacts.jsonl").write_text(
                "\n".join(json.dumps(artifact) for artifact in artifacts) + "\n",
                encoding="utf-8",
            )

            with (
                patch("codearch.index.PROCESSED_DATA_DIR", processed_dir),
                patch("codearch.index.CHROMA_DATA_DIR", Path(temp_dir) / "chroma"),
                patch("codearch.index.EMBEDDING_BATCH_SIZE", 2),
                patch.dict(
                    sys.modules,
                    {
                        "chromadb": fake_chromadb,
                        "sentence_transformers": fake_sentence_transformers,
                    },
                ),
                patch("builtins.print") as mock_print,
            ):
                indexed_count = build_index("owner", "repo")

        self.assertEqual(indexed_count, 3)
        mock_print.assert_any_call("Embedding artifacts...")
        mock_print.assert_any_call("Total artifacts to embed: 3")
        mock_print.assert_any_call("  batch 1/2")
        mock_print.assert_any_call("  batch 2/2")
        mock_print.assert_any_call("Final Chroma collection count: 3")


if __name__ == "__main__":
    unittest.main()
