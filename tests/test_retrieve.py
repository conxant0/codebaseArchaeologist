import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from codearch.retrieve import IndexNotFoundError
from codearch.retrieve import retrieve_relevant_artifacts


class Embedding:
    def tolist(self):
        return [0.1, 0.2, 0.3]


class FakeSentenceTransformer:
    model_name = None

    def __init__(self, model_name):
        FakeSentenceTransformer.model_name = model_name

    def encode(self, text, show_progress_bar=False):
        self.text = text
        self.show_progress_bar = show_progress_bar
        return Embedding()


class FakeCollection:
    query_args = None

    def query(self, **kwargs):
        FakeCollection.query_args = kwargs
        return {
            "ids": [["pr_1", "issue_2"]],
            "documents": [["Title: Add DI", "Title: Track DI issue"]],
            "metadatas": [
                [
                    {"source": "Pull Request #1", "type": "pull_request"},
                    {"source": "Issue #2", "type": "issue"},
                ]
            ],
            "distances": [[0.123, 0.456]],
        }


class FakePersistentClient:
    path = None
    collection_name = None

    def __init__(self, path):
        FakePersistentClient.path = path

    def get_collection(self, name):
        FakePersistentClient.collection_name = name
        return FakeCollection()


class RetrieveTest(unittest.TestCase):
    def test_retrieve_relevant_artifacts_queries_chroma(self):
        fake_chromadb = types.SimpleNamespace(PersistentClient=FakePersistentClient)
        fake_sentence_transformers = types.SimpleNamespace(
            SentenceTransformer=FakeSentenceTransformer
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_dir = Path(temp_dir) / "chroma"
            (chroma_dir / "owner_repo").mkdir(parents=True)
            with (
                patch("codearch.retrieve.CHROMA_DATA_DIR", chroma_dir),
                patch.dict(
                    sys.modules,
                    {
                        "chromadb": fake_chromadb,
                        "sentence_transformers": fake_sentence_transformers,
                    },
                ),
            ):
                artifacts = retrieve_relevant_artifacts(
                    "owner",
                    "repo",
                    "Why was dependency injection added?",
                    top_k=2,
                )

        self.assertEqual(FakeSentenceTransformer.model_name, "all-MiniLM-L6-v2")
        self.assertEqual(FakePersistentClient.path, str(chroma_dir / "owner_repo"))
        self.assertEqual(FakePersistentClient.collection_name, "codearch_artifacts")
        self.assertEqual(
            FakeCollection.query_args,
            {
                "query_embeddings": [[0.1, 0.2, 0.3]],
                "n_results": 2,
                "include": ["documents", "metadatas", "distances"],
            },
        )
        self.assertEqual(
            artifacts,
            [
                {
                    "id": "pr_1",
                    "source": "Pull Request #1",
                    "text": "Title: Add DI",
                    "distance": 0.123,
                    "metadata": {
                        "source": "Pull Request #1",
                        "type": "pull_request",
                    },
                },
                {
                    "id": "issue_2",
                    "source": "Issue #2",
                    "text": "Title: Track DI issue",
                    "distance": 0.456,
                    "metadata": {"source": "Issue #2", "type": "issue"},
                },
            ],
        )

    def test_retrieve_raises_index_not_found_when_chroma_path_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_dir = Path(temp_dir) / "chroma"
            with patch("codearch.retrieve.CHROMA_DATA_DIR", chroma_dir):
                with self.assertRaises(IndexNotFoundError) as context:
                    retrieve_relevant_artifacts(
                        "owner",
                        "repo",
                        "Where is auth handled?",
                    )

        self.assertEqual(str(context.exception), "No index found for owner/repo")

    def test_retrieve_wraps_missing_collection_as_index_not_found(self):
        class MissingCollectionClient(FakePersistentClient):
            def get_collection(self, name):
                raise ValueError("Collection does not exist")

        fake_chromadb = types.SimpleNamespace(PersistentClient=MissingCollectionClient)
        fake_sentence_transformers = types.SimpleNamespace(
            SentenceTransformer=FakeSentenceTransformer
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_dir = Path(temp_dir) / "chroma"
            (chroma_dir / "owner_repo").mkdir(parents=True)
            with (
                patch("codearch.retrieve.CHROMA_DATA_DIR", chroma_dir),
                patch.dict(
                    sys.modules,
                    {
                        "chromadb": fake_chromadb,
                        "sentence_transformers": fake_sentence_transformers,
                    },
                ),
            ):
                with self.assertRaises(IndexNotFoundError):
                    retrieve_relevant_artifacts(
                        "owner",
                        "repo",
                        "Where is auth handled?",
                    )


if __name__ == "__main__":
    unittest.main()
