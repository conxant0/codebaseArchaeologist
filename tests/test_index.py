import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codearch.index import load_artifacts


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


if __name__ == "__main__":
    unittest.main()
