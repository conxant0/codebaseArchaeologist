import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codearch.normalize import normalize_repo


class NormalizeTest(unittest.TestCase):
    def test_normalize_repo_returns_artifacts_and_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"
            repo_raw_dir = raw_dir / "owner_repo"
            repo_raw_dir.mkdir(parents=True)

            (repo_raw_dir / "issues.json").write_text(
                json.dumps(
                    [
                        {
                            "number": 123,
                            "title": "Fix login error",
                            "body": "Users cannot log in.",
                            "html_url": "https://github.com/owner/repo/issues/123",
                            "labels": [{"name": "bug"}, {"name": "auth"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (repo_raw_dir / "pulls.json").write_text(
                json.dumps(
                    [
                        {
                            "number": 45,
                            "title": "Add audit logs",
                            "body": "Records security events.",
                            "state": "closed",
                            "created_at": "2026-06-01T00:00:00Z",
                            "merged_at": "2026-06-02T00:00:00Z",
                            "html_url": "https://github.com/owner/repo/pull/45",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (repo_raw_dir / "commits.json").write_text(
                json.dumps(
                    [
                        {
                            "sha": "abcdef1234567890",
                            "html_url": "https://github.com/owner/repo/commit/abcdef",
                            "commit": {
                                "message": "Fix token refresh",
                                "author": {
                                    "name": "Ada Lovelace",
                                    "date": "2026-06-02T12:00:00Z",
                                },
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with (
                patch("codearch.normalize.RAW_DATA_DIR", raw_dir),
                patch("codearch.normalize.PROCESSED_DATA_DIR", processed_dir),
            ):
                artifacts = normalize_repo("owner", "repo")

            self.assertEqual(len(artifacts), 3)

            issue = artifacts[0]
            self.assertEqual(issue["id"], "issue_123")
            self.assertEqual(issue["source"], "Issue #123")
            self.assertIn("Title: Fix login error", issue["text"])
            self.assertIn("Body: Users cannot log in.", issue["text"])
            self.assertIn("Labels: bug, auth", issue["text"])
            self.assertEqual(
                issue["metadata"],
                {
                    "type": "issue",
                    "repo": "owner/repo",
                    "number": 123,
                    "url": "https://github.com/owner/repo/issues/123",
                    "title": "Fix login error",
                },
            )

            pull_request = artifacts[1]
            self.assertEqual(pull_request["id"], "pr_45")
            self.assertEqual(pull_request["source"], "Pull Request #45")
            self.assertIn("State: closed", pull_request["text"])
            self.assertIn("Created At: 2026-06-01T00:00:00Z", pull_request["text"])
            self.assertIn("Merged At: 2026-06-02T00:00:00Z", pull_request["text"])
            self.assertEqual(pull_request["metadata"]["type"], "pull_request")

            commit = artifacts[2]
            self.assertEqual(commit["id"], "commit_abcdef1")
            self.assertEqual(commit["source"], "Commit abcdef1")
            self.assertIn("Message: Fix token refresh", commit["text"])
            self.assertIn("Author: Ada Lovelace", commit["text"])
            self.assertIn("SHA: abcdef1234567890", commit["text"])
            self.assertEqual(commit["metadata"]["type"], "commit")

            artifacts_path = processed_dir / "owner_repo" / "artifacts.jsonl"
            lines = artifacts_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual([json.loads(line) for line in lines], artifacts)


if __name__ == "__main__":
    unittest.main()
