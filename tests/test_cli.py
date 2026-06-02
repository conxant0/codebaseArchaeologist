import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from codearch.cli import app
from codearch.retrieve import IndexNotFoundError


class CliTest(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("codearch.cli.build_index")
    @patch("codearch.cli.normalize_repo")
    @patch("codearch.cli.ingest_repo")
    def test_index_ingests_normalizes_and_indexes_repo(
        self,
        mock_ingest_repo,
        mock_normalize_repo,
        mock_build_index,
    ):
        mock_normalize_repo.return_value = [{"id": "issue_1"}]
        mock_build_index.return_value = 1

        result = self.runner.invoke(app, ["index", "https://github.com/example/repo"])

        self.assertEqual(result.exit_code, 0)
        mock_ingest_repo.assert_called_once_with("https://github.com/example/repo")
        mock_normalize_repo.assert_called_once_with("example", "repo")
        mock_build_index.assert_called_once_with("example", "repo")
        self.assertIn("Indexed 1 artifacts into Chroma", result.output)
        self.assertIn("Normalized artifacts", result.output)

    @patch("codearch.cli.retrieve_relevant_artifacts")
    def test_ask_retrieves_artifacts_for_repo(self, mock_retrieve_relevant_artifacts):
        mock_retrieve_relevant_artifacts.return_value = [
            {
                "id": "pr_1",
                "source": "Pull Request #1",
                "text": "Title: Add dependency injection",
                "distance": 0.123,
                "metadata": {"type": "pull_request"},
            },
        ]

        result = self.runner.invoke(
            app,
            ["ask", "Why was dependency injection added?", "--repo", "fastapi/fastapi"],
        )

        self.assertEqual(result.exit_code, 0)
        mock_retrieve_relevant_artifacts.assert_called_once_with(
            "fastapi",
            "fastapi",
            "Why was dependency injection added?",
        )
        self.assertIn("Top retrieved artifacts:", result.output)
        self.assertIn("- source: Pull Request #1", result.output)
        self.assertIn("type: pull_request", result.output)
        self.assertIn("distance: 0.123", result.output)
        self.assertIn("Title: Add dependency injection", result.output)

    @patch("codearch.cli.retrieve_relevant_artifacts")
    def test_ask_prints_missing_index_message(self, mock_retrieve_relevant_artifacts):
        mock_retrieve_relevant_artifacts.side_effect = IndexNotFoundError(
            "fastapi",
            "fastapi",
        )

        result = self.runner.invoke(
            app,
            ["ask", "Why was dependency injection added?", "--repo", "fastapi/fastapi"],
        )

        self.assertEqual(result.exit_code, 1)
        self.assertIn("No index found for fastapi/fastapi.", result.output)
        self.assertIn(
            "Run: codearch index https://github.com/fastapi/fastapi",
            result.output,
        )

    @patch("codearch.cli.retrieve_relevant_artifacts")
    def test_ask_warns_when_top_result_is_weak(self, mock_retrieve_relevant_artifacts):
        mock_retrieve_relevant_artifacts.return_value = [
            {
                "id": "issue_1",
                "source": "Issue #1",
                "text": "Title: Unrelated issue",
                "distance": 1.31,
                "metadata": {"type": "issue"},
            },
        ]

        result = self.runner.invoke(
            app,
            ["ask", "Why was dependency injection added?", "--repo", "fastapi/fastapi"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "Warning: retrieved artifacts may be weak matches for this question.",
            result.output,
        )
        self.assertIn("Top retrieved artifacts:", result.output)

    @patch("codearch.cli.retrieve_relevant_artifacts")
    def test_ask_does_not_warn_at_weak_threshold(
        self,
        mock_retrieve_relevant_artifacts,
    ):
        mock_retrieve_relevant_artifacts.return_value = [
            {
                "id": "issue_1",
                "source": "Issue #1",
                "text": "Title: Related issue",
                "distance": 1.3,
                "metadata": {"type": "issue"},
            },
        ]

        result = self.runner.invoke(
            app,
            ["ask", "Why was dependency injection added?", "--repo", "fastapi/fastapi"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("Warning:", result.output)

    def test_context_prints_placeholder(self):
        result = self.runner.invoke(app, ["context", "Add audit logging"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "Context placeholder for: Add audit logging",
            result.output,
        )


if __name__ == "__main__":
    unittest.main()
