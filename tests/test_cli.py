import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from codearch.cli import app


class CliTest(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("codearch.cli.ingest_repo")
    def test_index_ingests_repo(self, mock_ingest_repo):
        result = self.runner.invoke(app, ["index", "https://github.com/example/repo"])

        self.assertEqual(result.exit_code, 0)
        mock_ingest_repo.assert_called_once_with("https://github.com/example/repo")
        self.assertIn("Indexed https://github.com/example/repo", result.output)

    def test_ask_prints_placeholder(self):
        result = self.runner.invoke(app, ["ask", "Where is auth handled?"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "Ask placeholder for: Where is auth handled?",
            result.output,
        )

    def test_context_prints_placeholder(self):
        result = self.runner.invoke(app, ["context", "Add audit logging"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "Context placeholder for: Add audit logging",
            result.output,
        )


if __name__ == "__main__":
    unittest.main()
