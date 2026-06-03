import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import ANY
from unittest.mock import patch

from typer.testing import CliRunner

import codearch.cli as cli
from codearch.cli import app
from codearch.generate import GroqConfigurationError
from codearch.ingest_github import IngestResult
from codearch.retrieve import IndexNotFoundError


class CliTest(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_cli_configures_clean_terminal_output(self):
        cli._configure_terminal_output()
        self.assertEqual(os.environ["TOKENIZERS_PARALLELISM"], "false")

        recorded_warnings = []
        with warnings.catch_warnings():
            warnings.showwarning = (
                lambda *args, **kwargs: recorded_warnings.append(args)
            )
            warnings.warn(
                "`clean_up_tokenization_spaces` was not set. "
                "It will be set to `True` by default.",
                FutureWarning,
            )

        self.assertEqual(recorded_warnings, [])

    @patch("codearch.cli.github_token_missing")
    @patch("codearch.cli.build_index")
    @patch("codearch.cli.normalize_repo")
    @patch("codearch.cli.ingest_repo")
    def test_index_ingests_normalizes_and_indexes_repo(
        self,
        mock_ingest_repo,
        mock_normalize_repo,
        mock_build_index,
        mock_github_token_missing,
    ):
        mock_github_token_missing.return_value = True
        mock_ingest_repo.return_value = IngestResult(
            output_dir="/tmp/raw",
            mode="standard",
            fetched_counts={
                "issues": 1,
                "pull_requests": 1,
                "commits": 1,
                "docs": 1,
            },
            warnings=["Some commit history may not be indexed in standard mode."],
        )
        mock_normalize_repo.return_value = [
            {"id": "issue_1", "metadata": {"type": "issue"}},
            {"id": "pr_1", "metadata": {"type": "pull_request"}},
            {"id": "commit_abc1234", "metadata": {"type": "commit"}},
            {"id": "doc_readme", "metadata": {"type": "documentation"}},
        ]
        mock_build_index.return_value = 1

        with patch("codearch.cli._commit_staged_index") as mock_commit_staged_index:
            result = self.runner.invoke(
                app,
                ["index", "https://github.com/example/repo"],
            )

        self.assertEqual(result.exit_code, 0)
        mock_ingest_repo.assert_called_once_with(
            "https://github.com/example/repo",
            mode="standard",
            raw_data_dir=ANY,
        )
        mock_normalize_repo.assert_called_once_with(
            "example",
            "repo",
            mode="standard",
            fetched_counts={
                "issues": 1,
                "pull_requests": 1,
                "commits": 1,
                "docs": 1,
            },
            raw_data_dir=ANY,
            processed_data_dir=ANY,
            chroma_data_dir=ANY,
            manifest_raw_data_path=ANY,
            manifest_processed_artifacts_path=ANY,
            manifest_chroma_path=ANY,
        )
        mock_build_index.assert_called_once_with(
            "example",
            "repo",
            processed_data_dir=ANY,
            chroma_data_dir=ANY,
        )
        mock_commit_staged_index.assert_called_once()
        self.assertIn("Indexing example/repo", result.output)
        self.assertIn("Mode: standard", result.output)
        self.assertIn("This will fetch:", result.output)
        self.assertIn("- Issues: all", result.output)
        self.assertIn("- Pull Requests: all", result.output)
        self.assertIn("- Documentation: yes", result.output)
        self.assertIn("- Commits: selected", result.output)
        self.assertIn(
            "Building a local historical knowledge base. "
            "This may take several minutes.",
            result.output,
        )
        self.assertIn(
            "GitHub token not found. GitHub API rate limits may be lower.",
            result.output,
        )
        self.assertIn("Indexed example/repo using mode: standard", result.output)
        self.assertIn("Fetched:", result.output)
        self.assertIn("- Issues: 1", result.output)
        self.assertIn("- Pull Requests: 1", result.output)
        self.assertIn("- Commits: 1", result.output)
        self.assertIn("- Docs: 1", result.output)
        self.assertIn("Indexed artifacts:", result.output)
        self.assertIn("- Documentation: 1", result.output)
        self.assertIn("Storage:", result.output)
        self.assertIn("- Raw data: data/raw/example_repo/", result.output)
        self.assertIn(
            "- Processed artifacts: data/processed/example_repo/artifacts.jsonl",
            result.output,
        )
        self.assertIn(
            "- Manifest: data/processed/example_repo/manifest.json",
            result.output,
        )
        self.assertIn("- Chroma index: data/chroma/example_repo/", result.output)
        self.assertIn("Warnings:", result.output)
        self.assertIn(
            "- Some commit history may not be indexed in standard mode.",
            result.output,
        )

    @patch("codearch.cli.github_token_missing")
    @patch("codearch.cli.build_index")
    @patch("codearch.cli.normalize_repo")
    @patch("codearch.cli.ingest_repo")
    def test_index_passes_explicit_mode(
        self,
        mock_ingest_repo,
        mock_normalize_repo,
        mock_build_index,
        mock_github_token_missing,
    ):
        mock_github_token_missing.return_value = False
        mock_ingest_repo.return_value = IngestResult(
            output_dir="/tmp/raw",
            mode="recent",
            fetched_counts={
                "issues": 0,
                "pull_requests": 0,
                "commits": 0,
                "docs": 0,
            },
            warnings=[],
        )
        mock_normalize_repo.return_value = []

        with patch("codearch.cli._commit_staged_index") as mock_commit_staged_index:
            result = self.runner.invoke(
                app,
                ["index", "https://github.com/example/repo", "--mode", "recent"],
            )

        self.assertEqual(result.exit_code, 0)
        mock_ingest_repo.assert_called_once_with(
            "https://github.com/example/repo",
            mode="recent",
            raw_data_dir=ANY,
        )
        mock_normalize_repo.assert_called_once_with(
            "example",
            "repo",
            mode="recent",
            fetched_counts={
                "issues": 0,
                "pull_requests": 0,
                "commits": 0,
                "docs": 0,
            },
            raw_data_dir=ANY,
            processed_data_dir=ANY,
            chroma_data_dir=ANY,
            manifest_raw_data_path=ANY,
            manifest_processed_artifacts_path=ANY,
            manifest_chroma_path=ANY,
        )
        mock_build_index.assert_called_once_with(
            "example",
            "repo",
            processed_data_dir=ANY,
            chroma_data_dir=ANY,
        )
        mock_commit_staged_index.assert_called_once()
        self.assertIn("Mode: recent", result.output)
        self.assertIn("- Issues: recent", result.output)
        self.assertIn("- Pull Requests: recent", result.output)
        self.assertIn("- Documentation: no", result.output)
        self.assertIn("- Commits: recent", result.output)
        self.assertIn("Indexed example/repo using mode: recent", result.output)

    def test_commit_staged_index_swaps_staged_directories_into_live_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            live_raw = data_dir / "raw"
            live_processed = data_dir / "processed"
            live_chroma = data_dir / "chroma"
            staging_dir = data_dir / ".staging"
            staging_root = staging_dir / "owner_repo.stage"

            for base_dir, file_name, content in (
                (live_raw, "issues.json", "old raw"),
                (live_processed, "artifacts.jsonl", "old processed"),
                (live_chroma, "chroma.sqlite3", "old chroma"),
            ):
                repo_dir = base_dir / "owner_repo"
                repo_dir.mkdir(parents=True)
                (repo_dir / file_name).write_text(content, encoding="utf-8")

            for base_dir, file_name, content in (
                (staging_root / "raw", "issues.json", "new raw"),
                (staging_root / "processed", "artifacts.jsonl", "new processed"),
                (staging_root / "chroma", "chroma.sqlite3", "new chroma"),
            ):
                repo_dir = base_dir / "owner_repo"
                repo_dir.mkdir(parents=True)
                (repo_dir / file_name).write_text(content, encoding="utf-8")

            with (
                patch("codearch.cli.RAW_DATA_DIR", live_raw),
                patch("codearch.cli.PROCESSED_DATA_DIR", live_processed),
                patch("codearch.cli.CHROMA_DATA_DIR", live_chroma),
                patch("codearch.cli.STAGING_DATA_DIR", staging_dir),
            ):
                cli._commit_staged_index("owner", "repo", staging_root)

            self.assertEqual(
                (live_raw / "owner_repo" / "issues.json").read_text(
                    encoding="utf-8"
                ),
                "new raw",
            )
            self.assertEqual(
                (live_processed / "owner_repo" / "artifacts.jsonl").read_text(
                    encoding="utf-8"
                ),
                "new processed",
            )
            self.assertEqual(
                (live_chroma / "owner_repo" / "chroma.sqlite3").read_text(
                    encoding="utf-8"
                ),
                "new chroma",
            )
            self.assertFalse((staging_dir / "owner_repo.swap_backup").exists())

    def test_recover_interrupted_index_swap_restores_missing_live_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            live_raw = data_dir / "raw"
            live_processed = data_dir / "processed"
            live_chroma = data_dir / "chroma"
            staging_dir = data_dir / ".staging"
            backup_raw = staging_dir / "owner_repo.swap_backup" / "raw"
            backup_raw.mkdir(parents=True)
            (backup_raw / "issues.json").write_text("old raw", encoding="utf-8")
            (live_processed / "owner_repo").mkdir(parents=True)
            (live_chroma / "owner_repo").mkdir(parents=True)

            with (
                patch("codearch.cli.RAW_DATA_DIR", live_raw),
                patch("codearch.cli.PROCESSED_DATA_DIR", live_processed),
                patch("codearch.cli.CHROMA_DATA_DIR", live_chroma),
                patch("codearch.cli.STAGING_DATA_DIR", staging_dir),
            ):
                cli._recover_interrupted_index_swap("owner", "repo")

            self.assertEqual(
                (live_raw / "owner_repo" / "issues.json").read_text(
                    encoding="utf-8"
                ),
                "old raw",
            )
            self.assertFalse((staging_dir / "owner_repo.swap_backup").exists())

    def test_commit_staged_index_restores_backup_if_staged_rename_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            live_raw = data_dir / "raw"
            live_processed = data_dir / "processed"
            live_chroma = data_dir / "chroma"
            staging_dir = data_dir / ".staging"
            staging_root = staging_dir / "owner_repo.stage"

            for base_dir, file_name, content in (
                (live_raw, "issues.json", "old raw"),
                (live_processed, "artifacts.jsonl", "old processed"),
                (live_chroma, "chroma.sqlite3", "old chroma"),
            ):
                repo_dir = base_dir / "owner_repo"
                repo_dir.mkdir(parents=True)
                (repo_dir / file_name).write_text(content, encoding="utf-8")

            for base_dir, file_name, content in (
                (staging_root / "raw", "issues.json", "new raw"),
                (staging_root / "processed", "artifacts.jsonl", "new processed"),
                (staging_root / "chroma", "chroma.sqlite3", "new chroma"),
            ):
                repo_dir = base_dir / "owner_repo"
                repo_dir.mkdir(parents=True)
                (repo_dir / file_name).write_text(content, encoding="utf-8")

            original_rename = Path.rename

            def fail_staged_raw_rename(path, target):
                if path == staging_root / "raw" / "owner_repo":
                    raise OSError("staged rename failed")
                return original_rename(path, target)

            with (
                patch("codearch.cli.RAW_DATA_DIR", live_raw),
                patch("codearch.cli.PROCESSED_DATA_DIR", live_processed),
                patch("codearch.cli.CHROMA_DATA_DIR", live_chroma),
                patch("codearch.cli.STAGING_DATA_DIR", staging_dir),
                patch.object(Path, "rename", fail_staged_raw_rename),
            ):
                with self.assertRaisesRegex(OSError, "staged rename failed"):
                    cli._commit_staged_index("owner", "repo", staging_root)

            self.assertEqual(
                (live_raw / "owner_repo" / "issues.json").read_text(
                    encoding="utf-8"
                ),
                "old raw",
            )
            self.assertEqual(
                (live_processed / "owner_repo" / "artifacts.jsonl").read_text(
                    encoding="utf-8"
                ),
                "old processed",
            )
            self.assertEqual(
                (live_chroma / "owner_repo" / "chroma.sqlite3").read_text(
                    encoding="utf-8"
                ),
                "old chroma",
            )
            self.assertFalse((staging_dir / "owner_repo.swap_backup").exists())

    @patch("codearch.cli.answer_question")
    @patch("codearch.cli.retrieve_relevant_artifacts")
    def test_ask_retrieves_artifacts_for_repo(
        self,
        mock_retrieve_relevant_artifacts,
        mock_answer_question,
    ):
        mock_retrieve_relevant_artifacts.return_value = [
            {
                "id": "pr_1",
                "source": "Pull Request #1",
                "text": "Title: Add dependency injection",
                "distance": 0.123,
                "metadata": {"type": "pull_request"},
            },
        ]
        mock_answer_question.return_value = (
            "Dependency injection was added to improve testing (Pull Request #1)."
        )

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
        mock_answer_question.assert_called_once_with(
            "Why was dependency injection added?",
            mock_retrieve_relevant_artifacts.return_value,
        )
        self.assertIn(
            "Dependency injection was added to improve testing (Pull Request #1).",
            result.output,
        )
        self.assertIn("Question:", result.output)
        self.assertIn("Why was dependency injection added?", result.output)
        self.assertIn("Historical Findings", result.output)
        self.assertIn("Sources:", result.output)
        self.assertIn("- Pull Request #1", result.output)
        self.assertNotIn("Top retrieved artifacts:", result.output)

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

    @patch("codearch.cli.answer_question")
    @patch("codearch.cli.retrieve_relevant_artifacts")
    def test_ask_warns_when_top_result_is_weak(
        self,
        mock_retrieve_relevant_artifacts,
        mock_answer_question,
    ):
        mock_retrieve_relevant_artifacts.return_value = [
            {
                "id": "issue_1",
                "source": "Issue #1",
                "text": "Title: Unrelated issue",
                "distance": 1.31,
                "metadata": {"type": "issue"},
            },
        ]
        mock_answer_question.return_value = "The context is insufficient."

        result = self.runner.invoke(
            app,
            ["ask", "Why was dependency injection added?", "--repo", "fastapi/fastapi"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "Warning: Retrieved artifacts are related, but may not contain a direct "
            "historical explanation.",
            result.output,
        )
        self.assertIn("The context is insufficient.", result.output)

    @patch("codearch.cli.answer_question")
    @patch("codearch.cli.retrieve_relevant_artifacts")
    def test_ask_warns_when_all_retrieved_artifacts_are_commits(
        self,
        mock_retrieve_relevant_artifacts,
        mock_answer_question,
    ):
        mock_retrieve_relevant_artifacts.return_value = [
            {
                "id": "commit_abc1234",
                "source": "Commit abc1234",
                "text": "Message: Add authentication helper",
                "distance": 0.42,
                "metadata": {"type": "commit"},
            },
            {
                "id": "commit_def5678",
                "source": "Commit def5678",
                "text": "Message: Refactor auth token parsing",
                "distance": 0.51,
                "metadata": {"type": "commit"},
            },
        ]
        mock_answer_question.return_value = "No direct reason is available."

        result = self.runner.invoke(
            app,
            ["ask", "Why was there no user authentication?", "--repo", "fastapi/fastapi"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "Warning: Retrieved artifacts are related, but may not contain a direct "
            "historical explanation.",
            result.output,
        )
        self.assertIn("No direct reason is available.", result.output)

    @patch("codearch.cli.answer_question")
    @patch("codearch.cli.retrieve_relevant_artifacts")
    def test_ask_does_not_warn_at_weak_threshold(
        self,
        mock_retrieve_relevant_artifacts,
        mock_answer_question,
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
        mock_answer_question.return_value = "Issue #1 explains the change."

        result = self.runner.invoke(
            app,
            ["ask", "Why was dependency injection added?", "--repo", "fastapi/fastapi"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("Warning:", result.output)

    @patch("codearch.cli.answer_question")
    @patch("codearch.cli.retrieve_relevant_artifacts")
    def test_ask_falls_back_to_retrieved_sources_when_answer_has_no_citations(
        self,
        mock_retrieve_relevant_artifacts,
        mock_answer_question,
    ):
        mock_retrieve_relevant_artifacts.return_value = [
            {
                "id": "issue_1",
                "source": "Issue #1",
                "text": "Title: Related issue",
                "distance": 0.3,
                "metadata": {"type": "issue"},
            },
            {
                "id": "issue_1_duplicate",
                "source": "Issue #1",
                "text": "Body: Duplicate context",
                "distance": 0.4,
                "metadata": {"type": "issue"},
            },
        ]
        mock_answer_question.return_value = "The context is insufficient."

        result = self.runner.invoke(
            app,
            ["ask", "Why was dependency injection added?", "--repo", "fastapi/fastapi"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Sources:", result.output)
        self.assertEqual(result.output.count("- Issue #1"), 1)

    @patch("codearch.cli.answer_question")
    @patch("codearch.cli.retrieve_relevant_artifacts")
    def test_ask_prints_groq_configuration_error(
        self,
        mock_retrieve_relevant_artifacts,
        mock_answer_question,
    ):
        mock_retrieve_relevant_artifacts.return_value = [
            {
                "id": "issue_1",
                "source": "Issue #1",
                "text": "Title: Related issue",
                "distance": 0.3,
                "metadata": {"type": "issue"},
            },
        ]
        mock_answer_question.side_effect = GroqConfigurationError(
            "GROQ_API_KEY is not set. Set it in your environment or .env file."
        )

        result = self.runner.invoke(
            app,
            ["ask", "Why was dependency injection added?", "--repo", "fastapi/fastapi"],
        )

        self.assertEqual(result.exit_code, 1)
        self.assertIn(
            "GROQ_API_KEY is not set. Set it in your environment or .env file.",
            result.output,
        )
        self.assertNotIn("Traceback", result.output)

    @patch("codearch.cli.generate_context_pack")
    @patch("codearch.cli.retrieve_relevant_artifacts")
    def test_context_generates_context_pack_for_repo(
        self,
        mock_retrieve_relevant_artifacts,
        mock_generate_context_pack,
    ):
        mock_retrieve_relevant_artifacts.return_value = [
            {
                "id": "pr_3669",
                "source": "Pull Request #3669",
                "text": "Title: Join dependency execution paths",
                "distance": 0.86,
                "metadata": {"type": "pull_request"},
            },
            {
                "id": "issue_1105",
                "source": "Issue #1105",
                "text": "Title: Using Depends outside routes",
                "distance": 0.94,
                "metadata": {"type": "issue"},
            },
        ]
        mock_generate_context_pack.return_value = (
            "RELEVANT HISTORICAL EVIDENCE\n\n"
            "Pull Request #3669\n"
            "Distance: 0.86\n\n"
            "Summary:\nDependency execution was centralized.\n\n"
            "CONSTRAINTS\n- Preserve execution/concurrency separation."
        )

        result = self.runner.invoke(
            app,
            [
                "context",
                "Modify dependency injection behavior",
                "--repo",
                "fastapi/fastapi",
            ],
        )

        self.assertEqual(result.exit_code, 0)
        mock_retrieve_relevant_artifacts.assert_called_once_with(
            "fastapi",
            "fastapi",
            "Modify dependency injection behavior",
        )
        mock_generate_context_pack.assert_called_once_with(
            "Modify dependency injection behavior",
            mock_retrieve_relevant_artifacts.return_value,
        )
        self.assertIn("=" * 80, result.output)
        self.assertIn("CHANGE REQUEST", result.output)
        self.assertIn("Modify dependency injection behavior", result.output)
        self.assertIn("RELEVANT HISTORICAL EVIDENCE", result.output)
        self.assertIn("CONSTRAINTS", result.output)
        self.assertIn("SOURCES", result.output)
        self.assertIn("- Pull Request #3669", result.output)
        self.assertIn("- Issue #1105", result.output)
        self.assertNotIn("Context placeholder", result.output)

    @patch("codearch.cli.generate_context_pack")
    @patch("codearch.cli.retrieve_relevant_artifacts")
    def test_context_prints_groq_configuration_error(
        self,
        mock_retrieve_relevant_artifacts,
        mock_generate_context_pack,
    ):
        mock_retrieve_relevant_artifacts.return_value = [
            {
                "id": "issue_1",
                "source": "Issue #1",
                "text": "Title: Related issue",
                "distance": 0.3,
                "metadata": {"type": "issue"},
            },
        ]
        mock_generate_context_pack.side_effect = GroqConfigurationError(
            "GROQ_API_KEY is not set. Set it in your environment or .env file."
        )

        result = self.runner.invoke(
            app,
            ["context", "Modify dependency injection behavior", "--repo", "fastapi/fastapi"],
        )

        self.assertEqual(result.exit_code, 1)
        self.assertIn(
            "GROQ_API_KEY is not set. Set it in your environment or .env file.",
            result.output,
        )
        self.assertNotIn("Traceback", result.output)


if __name__ == "__main__":
    unittest.main()
