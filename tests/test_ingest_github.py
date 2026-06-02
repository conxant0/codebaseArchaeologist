import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from codearch.ingest_github import (
    _github_get,
    _print_spinner_progress,
    fetch_all_issues,
    fetch_commits,
    fetch_docs,
    fetch_issues,
    fetch_pull_requests,
    ingest_repo,
    parse_github_url,
)


class GitHubIngestionTest(unittest.TestCase):
    def test_parse_github_url_returns_owner_and_repo(self):
        self.assertEqual(
            parse_github_url("https://github.com/example-owner/example-repo"),
            ("example-owner", "example-repo"),
        )

    def test_parse_github_url_rejects_non_github_url(self):
        with self.assertRaises(ValueError):
            parse_github_url("https://gitlab.com/example-owner/example-repo")

    @patch("codearch.ingest_github.requests.get")
    @patch("codearch.ingest_github.load_dotenv")
    def test_fetch_issues_loads_dotenv_and_uses_optional_token(
        self,
        mock_load_dotenv,
        mock_get,
    ):
        mock_response = Mock()
        mock_response.json.return_value = [{"number": 1}]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {"GITHUB_TOKEN": "secret-token"}):
            result = fetch_issues("owner", "repo", limit=7)

        self.assertEqual(result, [{"number": 1}])
        mock_load_dotenv.assert_called_once()
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/issues",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": "Bearer secret-token",
            },
            params={
                "state": "all",
                "sort": "updated",
                "direction": "desc",
                "per_page": 7,
                "page": 1,
            },
            timeout=30,
        )

    @patch("codearch.ingest_github.requests.get")
    @patch("codearch.ingest_github.load_dotenv")
    def test_fetch_issues_filters_pull_request_records(
        self,
        mock_load_dotenv,
        mock_get,
    ):
        mock_response = Mock()
        mock_response.json.return_value = [
            {"number": 1, "title": "Real issue"},
            {
                "number": 2,
                "title": "Pull request returned by issues endpoint",
                "pull_request": {"url": "https://api.github.com/repos/o/r/pulls/2"},
            },
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {}, clear=True):
            with patch("builtins.print") as mock_print:
                result = fetch_issues("owner", "repo")

        self.assertEqual(result, [{"number": 1, "title": "Real issue"}])
        mock_load_dotenv.assert_called_once()
        mock_print.assert_any_call("Issue endpoint summary:")
        mock_print.assert_any_call("- Raw issue endpoint items fetched: 2")
        mock_print.assert_any_call("- Actual issues after filtering PRs: 1")
        mock_print.assert_any_call("- Pull requests skipped from issues endpoint: 1")

    @patch("codearch.ingest_github.requests.get")
    @patch("codearch.ingest_github.load_dotenv")
    def test_fetch_pull_requests_uses_pulls_endpoint(self, mock_load_dotenv, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = [{"number": 2}]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {}, clear=True):
            result = fetch_pull_requests("owner", "repo", limit=3)

        self.assertEqual(result, [{"number": 2}])
        mock_load_dotenv.assert_called_once()
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/pulls",
            headers={"Accept": "application/vnd.github+json"},
            params={"state": "all", "per_page": 3, "page": 1},
            timeout=30,
        )

    @patch("codearch.ingest_github.requests.get")
    @patch("codearch.ingest_github.load_dotenv")
    def test_fetch_commits_uses_commits_endpoint(self, mock_load_dotenv, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = [{"sha": "abc123"}]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {}, clear=True):
            result = fetch_commits("owner", "repo", limit=5)

        self.assertEqual(result, [{"sha": "abc123"}])
        mock_load_dotenv.assert_called_once()
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/commits",
            headers={"Accept": "application/vnd.github+json"},
            params={"per_page": 5, "page": 1},
            timeout=30,
        )

    @patch("codearch.ingest_github.requests.get")
    @patch("codearch.ingest_github.load_dotenv")
    def test_fetch_all_issues_paginates_until_short_page(
        self,
        mock_load_dotenv,
        mock_get,
    ):
        first_response = Mock()
        first_response.json.return_value = [{"number": number} for number in range(100)]
        first_response.raise_for_status.return_value = None
        second_response = Mock()
        second_response.json.return_value = [{"number": 101}]
        second_response.raise_for_status.return_value = None
        mock_get.side_effect = [first_response, second_response]

        with patch.dict("os.environ", {}, clear=True):
            result = fetch_all_issues("owner", "repo")

        self.assertEqual(len(result), 101)
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(
            mock_get.call_args_list[0].kwargs["params"],
            {
                "state": "all",
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,
                "page": 1,
            },
        )
        self.assertEqual(
            mock_get.call_args_list[1].kwargs["params"],
            {
                "state": "all",
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,
                "page": 2,
            },
        )

    @patch("codearch.ingest_github.requests.get")
    @patch("codearch.ingest_github.load_dotenv")
    def test_fetch_issues_scans_bounded_recent_pages_when_pages_are_full(
        self,
        mock_load_dotenv,
        mock_get,
    ):
        first_response = Mock()
        first_response.json.return_value = [
            {"number": number, "pull_request": {}} for number in range(100)
        ]
        first_response.raise_for_status.return_value = None
        second_response = Mock()
        second_response.json.return_value = [
            {"number": 101, "title": "Recent non-PR issue"}
        ]
        second_response.raise_for_status.return_value = None
        mock_get.side_effect = [first_response, second_response]

        with patch.dict("os.environ", {}, clear=True):
            result = fetch_issues("owner", "repo")

        self.assertEqual(result, [{"number": 101, "title": "Recent non-PR issue"}])
        self.assertEqual(mock_get.call_count, 2)

    @patch("codearch.ingest_github._github_get")
    def test_fetch_docs_fetches_markdown_documentation(self, mock_github_get):
        encoded_readme = "UmVhZG1lIGNvbnRlbnQ="
        encoded_doc = "RG9jIGNvbnRlbnQ="
        mock_github_get.side_effect = [
            {"default_branch": "main"},
            {
                "tree": [
                    {"type": "blob", "path": "README.md", "url": "readme-api"},
                    {"type": "blob", "path": "docs/guide.mdx", "url": "guide-api"},
                    {"type": "blob", "path": "src/app.py", "url": "python-api"},
                ]
            },
            {
                "html_url": "https://github.com/owner/repo/blob/main/README.md",
                "encoding": "base64",
                "content": encoded_readme,
            },
            {
                "html_url": "https://github.com/owner/repo/blob/main/docs/guide.mdx",
                "encoding": "base64",
                "content": encoded_doc,
            },
        ]

        docs = fetch_docs("owner", "repo")

        self.assertEqual(
            docs,
            [
                {
                    "path": "README.md",
                    "url": "https://github.com/owner/repo/blob/main/README.md",
                    "content": "Readme content",
                    "text": "Readme content",
                },
                {
                    "path": "docs/guide.mdx",
                    "url": "https://github.com/owner/repo/blob/main/docs/guide.mdx",
                    "content": "Doc content",
                    "text": "Doc content",
                },
            ],
        )

    @patch("codearch.ingest_github._github_get")
    def test_fetch_docs_standard_selects_english_docs_and_skips_low_value_paths(
        self,
        mock_github_get,
    ):
        encoded_readme = "UmVhZG1lIGNvbnRlbnQ="
        encoded_contributing = "Q29udHJpYnV0aW5nIGNvbnRlbnQ="
        encoded_english_doc = "RW5nbGlzaCBkb2MgY29udGVudA=="
        mock_github_get.side_effect = [
            {"default_branch": "main"},
            {
                "tree": [
                    {"type": "blob", "path": "README.md", "url": "readme-api"},
                    {
                        "type": "blob",
                        "path": "CONTRIBUTING.md",
                        "url": "contributing-api",
                    },
                    {"type": "blob", "path": "docs/en/tutorial.md", "url": "en-api"},
                    {"type": "blob", "path": "docs/fr/tutorial.md", "url": "fr-api"},
                    {
                        "type": "blob",
                        "path": "docs/en/generated/api.md",
                        "url": "generated-api",
                    },
                    {
                        "type": "blob",
                        "path": "docs/en/changelog/entry.md",
                        "url": "changelog-api",
                    },
                ]
            },
            {
                "html_url": "https://github.com/owner/repo/blob/main/README.md",
                "encoding": "base64",
                "content": encoded_readme,
            },
            {
                "html_url": "https://github.com/owner/repo/blob/main/CONTRIBUTING.md",
                "encoding": "base64",
                "content": encoded_contributing,
            },
            {
                "html_url": "https://github.com/owner/repo/blob/main/docs/en/tutorial.md",
                "encoding": "base64",
                "content": encoded_english_doc,
            },
        ]

        docs = fetch_docs("owner", "repo", mode="standard")

        self.assertEqual(
            [doc["path"] for doc in docs],
            ["README.md", "CONTRIBUTING.md", "docs/en/tutorial.md"],
        )

    @patch("codearch.ingest_github._github_get")
    def test_fetch_docs_standard_fallback_caps_docs_when_docs_en_is_missing(
        self,
        mock_github_get,
    ):
        doc_items = [
            {"type": "blob", "path": f"docs/page-{index}.md", "url": f"doc-{index}"}
            for index in range(301)
        ]
        content_records = [
            {
                "html_url": f"https://github.com/owner/repo/blob/main/docs/page-{index}.md",
                "encoding": "base64",
                "content": "RG9jIGNvbnRlbnQ=",
            }
            for index in range(300)
        ]
        mock_github_get.side_effect = [
            {"default_branch": "main"},
            {"tree": doc_items},
            *content_records,
        ]

        docs = fetch_docs("owner", "repo", mode="standard")

        self.assertEqual(len(docs), 300)
        self.assertEqual(docs[-1]["path"], "docs/page-299.md")

    @patch("codearch.ingest_github._github_get")
    def test_fetch_docs_skips_individual_doc_timeout(self, mock_github_get):
        encoded_doc = "RG9jIGNvbnRlbnQ="
        mock_github_get.side_effect = [
            {"default_branch": "main"},
            {
                "tree": [
                    {"type": "blob", "path": "README.md", "url": "readme-api"},
                    {"type": "blob", "path": "docs/guide.md", "url": "guide-api"},
                ]
            },
            requests.exceptions.Timeout("timed out"),
            {
                "html_url": "https://github.com/owner/repo/blob/main/docs/guide.md",
                "encoding": "base64",
                "content": encoded_doc,
            },
        ]

        with patch("builtins.print") as mock_print:
            docs = fetch_docs("owner", "repo", mode="standard")

        self.assertEqual([doc["path"] for doc in docs], ["docs/guide.md"])
        mock_print.assert_any_call("Skipped doc due to timeout: README.md")
        mock_print.assert_any_call("\r  docs: fetched 1/2, skipped 1")

    @patch("builtins.print")
    def test_print_spinner_progress_updates_same_terminal_line(self, mock_print):
        next_index = _print_spinner_progress("issues", "page 1, total: 100", 0)

        self.assertEqual(next_index, 1)
        mock_print.assert_called_once_with(
            "\r  | issues: page 1, total: 100",
            end="",
            flush=True,
        )

    @patch("codearch.ingest_github.time.sleep")
    @patch("codearch.ingest_github.requests.get")
    @patch("codearch.ingest_github.load_dotenv")
    def test_github_get_retries_timeouts(
        self,
        mock_load_dotenv,
        mock_get,
        mock_sleep,
    ):
        mock_response = Mock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status.return_value = None
        mock_get.side_effect = [
            requests.exceptions.Timeout("first timeout"),
            requests.exceptions.Timeout("second timeout"),
            mock_response,
        ]

        with patch.dict("os.environ", {}, clear=True):
            result = _github_get("/repos/owner/repo")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_get.call_count, 3)
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    @patch("codearch.ingest_github.requests.get")
    def test_fetch_issues_raises_clear_error_for_http_failures(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "404 Client Error"
        )
        mock_get.return_value = mock_response

        with self.assertRaisesRegex(
            RuntimeError,
            "GitHub API request failed \\(404\\): Not Found",
        ):
            fetch_issues("owner", "missing-repo")

    @patch("codearch.ingest_github.fetch_docs")
    @patch("codearch.ingest_github.fetch_selected_commits")
    @patch("codearch.ingest_github.fetch_all_pull_requests")
    @patch("codearch.ingest_github.fetch_all_issues")
    def test_ingest_repo_uses_standard_mode_by_default_and_saves_raw_json_files(
        self,
        mock_fetch_all_issues,
        mock_fetch_all_pull_requests,
        mock_fetch_selected_commits,
        mock_fetch_docs,
    ):
        mock_fetch_all_issues.return_value = [{"number": 1}]
        mock_fetch_all_pull_requests.return_value = [{"number": 2}]
        mock_fetch_selected_commits.return_value = [{"sha": "abc123"}]
        mock_fetch_docs.return_value = [{"path": "README.md"}]

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("codearch.ingest_github.RAW_DATA_DIR", Path(temp_dir)),
                patch.dict("os.environ", {"GITHUB_TOKEN": "token"}),
            ):
                result = ingest_repo("https://github.com/owner/repo")

            output_dir = result.output_dir
            self.assertEqual(output_dir, Path(temp_dir) / "owner_repo")
            self.assertEqual(result.mode, "standard")
            self.assertEqual(
                result.fetched_counts,
                {
                    "issues": 1,
                    "pull_requests": 1,
                    "commits": 1,
                    "docs": 1,
                },
            )
            self.assertEqual(
                json.loads((output_dir / "issues.json").read_text()),
                [{"number": 1}],
            )
            self.assertEqual(
                json.loads((output_dir / "pulls.json").read_text()),
                [{"number": 2}],
            )
            self.assertEqual(
                json.loads((output_dir / "commits.json").read_text()),
                [{"sha": "abc123"}],
            )
            self.assertEqual(
                json.loads((output_dir / "docs.json").read_text()),
                [{"path": "README.md"}],
            )

    @patch("codearch.ingest_github.fetch_commits")
    @patch("codearch.ingest_github.fetch_pull_requests")
    @patch("codearch.ingest_github.fetch_issues")
    def test_ingest_repo_recent_mode_skips_docs(
        self,
        mock_fetch_issues,
        mock_fetch_pull_requests,
        mock_fetch_commits,
    ):
        mock_fetch_issues.return_value = [{"number": 1}]
        mock_fetch_pull_requests.return_value = [{"number": 2}]
        mock_fetch_commits.return_value = [{"sha": "abc123"}]

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("codearch.ingest_github.RAW_DATA_DIR", Path(temp_dir)),
                patch.dict("os.environ", {"GITHUB_TOKEN": "token"}),
            ):
                result = ingest_repo("https://github.com/owner/repo", mode="recent")

            self.assertEqual(result.fetched_counts["docs"], 0)
            self.assertEqual(
                json.loads((result.output_dir / "docs.json").read_text()),
                [],
            )


if __name__ == "__main__":
    unittest.main()
