import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from codearch.ingest_github import (
    fetch_commits,
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
            params={"per_page": 7, "state": "all"},
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
            result = fetch_issues("owner", "repo")

        self.assertEqual(result, [{"number": 1, "title": "Real issue"}])
        mock_load_dotenv.assert_called_once()

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
            params={"per_page": 3, "state": "all"},
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
            params={"per_page": 5},
            timeout=30,
        )

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

    @patch("codearch.ingest_github.fetch_commits")
    @patch("codearch.ingest_github.fetch_pull_requests")
    @patch("codearch.ingest_github.fetch_issues")
    def test_ingest_repo_saves_raw_json_files(
        self,
        mock_fetch_issues,
        mock_fetch_pull_requests,
        mock_fetch_commits,
    ):
        mock_fetch_issues.return_value = [{"number": 1}]
        mock_fetch_pull_requests.return_value = [{"number": 2}]
        mock_fetch_commits.return_value = [{"sha": "abc123"}]

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("codearch.ingest_github.RAW_DATA_DIR", Path(temp_dir)):
                output_dir = ingest_repo("https://github.com/owner/repo")

            self.assertEqual(output_dir, Path(temp_dir) / "owner_repo")
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


if __name__ == "__main__":
    unittest.main()
