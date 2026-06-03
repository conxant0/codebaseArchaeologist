import json
import tempfile
import unittest
from pathlib import Path

from codearch.eval import calculate_recall
from codearch.eval import evaluate_question
from codearch.eval import load_evaluation_file


class EvalTest(unittest.TestCase):
    def test_load_evaluation_file_reads_json_array(self):
        cases = [
            {
                "repo": "sample/repo",
                "question": "Why was refreshSession added?",
                "expected_sources": ["Issue #184", "Pull Request #219"],
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            eval_path = Path(temp_dir) / "eval.json"
            eval_path.write_text(json.dumps(cases), encoding="utf-8")

            self.assertEqual(load_evaluation_file(eval_path), cases)

    def test_calculate_recall_counts_expected_sources_retrieved(self):
        recall = calculate_recall(
            ["Issue #184", "Pull Request #219"],
            ["Issue #184", "Commit 8f42c91"],
        )

        self.assertEqual(recall, 0.5)

    def test_evaluate_question_reports_hits_misses_and_status(self):
        case = {
            "repo": "sample/repo",
            "question": "Why was refreshSession added?",
            "expected_sources": ["Issue #184", "Pull Request #219"],
        }
        artifacts = [
            {"source": "Issue #184"},
            {"source": "Commit 8f42c91"},
            {"source": "Issue #184"},
        ]

        result = evaluate_question(case, artifacts)

        self.assertEqual(result["question"], "Why was refreshSession added?")
        self.assertEqual(result["expected_sources"], ["Issue #184", "Pull Request #219"])
        self.assertEqual(result["retrieved_sources"], ["Issue #184", "Commit 8f42c91"])
        self.assertEqual(result["hits"], ["Issue #184"])
        self.assertEqual(result["misses"], ["Pull Request #219"])
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["status"], "PARTIAL")


if __name__ == "__main__":
    unittest.main()
