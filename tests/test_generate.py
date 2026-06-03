import os
import sys
import types
import unittest
from unittest.mock import patch

from codearch.generate import GroqConfigurationError
from codearch.generate import answer_question
from codearch.generate import compute_evidence_strength
from codearch.generate import generate_context_pack


class FakeGroq:
    api_key = None
    create_args = None

    def __init__(self, api_key):
        FakeGroq.api_key = api_key
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        FakeGroq.create_args = kwargs
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content="Dependency injection was added for tests (Issue #123)."
                    )
                )
            ]
        )


class GenerateTest(unittest.TestCase):
    def test_answer_question_formats_context_and_calls_groq(self):
        fake_groq_module = types.SimpleNamespace(Groq=FakeGroq)
        artifacts = [
            {
                "source": "Issue #123",
                "text": "Title: Add dependency injection\nBody: Needed for tests",
                "distance": 0.42,
                "metadata": {
                    "type": "issue",
                    "url": "https://github.com/owner/repo/issues/123",
                },
            },
        ]

        with (
            patch.dict(sys.modules, {"groq": fake_groq_module}),
            patch.dict(
                os.environ,
                {
                    "GROQ_API_KEY": "test-key",
                    "GROQ_MODEL": "test-model",
                },
            ),
        ):
            answer = answer_question(
                "Why was dependency injection added?",
                artifacts,
                evidence_strength="strong",
            )

        self.assertEqual(
            answer,
            "Dependency injection was added for tests (Issue #123).",
        )
        self.assertEqual(FakeGroq.api_key, "test-key")
        self.assertEqual(FakeGroq.create_args["model"], "test-model")
        self.assertEqual(FakeGroq.create_args["temperature"], 0)
        self.assertEqual(FakeGroq.create_args["max_tokens"], 500)
        self.assertNotIn("max_completion_tokens", FakeGroq.create_args)
        user_prompt = FakeGroq.create_args["messages"][1]["content"]
        self.assertIn("[Source: Issue #123]", user_prompt)
        self.assertIn("Type: issue", user_prompt)
        self.assertIn("Distance: 0.42", user_prompt)
        self.assertIn(
            "URL: https://github.com/owner/repo/issues/123",
            user_prompt,
        )
        self.assertIn(
            "Content:\nTitle: Add dependency injection\nBody: Needed for tests",
            user_prompt,
        )
        self.assertIn("using only the provided context", user_prompt)
        self.assertIn("cite sources by source name", user_prompt)
        self.assertIn("Historical Findings", user_prompt)
        self.assertIn("Avoid filler phrases", user_prompt)
        self.assertIn("Based on the provided context", user_prompt)
        self.assertIn("Evidence Strength: strong", user_prompt)
        self.assertIn("Use this exact Evidence Strength value.", user_prompt)
        self.assertIn("Do not recalculate it.", user_prompt)
        self.assertIn("Do not change it.", user_prompt)
        self.assertNotIn("Evidence Strength labels:", user_prompt)
        self.assertNotIn("Evidence Strength: <strong|partial|weak>", user_prompt)

    def test_answer_question_prompt_guards_against_unsupported_original_cause(self):
        fake_groq_module = types.SimpleNamespace(Groq=FakeGroq)
        artifacts = [
            {
                "source": "Pull Request #9479",
                "text": "Title: Fix parameterless Depends() with generics",
                "distance": 0.93,
                "metadata": {
                    "type": "pull_request",
                    "url": "https://github.com/owner/repo/pull/9479",
                },
            },
        ]

        with (
            patch.dict(sys.modules, {"groq": fake_groq_module}),
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True),
        ):
            answer_question(
                "Why was dependency injection added?",
                artifacts,
                evidence_strength="partial",
            )

        user_prompt = FakeGroq.create_args["messages"][1]["content"]
        self.assertIn("Evidence Strength: partial", user_prompt)
        self.assertNotIn("Evidence Strength labels:", user_prompt)
        self.assertNotIn("Evidence Strength: <strong|partial|weak>", user_prompt)
        self.assertIn(
            "The retrieved evidence does not directly show why X was originally added.",
            user_prompt,
        )
        self.assertIn(
            "Do not say 'X was added because' unless a retrieved artifact directly "
            "supports that causal claim.",
            user_prompt,
        )

    def test_generate_context_pack_uses_change_context_prompt(self):
        fake_groq_module = types.SimpleNamespace(Groq=FakeGroq)
        artifacts = [
            {
                "source": "Pull Request #3669",
                "text": "Title: Join dependency execution paths",
                "distance": 0.86,
                "metadata": {
                    "type": "pull_request",
                    "url": "https://github.com/owner/repo/pull/3669",
                },
            },
        ]

        with (
            patch.dict(sys.modules, {"groq": fake_groq_module}),
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True),
        ):
            generate_context_pack(
                "Modify dependency injection behavior",
                artifacts,
                evidence_strength="strong",
            )

        user_prompt = FakeGroq.create_args["messages"][1]["content"]
        system_prompt = FakeGroq.create_args["messages"][0]["content"]
        self.assertIn("context pack", user_prompt)
        self.assertIn("# Codebase Archaeologist Context Pack", user_prompt)
        self.assertIn("## Change Request", user_prompt)
        self.assertIn("## Evidence Strength", user_prompt)
        self.assertIn("strong\n\n", user_prompt)
        self.assertNotIn("Choose one: strong | partial | weak", user_prompt)
        self.assertNotIn("Rules:", user_prompt)
        self.assertNotIn("strong: retrieved artifacts directly explain", user_prompt)
        self.assertNotIn("partial: retrieved artifacts are related", user_prompt)
        self.assertNotIn("weak: retrieved artifacts are only loosely related", user_prompt)
        self.assertIn("## Retrieved Findings", user_prompt)
        self.assertIn("### <artifact>", user_prompt)
        self.assertIn("Finding:", user_prompt)
        self.assertIn("Evidence:", user_prompt)
        self.assertIn("Why Relevant:", user_prompt)
        self.assertIn("## Supported Conclusions", user_prompt)
        self.assertIn("## Open Questions", user_prompt)
        self.assertIn("## Sources", user_prompt)
        self.assertNotIn("## Historical Context", user_prompt)
        self.assertNotIn("## Architectural Constraints", user_prompt)
        self.assertNotIn("## Risks Before Editing", user_prompt)
        self.assertIn("Do not generate code", user_prompt)
        self.assertIn(
            "Do not generate risks, constraints, or recommendations unless "
            "explicitly present",
            user_prompt,
        )
        self.assertIn("prepare context before editing", user_prompt)
        self.assertIn("If retrieved artifacts do not directly support a claim", user_prompt)
        self.assertIn("evidence-first", user_prompt)
        self.assertIn("extracted findings", user_prompt)
        self.assertIn("cited artifacts", user_prompt)
        self.assertIn("historical evidence report", user_prompt)
        self.assertIn("Prefer extraction over summarization", user_prompt)
        self.assertIn("Treat uncertainty as valuable information", user_prompt)
        self.assertIn("Every major statement must be traceable", user_prompt)
        self.assertIn("Which retrieved artifact supports this claim?", user_prompt)
        self.assertIn(
            "Do not infer architectural intent unless directly supported",
            user_prompt,
        )
        self.assertIn("[Source: Pull Request #3669]", user_prompt)
        self.assertIn("Distance: 0.86", user_prompt)
        self.assertIn("historian and researcher", system_prompt)

    def test_compute_evidence_strength_uses_average_distance(self):
        self.assertEqual(
            compute_evidence_strength(
                [
                    {"distance": 0.85},
                    {"distance": 0.89},
                ]
            ),
            "strong",
        )
        self.assertEqual(
            compute_evidence_strength(
                [
                    {"distance": 0.9},
                    {"distance": 1.0},
                    {"distance": 1.1},
                ]
            ),
            "partial",
        )
        self.assertEqual(
            compute_evidence_strength(
                [
                    {"distance": 1.2},
                    {"distance": 1.4},
                ]
            ),
            "weak",
        )

    def test_compute_evidence_strength_returns_weak_without_distances(self):
        self.assertEqual(compute_evidence_strength([]), "weak")
        self.assertEqual(compute_evidence_strength([{}]), "weak")

    def test_compute_evidence_strength_uses_top_three_distances(self):
        self.assertEqual(
            compute_evidence_strength(
                [
                    {"distance": 0.8},
                    {"distance": 0.9},
                    {"distance": 1.0},
                    {"distance": 2.0},
                ]
            ),
            "partial",
        )

    def test_answer_question_uses_default_model(self):
        fake_groq_module = types.SimpleNamespace(Groq=FakeGroq)

        with (
            patch.dict(sys.modules, {"groq": fake_groq_module}),
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True),
        ):
            answer_question("Question?", [], evidence_strength="weak")

        self.assertEqual(FakeGroq.create_args["model"], "llama-3.1-8b-instant")

    def test_answer_question_raises_configuration_error_when_api_key_is_missing(self):
        fake_groq_module = types.SimpleNamespace(Groq=FakeGroq)

        with (
            patch.dict(sys.modules, {"groq": fake_groq_module}),
            patch("codearch.generate.load_dotenv"),
            patch.dict(os.environ, {}, clear=True),
        ):
            with self.assertRaisesRegex(
                GroqConfigurationError,
                "GROQ_API_KEY is not set",
            ):
                answer_question("Question?", [], evidence_strength="weak")


if __name__ == "__main__":
    unittest.main()
