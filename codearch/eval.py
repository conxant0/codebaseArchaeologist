import csv
import json
from pathlib import Path
from typing import Callable


def load_evaluation_file(evaluation_file: str | Path) -> list[dict]:
    path = Path(evaluation_file)
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Evaluation file must contain a JSON array.")

    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Evaluation case {index} must be an object.")
        for field in ("repo", "question", "expected_sources"):
            if field not in case:
                raise ValueError(f"Evaluation case {index} is missing '{field}'.")
        if not isinstance(case["expected_sources"], list):
            raise ValueError(
                f"Evaluation case {index} field 'expected_sources' must be a list."
            )

    return cases


def evaluate_question(case: dict, retrieved_artifacts: list[dict]) -> dict:
    expected_sources = case["expected_sources"]
    retrieved_sources = _unique_sources(retrieved_artifacts)
    hits = [source for source in expected_sources if source in retrieved_sources]
    misses = [source for source in expected_sources if source not in retrieved_sources]
    recall = calculate_recall(expected_sources, retrieved_sources)

    return {
        "question": case["question"],
        "expected_sources": expected_sources,
        "retrieved_sources": retrieved_sources,
        "hits": hits,
        "misses": misses,
        "recall": recall,
        "status": _status_for_recall(recall),
    }


def calculate_recall(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    if not expected_sources:
        return 0.0

    hits = sum(1 for source in expected_sources if source in retrieved_sources)
    return hits / len(expected_sources)


def print_evaluation_summary(
    results: list[dict],
    output: Callable[[str], None] = print,
):
    question_count = len(results)
    average_recall = (
        sum(result["recall"] for result in results) / question_count
        if question_count
        else 0.0
    )
    perfect_retrievals = sum(1 for result in results if result["status"] == "PASS")
    partial_retrievals = sum(1 for result in results if result["status"] == "PARTIAL")
    missed_retrievals = sum(1 for result in results if result["status"] == "MISS")

    output("=" * 50)
    output("Evaluation Results")
    output("=" * 50)
    output("")
    output(f"Questions Evaluated: {question_count}")
    output(f"Average Recall@K: {average_recall:.2f}")
    output(f"Perfect Retrievals: {perfect_retrievals}")
    output(f"Partial Retrievals: {partial_retrievals}")
    output(f"Missed Retrievals: {missed_retrievals}")
    output("")
    output("=" * 50)

    for result in results:
        output("")
        output("Question:")
        output(result["question"])
        output("")
        output(f"Recall@K: {_format_recall(result['recall'])}")
        output("")
        output("Expected:")
        for source in result["expected_sources"]:
            output(f"- {source}")
        output("")
        output("Retrieved:")
        for source in result["retrieved_sources"]:
            output(f"- {source}")
        output("")
        output("Status:")
        output(result["status"])
        output("")
        output("=" * 50)


def write_evaluation_csv(results: list[dict], csv_path: str | Path):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["question", "recall", "status", "expected", "retrieved"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "question": result["question"],
                    "recall": result["recall"],
                    "status": result["status"],
                    "expected": "; ".join(result["expected_sources"]),
                    "retrieved": "; ".join(result["retrieved_sources"]),
                }
            )


def _unique_sources(artifacts: list[dict]) -> list[str]:
    sources = []
    for artifact in artifacts:
        source = artifact.get("source")
        if source and source not in sources:
            sources.append(source)

    return sources


def _status_for_recall(recall: float) -> str:
    if recall == 1.0:
        return "PASS"
    if recall == 0.0:
        return "MISS"
    return "PARTIAL"


def _format_recall(recall: float) -> str:
    return f"{recall:.2f}".rstrip("0").rstrip(".") if recall % 1 else f"{recall:.1f}"
