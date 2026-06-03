import os

from dotenv import load_dotenv


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


class GroqConfigurationError(Exception):
    pass


def answer_question(question: str, artifacts: list[dict]) -> str:
    """Generate an answer from retrieved artifacts using Groq."""
    context = _format_artifacts_context(artifacts)
    prompt = (
        "Generate concise Historical Findings using only the provided context. "
        "If the context is insufficient, say so. "
        "cite sources by source name. "
        "Avoid filler phrases such as 'Based on the provided context', "
        "'It seems that', and 'You would need to'. "
        "Do not speculate or recommend implementation steps. "
        "Do not claim retrieved artifacts explain the original reason for a "
        "feature when they only show later documentation, refinements, issues, "
        "or follow-up PRs. "
        "If the user asks 'Why was X added?' and the artifacts do not directly "
        "explain the original decision, say: "
        "'The retrieved evidence does not directly show why X was originally added.' "
        "Then summarize what the evidence does show. "
        "Do not say 'X was added because' unless a retrieved artifact directly "
        "supports that causal claim.\n\n"
        "Evidence Strength labels:\n"
        "- strong: retrieved artifacts directly explain the decision\n"
        "- partial: artifacts are related but mostly show later evolution/docs\n"
        "- weak: artifacts are only loosely related\n\n"
        "Output one short section per useful artifact:\n"
        "Evidence Strength: <strong|partial|weak>\n\n"
        "Artifact N: <source>\n"
        "Distance: <distance>\n\n"
        "Summary:\n"
        "<1-3 concise sentences>\n\n"
        "Why it matters:\n"
        "<1-2 concise sentences>\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )

    return _generate(prompt, "Answer code history questions using only provided context.")


def generate_context_pack(change_request: str, artifacts: list[dict]) -> str:
    """Generate historical context for a planned code change."""
    context = _format_artifacts_context(artifacts)
    prompt = (
        "Create a concise context pack for a developer before a planned code change. "
        "Use only the provided context. "
        "Do not answer the change request. "
        "Do not suggest code, implementations, libraries, or next steps. "
        "Focus on historical evidence, constraints, risks, related discussions, "
        "and architectural decisions.\n\n"
        "Output these sections exactly:\n"
        "RELEVANT HISTORICAL EVIDENCE\n"
        "<source>\n"
        "Distance: <distance>\n\n"
        "Summary:\n"
        "<concise historical summary>\n\n"
        "Potential impact:\n"
        "<what this history may constrain or affect>\n\n"
        "CONSTRAINTS\n"
        "- <historical constraint>\n\n"
        "RISKS\n"
        "- <risk implied by the evidence>\n\n"
        "RELATED DISCUSSIONS\n"
        "- <source>: <short note>\n\n"
        "If evidence is insufficient, state that in the relevant section.\n\n"
        f"CHANGE REQUEST:\n{change_request}\n\n"
        f"Context:\n{context}"
    )

    return _generate(
        prompt,
        "You are a historical context retrieval tool for mature repositories.",
    )


def _generate(prompt: str, system_prompt: str) -> str:
    from groq import Groq

    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise GroqConfigurationError(
            "GROQ_API_KEY is not set. Set it in your environment or .env file."
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_tokens=500,
    )

    return response.choices[0].message.content


def _format_artifacts_context(artifacts: list[dict]) -> str:
    context_parts = []

    for artifact in artifacts:
        metadata = artifact.get("metadata") or {}
        source = artifact.get("source") or metadata.get("source") or "Unknown source"
        artifact_type = metadata.get("type", "")
        url = artifact.get("url") or metadata.get("url") or ""
        distance = artifact.get("distance", "")
        text = artifact.get("text") or ""

        context_parts.append(
            f"[Source: {source}]\n"
            f"Type: {artifact_type}\n"
            f"URL: {url}\n"
            f"Distance: {distance}\n"
            "Content:\n"
            f"{text}"
        )

    return "\n\n".join(context_parts)
