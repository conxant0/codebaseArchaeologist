import os

from dotenv import load_dotenv

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


class GroqConfigurationError(Exception):
    pass


def compute_evidence_strength(retrieved_artifacts: list[dict]) -> str:
    # V1 evidence strength measures retrieval quality, not full answer correctness.
    distances = [
        artifact["distance"]
        for artifact in retrieved_artifacts[:3]
        if artifact.get("distance") is not None
    ]
    if not distances:
        return "weak"

    avg_distance = sum(distances) / len(distances)
    if avg_distance < 0.9:
        return "strong"
    if avg_distance < 1.2:
        return "partial"
    return "weak"


def answer_question(
    question: str,
    artifacts: list[dict],
    evidence_strength: str,
) -> str:
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
        f"Evidence Strength: {evidence_strength}\n"
        "Use this exact Evidence Strength value. "
        "Do not recalculate it. "
        "Do not change it.\n\n"
        "Output one short section per useful artifact:\n"
        "Artifact N: <source>\n"
        "Distance: <distance>\n\n"
        "Summary:\n"
        "<1-3 concise sentences>\n\n"
        "Why it matters:\n"
        "<1-2 concise sentences>\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )

    return _generate(
        prompt, "Answer code history questions using only provided context."
    )


def generate_context_pack(
    change_request: str,
    artifacts: list[dict],
    evidence_strength: str,
) -> str:
    """Generate historical context for a planned code change."""
    context = _format_artifacts_context(artifacts)
    prompt = (
        "You are generating a historical evidence report. "
        "Create a concise Markdown context pack for a developer before a planned "
        "code change. Make the context pack evidence-first. "
        "Use only retrieved artifacts. Prefer extraction over summarization. "
        "Prefer extracted findings and cited artifacts over interpretation. "
        "Every major statement must be traceable to a retrieved artifact. "
        "If retrieved artifacts do not directly support a claim, say evidence is "
        "insufficient. If evidence is incomplete, say so. "
        "Treat uncertainty as valuable information. "
        "Do not infer architectural intent unless directly supported. "
        "Do not generate risks, constraints, or recommendations unless explicitly "
        "present in retrieved artifacts. "
        "Do not invent constraints, risks, architectural intent, or original "
        "motivations. Do not generate code. "
        "The purpose is to prepare context before editing, not to edit the code.\n\n"
        "Quality check before returning: for every major claim ask, "
        "'Which retrieved artifact supports this claim?' If none, remove the claim.\n\n"
        "Return only Markdown with exactly these sections:\n\n"
        "# Codebase Archaeologist Context Pack\n\n"
        "## Change Request\n\n"
        f"{change_request}\n\n"
        "## Evidence Strength\n\n"
        f"{evidence_strength}\n\n"
        "## Retrieved Findings\n\n"
        "For each artifact:\n\n"
        "### <artifact>\n\n"
        "Type:\n"
        "<issue/pr/commit/doc>\n\n"
        "Finding:\n"
        "A concise factual statement directly supported by the artifact.\n\n"
        "Evidence:\n"
        "Short supporting excerpt or summary.\n\n"
        "Why Relevant:\n"
        "Why this artifact was retrieved for the change request.\n\n"
        "## Supported Conclusions\n\n"
        "Only conclusions directly supported by multiple retrieved artifacts. "
        "If there are none, say 'Evidence is insufficient for supported "
        "conclusions.'\n\n"
        "## Open Questions\n\n"
        "List important questions that the retrieved evidence does NOT answer. "
        "This section is extremely important.\n\n"
        "## Sources\n\n"
        "List all retrieved artifacts.\n\n"
        f"Context:\n{context}"
    )

    return _generate(
        prompt,
        "You are a historian and researcher for mature software repositories.",
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
