import json
from pathlib import Path


PROCESSED_DATA_DIR = Path("data/processed")
CHROMA_DATA_DIR = Path("data/chroma")
COLLECTION_NAME = "codearch_artifacts"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_artifacts(owner: str, repo: str) -> list[dict]:
    artifacts_path = PROCESSED_DATA_DIR / f"{owner}_{repo}" / "artifacts.jsonl"

    with artifacts_path.open(encoding="utf-8") as artifacts_file:
        return [json.loads(line) for line in artifacts_file if line.strip()]


def build_index(owner: str, repo: str) -> int:
    artifacts = load_artifacts(owner, repo)
    chroma_path = CHROMA_DATA_DIR / f"{owner}_{repo}"

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    client = PersistentClient(path=str(chroma_path))
    _delete_collection_if_exists(client, COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME)

    if not artifacts:
        return 0

    model = SentenceTransformer(MODEL_NAME)
    documents = [artifact["text"] for artifact in artifacts]
    embeddings = model.encode(documents, show_progress_bar=False)

    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()

    collection.add(
        ids=[artifact["id"] for artifact in artifacts],
        documents=documents,
        embeddings=embeddings,
        metadatas=[_metadata_for_chroma(artifact) for artifact in artifacts],
    )

    return len(artifacts)


def _delete_collection_if_exists(client, name: str):
    collection_names = [
        collection.name if hasattr(collection, "name") else collection
        for collection in client.list_collections()
    ]

    if name in collection_names:
        client.delete_collection(name=name)


def _metadata_for_chroma(artifact: dict) -> dict:
    metadata = dict(artifact.get("metadata") or {})
    metadata["source"] = artifact.get("source", "")

    return {
        key: value if isinstance(value, str | int | float | bool) else ""
        for key, value in metadata.items()
    }
