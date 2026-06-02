import json
from pathlib import Path


PROCESSED_DATA_DIR = Path("data/processed")
CHROMA_DATA_DIR = Path("data/chroma")
COLLECTION_NAME = "codearch_artifacts"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 100


def load_artifacts(
    owner: str,
    repo: str,
    processed_data_dir: Path | None = None,
) -> list[dict]:
    processed_data_dir = processed_data_dir or PROCESSED_DATA_DIR
    artifacts_path = processed_data_dir / f"{owner}_{repo}" / "artifacts.jsonl"

    with artifacts_path.open(encoding="utf-8") as artifacts_file:
        return [json.loads(line) for line in artifacts_file if line.strip()]


def build_index(
    owner: str,
    repo: str,
    processed_data_dir: Path | None = None,
    chroma_data_dir: Path | None = None,
) -> int:
    processed_data_dir = processed_data_dir or PROCESSED_DATA_DIR
    chroma_data_dir = chroma_data_dir or CHROMA_DATA_DIR
    artifacts = load_artifacts(owner, repo, processed_data_dir=processed_data_dir)
    chroma_path = chroma_data_dir / f"{owner}_{repo}"

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    client = PersistentClient(path=str(chroma_path))
    _delete_collection_if_exists(client, COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME)

    print("Embedding artifacts...")
    print(f"Total artifacts to embed: {len(artifacts)}")

    if not artifacts:
        print("Final Chroma collection count: 0")
        return 0

    model = SentenceTransformer(MODEL_NAME)
    total_batches = (
        len(artifacts) + EMBEDDING_BATCH_SIZE - 1
    ) // EMBEDDING_BATCH_SIZE

    for batch_number, start_index in enumerate(
        range(0, len(artifacts), EMBEDDING_BATCH_SIZE),
        start=1,
    ):
        print(f"  batch {batch_number}/{total_batches}")
        batch = artifacts[start_index : start_index + EMBEDDING_BATCH_SIZE]
        documents = [artifact["text"] for artifact in batch]
        embeddings = model.encode(documents, show_progress_bar=False)

        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        collection.add(
            ids=[artifact["id"] for artifact in batch],
            documents=documents,
            embeddings=embeddings,
            metadatas=[_metadata_for_chroma(artifact) for artifact in batch],
        )

    collection_count = collection.count()
    print(f"Final Chroma collection count: {collection_count}")

    return collection_count


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
