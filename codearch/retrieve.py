from codearch.index import CHROMA_DATA_DIR
from codearch.index import COLLECTION_NAME
from codearch.index import MODEL_NAME


class IndexNotFoundError(Exception):
    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        super().__init__(f"No index found for {owner}/{repo}")


def retrieve_relevant_artifacts(
    owner: str,
    repo: str,
    question: str,
    top_k: int = 5,
) -> list[dict]:
    chroma_path = CHROMA_DATA_DIR / f"{owner}_{repo}"
    if not chroma_path.exists():
        raise IndexNotFoundError(owner, repo)

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    question_embedding = model.encode(question, show_progress_bar=False)

    if hasattr(question_embedding, "tolist"):
        question_embedding = question_embedding.tolist()

    client = PersistentClient(path=str(chroma_path))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise IndexNotFoundError(owner, repo) from exc

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    artifacts = []
    for artifact_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        metadata = metadata or {}
        artifacts.append(
            {
                "id": artifact_id,
                "source": metadata.get("source", ""),
                "text": document,
                "distance": distance,
                "metadata": metadata,
            }
        )

    return artifacts
