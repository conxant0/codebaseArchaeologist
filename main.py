from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
COLLECTION_NAME = "local_docs"


def load_text_files(data_dir: Path):
    documents = []

    for file_path in sorted(data_dir.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")
        documents.append(
            {
                "source": file_path.name,
                "text": content,
            }
        )

    return documents


def chunk_text(text: str):

    pass


def build_chroma_collection(chunks, embedder):
    pass


def retrieve_relevant_chunks(question: str, collection, embedder, top_k: int = 3):
    pass


def ask_groq(question: str, chunks):
    pass


def main():
    load_dotenv()

    documents = load_text_files(DATA_DIR)
    print(f"Loaded {len(documents)} documents")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [doc["text"] for doc in documents]
    embeddings = embedder.encode(texts)

    client = chromadb.Client()

    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids = [f"doc_{i}" for i in range(len(documents))]
    metadatas = [{"source": doc["source"]} for doc in documents]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print(f"Stored {collection.count()} documents in Chroma")

    question = "Why was refreshSession added?"

    question_embedding = embedder.encode(question)

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=3,
    )

    print(results)

    print(type(embeddings))
    print(embeddings.shape)


if __name__ == "__main__":
    main()
