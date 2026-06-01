import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq
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
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(texts)

    client = chromadb.Client()
    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids = [f"doc_{i}" for i in range(len(chunks))]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    return collection


def retrieve_relevant_chunks(question: str, collection, embedder, top_k: int = 3):
    question_embedding = embedder.encode(question)

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k,
    )

    chunks = []

    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "source": metadata["source"],
                "text": doc,
                "distance": distance,
            }
        )

    return chunks


def ask_groq(question: str, chunks):
    context_parts = []

    for chunk in chunks:
        context_parts.append(
            f"[Source: {chunk['source']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say you do not know.\n\n"
        f"Context:\n\n{context}\n\n"
        f"Question:\n{question}"
    )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only the provided context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_completion_tokens=300,
    )

    return response.choices[0].message.content


def main():
    load_dotenv()

    documents = load_text_files(DATA_DIR)
    print(f"Loaded {len(documents)} documents")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    collection = build_chroma_collection(documents, embedder)
    print(f"Stored {collection.count()} documents in Chroma")

    question = "What happens when an access token expires?"

    chunks = retrieve_relevant_chunks(
        question=question,
        collection=collection,
        embedder=embedder,
        top_k=3,
    )

    answer = ask_groq(question, chunks)

    print(answer)


if __name__ == "__main__":
    main()
