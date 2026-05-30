from pathlib import Path

from dotenv import load_dotenv

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


if __name__ == "__main__":
    main()
