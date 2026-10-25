
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# 1. Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "chroma_db"

# 2. Configuration
COLLECTION_NAME = "insurance_policies"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def search_documents(question, model, collection, top_k=3):
    """Find the most relevant document chunks for a question."""

    # Convert the question into an embedding
    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    # Search Chroma for the closest vectors
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    return results


def main():

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to Chroma...")
    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(f"Database contains {collection.count()} chunks")

    if collection.count() == 0:
        print("No chunks available. Build the database first.")
        return

    while True:

        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        results = search_documents(
            question, model, collection, top_k=3
        )

        print("\nMost relevant passages:\n")

        for i, (text, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ),
            start=1
        ):

            print("=" * 60)
            print(f"RESULT {i}")
            print(f"Source: {metadata['source']}")
            print(f"Page: {metadata['page']}")
            print(f"Cosine distance: {distance:.4f}")
            print("-" * 60)
            print(text)
            print()


if __name__ == "__main__":
    main()
