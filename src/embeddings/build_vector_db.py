
from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


# 1. Project configuration
BASE_DIR = Path(__file__).resolve().parents[2]

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
DB_PATH = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "insurance_policies"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    # 2. Load the chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No chunks found in chunks.json")

    print(f"Loaded {len(chunks)} chunks")

    # 3. Load the embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # 4. Extract the text of each chunk
    texts = [chunk["text"] for chunk in chunks]

    # 5. Convert texts to numerical vectors
    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print(f"Embeddings shape: {embeddings.shape}")

    # 6. Connect to Chroma
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # Use cosine distance for normalized embeddings
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}}
    )

    # 7. Prepare metadata and identifiers
    ids = [
        f"{chunk['source']}::{chunk['chunk_id']}"
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]

    # 8. Save the embeddings in batches
    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size

        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end]
        )

    print("\nVector database created successfully!")
    print(f"Total stored chunks: {collection.count()}")
    print(f"Database location: {DB_PATH}")


if __name__ == "__main__":
    main()
