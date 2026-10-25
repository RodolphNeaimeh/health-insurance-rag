
from pathlib import Path
import json
import pymupdf

BASE_DIR = Path(__file__).resolve().parents[2]

PDF_PATH = BASE_DIR / "data" / "raw" / "sample_policy.pdf"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "chunks.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start += step

    return chunks


def process_pdf(pdf_path):
    all_chunks = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)

            page_chunks = split_text(
                text, CHUNK_SIZE, CHUNK_OVERLAP
            )

            for index, chunk_text in enumerate(page_chunks):
                all_chunks.append({
                    "chunk_id": f"page_{page_number}_chunk_{index}",
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": chunk_text
                })

    return all_chunks


def main():
    chunks = process_pdf(PDF_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)

    print(f"Successfully created {len(chunks)} chunks!")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
