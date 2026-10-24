
from pathlib import Path
import pymupdf

# Locate the project root
BASE_DIR = Path(__file__).resolve().parents[2]

# Input and output paths
PDF_PATH = BASE_DIR / "data" / "raw" / "sample_policy.pdf"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "sample_policy.txt"


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF, keeping page numbers."""

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages_text = []

    with pymupdf.open(pdf_path) as document:
        print(f"Total pages: {len(document)}")

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)

            pages_text.append(
                f"\n--- PAGE {page_number} ---\n{text}"
            )

            print(
                f"Page {page_number}: {len(text)} characters"
            )

    return "\n".join(pages_text)


def main():
    text = extract_pdf_text(PDF_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(text, encoding="utf-8")

    print(f"\nText saved to: {OUTPUT_PATH}")
    print(f"Total extracted characters: {len(text)}")


if __name__ == "__main__":
    main()
