import argparse
import os
import pickle

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    step = size - overlap          # how far we slide forward each time
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def build_index(pdf_path: str, index_dir: str = "index") -> None:
    os.makedirs(index_dir, exist_ok=True)

    text = extract_text(pdf_path)
    if not text.strip():
        raise ValueError("No extractable text found (scanned PDF? needs OCR).")

    chunks = chunk_text(text)

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"Done. {index.ntotal} vectors indexed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--index-dir", default="index")
    args = parser.parse_args()
    build_index(args.pdf_path, args.index_dir)