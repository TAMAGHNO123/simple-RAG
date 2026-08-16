from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss
import numpy as np
import ollama

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "sample.pdf"

#extraction of given text
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text

# chunking the text
def chunk_text(text: str, chunk_size: int = 500,overlap: int=50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# turning chunks into vector embeddings
class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []   #stores orig txt chunk

    def build(self, chunks: list[str]):
        self.chunks = chunks
        embeddings = np.asarray(self.model.encode(chunks, show_progress_bar=True), dtype=np.float32)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
    def search(self, query: str, k: int = 3)->list[str]:
        if self.index is None:
            raise ValueError("Index has not been built yet. Call build(chunks) first.")
        query_vec = np.asarray(self.model.encode([query]), dtype=np.float32)
        distances, indices = self.index.search(query_vec, k)
        return [self.chunks[i] for i in indices[0]]


def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question using only the context below.
If the answer isn't in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {query}

Answer:"""
    return prompt


# calling ollama
def call_llm(prompt: str, model: str = "llama3.2") -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def main():
    pdf_path = DEFAULT_PDF_PATH
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("No text in PDF.")
        return
    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    print("Building vector store (embedding chunks)...")
    store = VectorStore()
    store.build(chunks)
    print("\nReady! Ask questions about your PDF.\n")
    while True:
        query = input("Ask a question (or type 'quit'): ")
        if query.lower() == "quit":
            break
        retrieved = store.search(query, k=3)
        prompt = build_prompt(query, retrieved)
        print("\n--- Retrieved chunks ---")
        for i, chunk in enumerate(retrieved):
            print(f"[{i+1}] {chunk[:100]}...")
        print("\nGenerating answer...")
        answer = call_llm(prompt)
        print(f"\n--- Answer ---\n{answer}\n")
 
 
if __name__ == "__main__":
    main()
 
