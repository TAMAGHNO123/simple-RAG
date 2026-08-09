# Chat with Your PDF — A Local RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that lets you ask natural-language questions about any PDF document and get grounded, hallucination-resistant answers — running entirely with free/local tools.

## What it does

1. Extracts text from a PDF
2. Splits the text into overlapping chunks
3. Converts each chunk into a vector embedding
4. Stores embeddings in a searchable vector index
5. On a user query, retrieves the most relevant chunks
6. Feeds the query + retrieved context into a local LLM to generate a grounded answer

The model is instructed to answer **only** from the retrieved context — if the answer isn't in the document, it says so instead of making something up.

## Tech Stack

| Component | Tool Used | Why |
|---|---|---|
| PDF parsing | `pypdf` | Lightweight, pure-Python text extraction |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Fast, small (384-dim), runs on CPU |
| Vector store | `FAISS` (Facebook AI Similarity Search) | Free, local, no external service needed |
| LLM inference | `Ollama` running `llama3.2` | Fully local, free, no API costs or rate limits |
| Language | Python 3.11 | — |

## Architecture

```
PDF file
   │
   ▼
Text Extraction (pypdf)
   │
   ▼
Chunking (500 chars, 50 char overlap)
   │
   ▼
Embedding (sentence-transformers)
   │
   ▼
Vector Index (FAISS)
   │
   ▼
User Query ──► Embed Query ──► Similarity Search ──► Top-K Chunks
                                                          │
                                                          ▼
                                              Prompt Construction
                                                          │
                                                          ▼
                                              Local LLM (Ollama / llama3.2)
                                                          │
                                                          ▼
                                                    Grounded Answer
```

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd chat-with-pdf

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull the model
# Download from https://ollama.com
ollama pull llama3.2
```

## Usage

1. Place your PDF in the project folder, named `sample.pdf`
2. Run:
```bash
python rag_pipeline_ollama.py
```
3. Ask questions about the document interactively in the terminal. Type `quit` to exit.

## Design Decisions

- **Chunking with overlap (500 chars, 50 char overlap):** Prevents losing context when relevant information spans a chunk boundary — a sentence cut in half at the edge of one chunk still appears intact in the next.
- **`IndexFlatL2` (brute-force search) instead of an approximate index (e.g. HNSW):** Chosen for correctness and simplicity on small document sizes (a few hundred chunks). Would need to switch to an approximate nearest-neighbor index for scaling to large corpora.
- **Strict "answer only from context" prompt:** Directly reduces hallucination — a core failure mode of naive LLM Q&A without retrieval grounding.
- **Local LLM (Ollama) over a cloud API:** Removes cost and rate-limit constraints for local development and experimentation; trades off inference speed/quality depending on hardware.

## Challenges Faced

- **Windows symlink warning from Hugging Face cache:** By default, `huggingface_hub` uses symlinks to avoid duplicating cached model files, which Windows doesn't support without Developer Mode or admin rights. Resolved by allowing the fallback (non-symlink) caching mode; only affects disk usage, not functionality.
- **PATH not recognizing `ollama` after installation on Windows:** The installer registers Ollama in PATH, but existing terminal sessions don't pick up the change until restarted. Resolved by reopening the terminal after installation.
- **Chunk size tuning:** Initial fixed-size character chunking occasionally split sentences awkwardly mid-word. Overlap between chunks mitigates most context loss; a future improvement is switching to sentence-aware or semantic chunking.
- **Scanned/handwritten PDFs are not supported out of the box:** `pypdf` only extracts existing digital text layers. Documents that are scanned images (including handwritten notes) require an OCR preprocessing step (e.g. Tesseract, EasyOCR, or a cloud OCR API) before this pipeline can process them — a planned extension.

## Known Limitations / Knowledge Gaps

- No re-ranking step after initial retrieval — retrieval quality depends entirely on embedding similarity, which can occasionally surface less relevant chunks over more relevant ones with subtly different wording.
- No evaluation harness yet (e.g. retrieval recall@k, answer faithfulness scoring) — currently validated manually by inspection.
- Single-document only; no multi-document corpus support or metadata filtering.
- No persistence layer — the FAISS index is rebuilt from scratch on every run rather than cached to disk.

## Roadmap

- [ ] Add sentence-aware/semantic chunking
- [ ] Add a reranking step (e.g. cross-encoder) before passing chunks to the LLM
- [ ] Persist the FAISS index to disk to avoid re-embedding on every run
- [ ] Add OCR support for scanned/handwritten PDFs
- [ ] Wrap the pipeline in a FastAPI backend with a simple web UI
- [ ] Add retrieval evaluation metrics (recall@k, MRR)
- [ ] Containerize with Docker for reproducible deployment

## License

MIT
