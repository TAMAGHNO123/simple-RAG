# Chat with Your PDF - Local RAG Pipeline

A local Retrieval-Augmented Generation (RAG) project for asking natural-language questions about PDF documents. The backend extracts PDF text, chunks it, embeds it, searches relevant chunks with FAISS, and sends grounded context to a local Ollama model.

## Project Structure

```text
chat-with-pdf/
  backend/
    __init__.py
    rag_pipeline_ollama.py
    requirements.txt
  frontend/
    README.md
  data/
    README.md
    sample.pdf
  README.md
```

## Backend

The current app is a Python CLI pipeline.

```bash
cd backend
python -m venv ../venv
../venv/Scripts/activate
pip install -r requirements.txt
```

Install Ollama from <https://ollama.com>, then pull the default model:

```bash
ollama pull llama3.2
```

Run the pipeline:

```bash
python rag_pipeline_ollama.py
```

By default, the CLI reads:

```text
data/sample.pdf
```

To use another document, place it in `data/` and update `DEFAULT_PDF_PATH` in `backend/rag_pipeline_ollama.py`.

## Frontend

`frontend/` is reserved for the future web UI. A common professional next step would be adding a Vite React app or a Next.js app here after the Python backend is exposed through an API such as FastAPI.

## Tech Stack

| Area | Tool |
|---|---|
| PDF parsing | `pypdf` |
| Embeddings | `sentence-transformers` |
| Vector search | `faiss-cpu` |
| LLM runtime | `ollama` |
| Language | Python |

## Roadmap

- Add a FastAPI backend service
- Add a web frontend in `frontend/`
- Support uploading PDFs through the UI
- Persist FAISS indexes
- Add OCR support for scanned PDFs
