import os
import pickle
from contextlib import asynccontextmanager

import faiss
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
INDEX_DIR = os.environ.get("INDEX_DIR", "index")
DEFAULT_TOP_K = 3

state: dict = {}   


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    chunks_path = os.path.join(INDEX_DIR, "chunks.pkl")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise RuntimeError(f"No index in '{INDEX_DIR}/'. Run ingest.py first.")

    state["index"] = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        state["chunks"] = pickle.load(f)
    state["model"] = SentenceTransformer(EMBED_MODEL)

    yield          # <- app runs while paused here
    state.clear()  # cleanup on shutdown


app = FastAPI(title="Chat with your PDF", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K


class Answer(BaseModel):
    answer: str
    sources: list[str]


def retrieve(question: str, top_k: int) -> list[str]:
    q_vec = state["model"].encode([question], convert_to_numpy=True)
    _distances, indices = state["index"].search(q_vec, top_k)
    return [state["chunks"][i] for i in indices[0] if i != -1]


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    return (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def call_ollama(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


@app.get("/health")
def health():
    return {
        "status": "ok" if "index" in state else "index not loaded",
        "vectors": state["index"].ntotal if "index" in state else 0,
        "model": OLLAMA_MODEL,
    }


@app.post("/ask", response_model=Answer)
def ask(query: Query):
    if "index" not in state:
        raise HTTPException(503, "Index not loaded")
    if not query.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    chunks = retrieve(query.question, query.top_k)
    if not chunks:
        raise HTTPException(404, "No relevant context found in the document")

    prompt = build_prompt(query.question, chunks)
    try:
        answer_text = call_ollama(prompt)
    except requests.RequestException as e:
        raise HTTPException(502, f"Could not reach Ollama at {OLLAMA_URL}: {e}")

    return Answer(answer=answer_text, sources=chunks)