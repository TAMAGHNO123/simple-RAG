import { useEffect, useRef, useState, type FormEvent } from "react";
import type { AskResponse, HealthResponse, Turn } from "./types";

const API_BASE = ""; // same-origin, works with the Vite proxy

type BackendStatus = "checking" | "online" | "offline";

export default function App() {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [question, setQuestion] = useState<string>("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let isMounted = true;

    fetch(`${API_BASE}/health`)
      .then((res) => res.json() as Promise<HealthResponse>)
      .then((data) => {
        if (isMounted) {
          setBackendStatus(data.status === "ok" ? "online" : "offline");
        }
      })
      .catch(() => {
        if (isMounted) {
          setBackendStatus("offline");
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  async function handleSubmit(e: FormEvent<HTMLFormElement>): Promise<void> {
    e.preventDefault();
    const q = question.trim();
    if (!q) return;

    setQuestion("");
    setTurns((prev) => [...prev, { question: q, answer: "", sources: [], status: "pending" }]);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data: AskResponse & { detail?: string } = await res.json();
      if (!res.ok) throw new Error(data.detail ?? "Request failed");

      setTurns((prev) =>
        prev.map((t, i) =>
          i === prev.length - 1 ? { ...t, answer: data.answer, sources: data.sources, status: "done" } : t
        )
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setTurns((prev) =>
        prev.map((t, i) => (i === prev.length - 1 ? { ...t, answer: message, status: "error" } : t))
      );
    }
  }

  return (
    <div className="page">
      <header className="masthead">
        <h1>Chat with your PDF</h1>
        <span className={`status ${backendStatus === "online" ? "online" : "offline"}`}>
          {backendStatus === "checking" ? "checking backend" : backendStatus}
        </span>
      </header>

      {turns.length === 0 ? (
        <div className="empty-state">No questions asked yet. Type one below.</div>
      ) : (
        <div className="thread">
          {turns.map((turn, i) => (
            <div className="turn" key={i}>
              <div className="eyebrow">Q{i + 1}</div>
              <div className="question">{turn.question}</div>
              <div
                className={`answer ${turn.status === "pending" ? "pending" : ""} ${
                  turn.status === "error" ? "error" : ""
                }`}
              >
                {turn.status === "pending" ? "Reading the document..." : turn.answer}
              </div>
              {turn.sources.length > 0 && (
                <div className="sources">
                  <div className="sources-label">Highlighted in the source</div>
                  {turn.sources.map((s, j) => (
                    <div className="highlight-strip" key={j}>
                      {s}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      <div ref={bottomRef} />

      <div className="composer">
        <form onSubmit={handleSubmit}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about the document..."
            aria-label="Question"
          />
          <button type="submit" disabled={!question.trim()}>
            Ask
          </button>
        </form>
      </div>
    </div>
  );
}
