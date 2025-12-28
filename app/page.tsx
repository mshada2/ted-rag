"use client";

import { useMemo, useState } from "react";

type Mode = "qa" | "title_list" | "title_speaker";

type PromptResponse = {
  response: string;
  context: any[];
  Augmented_prompt: any;
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("qa");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<PromptResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const placeholder = useMemo(() => {
    if (mode === "title_list") return "Give me exactly 3 TED talk titles about creativity.";
    if (mode === "title_speaker") return "Give me 1 TED talk title and speaker about creativity.";
    return "Ask a question about TED talks (answer must come from retrieved TED transcript chunks).";
  }, [mode]);

  const panelStyle: React.CSSProperties = {
    whiteSpace: "pre-wrap",
    padding: 12,
    background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.12)",
    color: "inherit",
  };

  async function run() {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const body: any = { question: question || placeholder };
      if (mode !== "qa") body.mode = mode;

      const r = await fetch("/api/prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const j = await r.json();
      if (!r.ok) throw new Error(j?.error ?? `HTTP ${r.status}`);
      setData(j);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 900, margin: "40px auto", padding: 16, fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8 }}>TED-RAG</h1>
      <p style={{ marginTop: 0, opacity: 0.85 }}>
        Answers must be grounded in retrieved TED transcript context. If not supported: “I don’t know based on the
        provided TED data.”
      </p>

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 16 }}>
        <label style={{ fontWeight: 600 }}>Mode</label>
        <select value={mode} onChange={(e) => setMode(e.target.value as Mode)} style={{ padding: 8 }}>
          <option value="qa">qa</option>
          <option value="title_list">title_list</option>
          <option value="title_speaker">title_speaker</option>
        </select>
      </div>

      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder={placeholder}
        rows={4}
        style={{ width: "100%", marginTop: 12, padding: 12, fontSize: 14 }}
      />

      <button
        onClick={run}
        disabled={loading}
        style={{ marginTop: 12, padding: "10px 14px", fontWeight: 700, cursor: loading ? "not-allowed" : "pointer" }}
      >
        {loading ? "Running..." : "Ask"}
      </button>

      {error && (
        <div style={{ marginTop: 16, padding: 12, background: "rgba(255,0,0,0.12)", border: "1px solid rgba(255,0,0,0.25)" }}>
          <b>Error:</b> {error}
        </div>
      )}

      {data && (
        <div style={{ marginTop: 18 }}>
          <h2 style={{ fontSize: 18, marginBottom: 6 }}>Response</h2>
          <pre style={panelStyle}>{data.response}</pre>

          <details style={{ marginTop: 12 }}>
            <summary style={{ cursor: "pointer", fontWeight: 700 }}>Retrieved context (debug)</summary>
            <pre style={panelStyle}>{JSON.stringify(data.context, null, 2)}</pre>
          </details>

          <details style={{ marginTop: 12 }}>
            <summary style={{ cursor: "pointer", fontWeight: 700 }}>Augmented prompt (debug)</summary>
            <pre style={panelStyle}>{JSON.stringify(data.Augmented_prompt, null, 2)}</pre>
          </details>
        </div>
      )}
    </main>
  );
}
