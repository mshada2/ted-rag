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
    if (mode === "title_speaker") return "Give me 1 TED talk title and speaker about fear.";
    return "Ask a question about TED talks (answer must come from retrieved TED transcript chunks).";
  }, [mode]);

  async function run() {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const body = {
        question: (question || placeholder).trim(),
        mode, // IMPORTANT: always send, including "qa"
      };

      const r = await fetch("/api/prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const json = await r.json().catch(() => ({}));

      if (!r.ok) {
        setError(json?.error ? String(json.error) : `HTTP ${r.status}`);
        setLoading(false);
        return;
      }

      setData(json as PromptResponse);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ minHeight: "100vh", background: "#0b0b0c", color: "#f2f2f2", padding: 32 }}>
      <h1 style={{ fontSize: 44, fontWeight: 800, marginBottom: 8 }}>TED-RAG</h1>
      <p style={{ maxWidth: 900, color: "#b7b7b7", marginTop: 0, marginBottom: 24 }}>
        Answers must be grounded in retrieved TED transcript context. If not supported: “I don’t know based on the provided
        TED data.”
      </p>

      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
        <label style={{ fontWeight: 700 }}>Mode</label>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as Mode)}
          style={{
            background: "#141418",
            color: "#f2f2f2",
            border: "1px solid #2c2c33",
            borderRadius: 6,
            padding: "8px 10px",
            minWidth: 160,
          }}
        >
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
        style={{
          width: "100%",
          maxWidth: 980,
          background: "#141418",
          color: "#f2f2f2",
          border: "1px solid #2c2c33",
          borderRadius: 8,
          padding: 14,
          fontSize: 16,
          lineHeight: 1.4,
        }}
      />

      <div style={{ marginTop: 12 }}>
        <button
          onClick={run}
          disabled={loading}
          style={{
            background: loading ? "#2a2a33" : "#1f1f27",
            color: "#ffffff",
            border: "1px solid #3a3a48",
            borderRadius: 8,
            padding: "10px 18px",
            cursor: loading ? "not-allowed" : "pointer",
            fontWeight: 700,
          }}
        >
          {loading ? "Working…" : "Ask"}
        </button>
      </div>

      <div style={{ marginTop: 22, maxWidth: 980 }}>
        <h2 style={{ fontSize: 20, marginBottom: 10 }}>Response</h2>

        {error && (
          <pre
            style={{
              whiteSpace: "pre-wrap",
              background: "#2a1212",
              border: "1px solid #5a2a2a",
              borderRadius: 8,
              padding: 14,
              color: "#ffd5d5",
            }}
          >
            {error}
          </pre>
        )}

        {data && (
          <>
            <pre
              style={{
                whiteSpace: "pre-wrap",
                background: "#0f0f13",
                border: "1px solid #2c2c33",
                borderRadius: 8,
                padding: 14,
                color: "#f2f2f2",
                minHeight: 48,
              }}
            >
              {data.response}
            </pre>

            <details style={{ marginTop: 14 }}>
              <summary style={{ cursor: "pointer", fontWeight: 700 }}>Retrieved context (debug)</summary>
              <pre
                style={{
                  marginTop: 10,
                  whiteSpace: "pre-wrap",
                  background: "#0f0f13",
                  border: "1px solid #2c2c33",
                  borderRadius: 8,
                  padding: 14,
                  color: "#f2f2f2",
                  overflowX: "auto",
                }}
              >
                {JSON.stringify(data.context, null, 2)}
              </pre>
            </details>

            <details style={{ marginTop: 12 }}>
              <summary style={{ cursor: "pointer", fontWeight: 700 }}>Augmented prompt (debug)</summary>
              <pre
                style={{
                  marginTop: 10,
                  whiteSpace: "pre-wrap",
                  background: "#0f0f13",
                  border: "1px solid #2c2c33",
                  borderRadius: 8,
                  padding: 14,
                  color: "#f2f2f2",
                  overflowX: "auto",
                }}
              >
                {JSON.stringify(data.Augmented_prompt, null, 2)}
              </pre>
            </details>
          </>
        )}
      </div>
    </main>
  );
}
