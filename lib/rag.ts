// lib/rag.ts
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";

/**
 * IMPORTANT: Your /api/stats endpoint must reflect these values.
 * The assignment constrains overlap_ratio <= 0.3 and top_k <= 30. :contentReference[oaicite:3]{index=3}
 */
export const RAG_CONFIG = {
  // This is what YOU report; the true chunking is done in your ingest script.
  // Keep these aligned with scripts/ingest.ts.
  chunk_size: 1024,
  overlap_ratio: 0.2,
  top_k: 25, // <= 30 per assignment :contentReference[oaicite:4]{index=4}
  min_score: 0.12,
  chat_model: process.env.LLMOD_CHAT_MODEL ?? "RPRTHPB-gpt-5-mini",
  embed_model: process.env.LLMOD_EMBED_MODEL ?? "RPRTHPB-text-embedding-3-small",
} as const;

export type PublicContextChunk = {
  talk_id: string;
  title: string;
  chunk: string;
  score: number;
};

type RetrievedMatch = {
  talk_id: string;
  title: string;
  speaker_1?: string;
  topics?: string[];
  chunk: string;
  score: number;
};

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`Missing env var: ${name}`);
  return v;
}

function getOpenAI() {
  const apiKey = requireEnv("LLMOD_API_KEY");
  const baseURL = process.env.LLMOD_BASE_URL ?? "https://api.llmod.ai";
  return new OpenAI({ apiKey, baseURL });
}

function getPineconeIndex() {
  const apiKey = requireEnv("PINECONE_API_KEY");
  const indexName = requireEnv("PINECONE_INDEX");
  const host = process.env.PINECONE_HOST; // recommended

  const pc = new Pinecone({ apiKey });

  // Support multiple Pinecone SDK shapes without you needing to guess versions:
  const anyPc = pc as any;
  if (typeof anyPc.index === "function") return host ? anyPc.index(indexName, host) : anyPc.index(indexName);
  if (typeof anyPc.Index === "function") return host ? anyPc.Index(indexName, host) : anyPc.Index(indexName);

  throw new Error("Could not create Pinecone index handle (SDK API mismatch).");
}

async function embedQuery(text: string): Promise<number[]> {
  const client = getOpenAI();
  const res = await client.embeddings.create({
    model: RAG_CONFIG.embed_model,
    input: text,
  });
  return res.data[0].embedding as unknown as number[];
}

async function retrieve(question: string): Promise<RetrievedMatch[]> {
  const vector = await embedQuery(question);
  const index = getPineconeIndex();

  const qres = await (index as any).query({
    vector,
    topK: RAG_CONFIG.top_k,
    includeMetadata: true,
  });

  const matches = (qres?.matches ?? []) as any[];
  const out: RetrievedMatch[] = [];

  for (const m of matches) {
    const score = typeof m.score === "number" ? m.score : 0;
    if (score < RAG_CONFIG.min_score) continue;

    const md = (m.metadata ?? {}) as Record<string, any>;
    const talk_id = String(md.talk_id ?? md.id ?? "");
    const title = String(md.title ?? "");
    const chunk = String(md.chunk ?? md.text ?? "");
    const speaker_1 = md.speaker_1 ? String(md.speaker_1) : undefined;

    let topics: string[] | undefined;
    if (Array.isArray(md.topics)) topics = md.topics.map((x: any) => String(x));
    else if (typeof md.topics === "string") {
      // tolerate CSV-like string
      topics = md.topics
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    }

    if (!talk_id || !title || !chunk) continue;

    out.push({ talk_id, title, speaker_1, topics, chunk, score });
  }

  // Highest score first
  out.sort((a, b) => b.score - a.score);
  return out;
}

/**
 * Required system prompt section (must be included or extremely similar). :contentReference[oaicite:5]{index=5}
 */
function buildSystemPrompt(): string {
  return [
    "You are a TED Talk assistant that answers questions strictly and",
    "only based on the TED dataset context provided to you (metadata",
    "and transcript passages). You must not use any external",
    "knowledge, the open internet, or information that is not explicitly",
    "contained in the retrieved context. If the answer cannot be",
    "determined from the provided context, respond: “I don’t know",
    "based on the provided TED data.” Always explain your answer",
    "using the given context, quoting or paraphrasing the relevant",
    "transcript or metadata when helpful.",
    "",
    "Additional style rules:",
    "- If the user requests multiple talk titles (e.g., exactly 3), output only titles in a list.",
    "- Do not invent talk titles, speakers, or facts not present in the provided context.",
    "- When the user request specifies an exact output format (e.g., Title/Speaker only), follow that format exactly, even if it includes no explanation.",
  ].join("\n");
}


function detectMode(question: string): "title_list" | "title_speaker" | "general" {
  const q = question.toLowerCase();
  if (q.includes("exactly 3") && q.includes("title")) return "title_list";
  if (q.includes("provide the title and speaker") || (q.includes("title") && q.includes("speaker"))) return "title_speaker";
  return "general";
}

function buildCandidates(matches: RetrievedMatch[], maxTalks: number): RetrievedMatch[] {
  // Group all matches by talk_id
  const grouped = new Map<string, RetrievedMatch[]>();
  for (const m of matches) {
    const arr = grouped.get(m.talk_id) ?? [];
    arr.push(m);
    grouped.set(m.talk_id, arr);
  }

  // For each talk: keep best match as the "candidate", but merge top evidence chunks into candidate.chunk
  const candidates: RetrievedMatch[] = [];

  for (const arr of grouped.values()) {
    // Sort by score descending
    arr.sort((a, b) => b.score - a.score);

    const best = arr[0];
    const topEvidence = arr.slice(0, 2); // keep up to 2 evidence snippets per talk

    const mergedEvidence = topEvidence
      .map((e) => e.chunk)
      .join("\n\n---\n\n"); // delimiter so it’s obvious it’s multiple snippets

    candidates.push({
      ...best,
      chunk: mergedEvidence, // IMPORTANT: this is now the evidence text we’ll show the model
      score: best.score,
    });
  }

  // Sort talks by best score, return top maxTalks
  candidates.sort((a, b) => b.score - a.score);
  return candidates.slice(0, maxTalks);
}



function cleanTopicList(topics?: string[]): string[] {
  if (!topics?.length) return [];

  // Handle the case where topics is ["['a','b','c']"] (a single string that looks like a list)
  if (topics.length === 1) {
    const t = topics[0].trim();
    if (t.startsWith("[") && t.endsWith("]")) {
      const inner = t.slice(1, -1);
      const parsed = inner
        .split(",")
        .map((x) => x.trim().replace(/^['"]|['"]$/g, ""))
        .filter(Boolean);
      if (parsed.length) return parsed;
    }
  }

  // Normal case: array of strings
  return topics
    .map((x) => x.trim().replace(/^['"]|['"]$/g, ""))
    .filter(Boolean);
}

function clip(text: string, maxChars: number) {
  const t = text.replace(/\s+/g, " ").trim();
  if (t.length <= maxChars) return t;
  return t.slice(0, maxChars - 1) + "…";
}

function buildCandidatesBlock(cands: RetrievedMatch[]): string {
  return cands
    .map((c, i) => {
      const topics = cleanTopicList(c.topics);
      const speaker = c.speaker_1 ? `SPEAKER: ${c.speaker_1}` : `SPEAKER:`;
      const topicsLine = `TOPICS: ${JSON.stringify(topics)}`;

      // Evidence comes from c.chunk (now merged by buildCandidates)
      const evidence = c.chunk ? clip(c.chunk, 700) : "(none)";

      return [
        `[#${i + 1}] talk_id=${c.talk_id}`,
        `TITLE: ${c.title}`,
        speaker,
        topicsLine,
        `EVIDENCE:\n"${evidence}"`,
        `SCORE: ${c.score}`,
      ].join("\n");
    })
    .join("\n\n");
}



function validateTitleSpeakerResponse(resp: string, cands: RetrievedMatch[]): boolean {
  // Must mention a title AND speaker that appear in candidates (exact substring match).
  const titles = cands.map((c) => c.title).filter(Boolean);
  const speakers = cands.map((c) => c.speaker_1).filter((s): s is string => Boolean(s));

  const hasTitle = titles.some((t) => resp.includes(t));
  const hasSpeaker = speakers.some((s) => resp.includes(s));
  return hasTitle && hasSpeaker;
}

function validateTitleListResponse(resp: string, cands: RetrievedMatch[], n: number): boolean {
  const lines = resp
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

  if (lines.length !== n) return false;

  const titles = new Set(cands.map((c) => c.title));
  for (const line of lines) {
    const cleaned = line.replace(/^[\-\*\d\.\)\s]+/, "").trim();
    if (!titles.has(cleaned)) return false;
  }
  return true;
}

async function callModel(system: string, user: string): Promise<string> {
  const client = getOpenAI();
  const res = await client.chat.completions.create({
    model: RAG_CONFIG.chat_model,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    // gpt-5-mini requires temperature=1 when temperature is specified (you hit this earlier).
    temperature: 1,
  });

  return res.choices?.[0]?.message?.content?.trim() ?? "";
}

export async function answerQuestion(question: string) {
  const matches = await retrieve(question);

  // Public context: always return chunk+score (required output shape). :contentReference[oaicite:6]{index=6}
  const publicContext: PublicContextChunk[] = matches.map((m) => ({
    talk_id: m.talk_id,
    title: m.title,
    chunk: m.chunk,
    score: m.score,
  }));

  const system = buildSystemPrompt();
  const mode = detectMode(question);

  // Build candidates for “must pick from these” prompts
  const cands = buildCandidates(matches, 8);
  const candidatesBlock = buildCandidatesBlock(cands);

  let userPrompt = "";
  if (mode === "title_speaker") {
    userPrompt = [
      "Choose exactly ONE talk from the CANDIDATES that best matches the QUESTION.",
      "Output format (exactly):",
      "Title: <title>",
      "Speaker: <speaker>",
      "",
      "How to decide a match:",
      "- A candidate is a valid match if its TITLE, TOPICS, or CHUNK explicitly mentions the key theme or close synonyms.",
      "- For 'fear'/'anxiety' queries, treat mentions of fear, anxious/anxiety, worry, panic, stress, or self-doubt as relevant.",
      "- Prefer the candidate whose CHUNK most directly discusses the theme asked in the QUESTION.",
      "",
      "Rules:",
      "- You MUST copy the Title and Speaker exactly as written in the CANDIDATES.",
      "- If NONE of the candidates contain the theme (or close synonyms) in TITLE/TOPICS/CHUNK, respond exactly:",
      "I don't know based on the provided TED data.",
      "",
      `QUESTION:\n${question}`,
      "",
      `CANDIDATES:\n${candidatesBlock || "(none)"}`,
  ].join("\n");
  }
 else if (mode === "title_list") {
    userPrompt = [
      "Return a list of exactly 3 talk titles that best match the QUESTION.",
      "Rules:",
      "- Output ONLY the titles as a list (no commentary).",
      "- Choose titles ONLY from the CANDIDATES below.",
      "- Do not repeat titles.",
      "- If you cannot find 3 valid titles from the candidates, respond: I don’t know based on the provided TED data.",
      "",
      `QUESTION:\n${question}`,
      "",
      `CANDIDATES:\n${candidatesBlock || "(none)"}`,
    ].join("\n");
  } else {
    // General Q&A with raw transcript chunks
    const contextBlock = matches
      .slice(0, RAG_CONFIG.top_k)
      .map((m, i) => `[#${i + 1}] talk_id=${m.talk_id}\nTITLE: ${m.title}\nCHUNK: ${m.chunk}\nSCORE: ${m.score}`)
      .join("\n\n");

    userPrompt = [
      "Answer the question using only the CONTEXT below.",
      "If the answer is not explicitly supported by the context, say: I don’t know based on the provided TED data.",
      "",
      `QUESTION:\n${question}`,
      "",
      `CONTEXT:\n${contextBlock || "(none)"}`,
    ].join("\n");
  }

  let response = await callModel(system, userPrompt);

  // Hard guardrails: reject hallucinated title/speaker or invalid title-lists.
  if (mode === "title_speaker") {
    if (!validateTitleSpeakerResponse(response, cands)) {
      response = "I don’t know based on the provided TED data.";
    }
  }
  if (mode === "title_list") {
    if (!validateTitleListResponse(response, cands, 3)) {
      response = "I don’t know based on the provided TED data.";
    }
  }

  return {
    response,
    context: publicContext,
    Augmented_prompt: { System: system, User: userPrompt },
  };
}
