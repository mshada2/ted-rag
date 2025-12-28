import fs from "fs";
import path from "path";
import dotenv from "dotenv";
import { parse } from "csv-parse";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";

// Load .env.local for this script
dotenv.config({ path: path.join(process.cwd(), ".env.local") });

function need(name: string) {
  const v = process.env[name];
  if (!v) throw new Error(`Missing env var: ${name}`);
  return v;
}

const OPENAI_API_KEY = need("OPENAI_API_KEY");
const OPENAI_BASE_URL = need("OPENAI_BASE_URL");
const PINECONE_API_KEY = need("PINECONE_API_KEY");
const PINECONE_INDEX = need("PINECONE_INDEX");
const PINECONE_INDEX_HOST = need("PINECONE_INDEX_HOST");

// Controls
const MAX_TALKS = Number(process.env.MAX_TALKS ?? 50); // number of talks to process this run
const DRY_RUN = process.env.DRY_RUN === "1";
const FORCE_REEMBED = process.env.FORCE_REEMBED === "1";

// Chunking (must respect assignment constraints)
const CHUNK_SIZE_TOKENS = 1024; // <= 2048
const OVERLAP_RATIO = 0.2; // <= 0.3
const CHARS_PER_TOKEN = 4;

// Batching
const EMBED_BATCH = 64;

function normalizeBaseURL(raw: string) {
  const trimmed = raw.trim().replace(/\/+$/, "");
  return trimmed.endsWith("/v1") ? trimmed : `${trimmed}/v1`;
}

function approxTokensFromText(s: string) {
  return Math.ceil((s ?? "").length / CHARS_PER_TOKEN);
}

function chunkText(text: string): string[] {
  const chunkChars = Math.floor(CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN);
  const overlapChars = Math.floor(chunkChars * OVERLAP_RATIO);
  const step = Math.max(1, chunkChars - overlapChars);

  const cleaned = (text ?? "").replace(/\s+/g, " ").trim();
  const chunks: string[] = [];

  for (let start = 0; start < cleaned.length; start += step) {
    const end = Math.min(cleaned.length, start + chunkChars);
    const chunk = cleaned.slice(start, end).trim();
    if (chunk) chunks.push(chunk);
    if (end === cleaned.length) break;
  }
  return chunks;
}

function pick(row: any, keys: string[]) {
  for (const k of keys) {
    if (row[k] !== undefined && row[k] !== null && String(row[k]).trim() !== "") {
      return String(row[k]);
    }
  }
  return "";
}

// ---- skip-already-ingested state file (local) ----
const STATE_PATH = path.join(process.cwd(), "scripts", "ingested_ids.json");

function loadIngestedSet(): Set<string> {
  try {
    const raw = fs.readFileSync(STATE_PATH, "utf-8");
    const arr = JSON.parse(raw);
    return new Set(Array.isArray(arr) ? arr.map(String) : []);
  } catch {
    return new Set();
  }
}

function saveIngestedSet(s: Set<string>) {
  fs.writeFileSync(STATE_PATH, JSON.stringify(Array.from(s), null, 2), "utf-8");
}

async function main() {
  const csvPath = path.join(process.cwd(), "data", "ted_talks_en.csv");
  if (!fs.existsSync(csvPath)) throw new Error(`CSV not found at: ${csvPath}`);

  const ingested = loadIngestedSet();

  const openai = new OpenAI({
    apiKey: OPENAI_API_KEY,
    baseURL: normalizeBaseURL(OPENAI_BASE_URL),
  });

  const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
  const index = pc.index(PINECONE_INDEX, PINECONE_INDEX_HOST);

  console.log(`Reading CSV: ${csvPath}`);
  console.log(`MAX_TALKS=${MAX_TALKS} | DRY_RUN=${DRY_RUN ? "1" : "0"} | FORCE_REEMBED=${FORCE_REEMBED ? "1" : "0"}`);
  console.log(`Local ingested state size: ${ingested.size} talks (${STATE_PATH})`);

  let processedTalks = 0;
  let vectorsUpserted = 0;

  // DRY RUN counters
  let estChunks = 0;
  let estTokens = 0;

  const parser = parse({
    columns: true,
    bom: true,
    relax_quotes: true,
    relax_column_count: true,
    skip_empty_lines: true,
  });

  const stream = fs.createReadStream(csvPath).pipe(parser);

  for await (const row of stream as any) {
    if (processedTalks >= MAX_TALKS) break;

    const talk_id = pick(row, ["talk_id", "id", "talkid"]);
    const title = pick(row, ["title", "name"]);
    const transcript = pick(row, ["transcript", "text", "transcript_en"]);
    const topics = pick(row, ["topics"]);
    const speaker_1 = pick(row, ["speaker_1"]);
    const url = pick(row, ["url"]);

    if (!talk_id || !title || !transcript) continue;

    // If re-embedding, only re-embed talks already in ingested_ids.json
    if (FORCE_REEMBED) {
      if (!ingested.has(talk_id)) continue;
    } else {
      // Normal mode: skip already ingested
      if (ingested.has(talk_id)) continue;
    }

    const chunks = chunkText(transcript);
    if (chunks.length === 0) continue;

    processedTalks += 1;

    if (DRY_RUN) {
      for (const c of chunks) {
        estChunks += 1;
        // enriched text is slightly longer; token estimate still fine
        estTokens += approxTokensFromText(c) + approxTokensFromText(title) + approxTokensFromText(topics) + approxTokensFromText(speaker_1);
      }
      console.log(`[DRY_RUN] talk_id=${talk_id} title="${title}" chunks=${chunks.length}`);
      continue;
    }

    console.log(`Talk ${processedTalks}/${MAX_TALKS}: talk_id=${talk_id} | chunks=${chunks.length}`);

    for (let i = 0; i < chunks.length; i += EMBED_BATCH) {
      const batch = chunks.slice(i, i + EMBED_BATCH);

      // Embed enriched text to improve topical retrieval
      const enriched = batch.map((chunkTextValue) => {
        return `TITLE: ${title}\nSPEAKER: ${speaker_1}\nTOPICS: ${topics}\nCHUNK: ${chunkTextValue}`;
      });

      const embResp = await openai.embeddings.create({
        model: "RPRTHPB-text-embedding-3-small",
        input: enriched,
      });

      const vectors = embResp.data.map((d, j) => {
        const chunkIndex = i + j;
        const chunkTextValue = batch[j];

        return {
          id: `talk_${talk_id}_chunk_${chunkIndex}`,
          values: d.embedding,
          metadata: {
            talk_id,
            title,
            speaker_1,
            topics,
            url,
            chunk_index: chunkIndex,
            chunk: chunkTextValue,
          },
        };
      });

      await index.upsert(vectors);
      vectorsUpserted += vectors.length;
      console.log(`  Upserted ${vectors.length} vectors (total=${vectorsUpserted})`);
    }

    // Mark as ingested after success (even in FORCE_REEMBED)
    ingested.add(talk_id);
    saveIngestedSet(ingested);
  }

  console.log("Done.");

  if (DRY_RUN) {
    const embeddingCostUsd = (estTokens / 1_000_000) * 0.02; // rough estimate
    console.log(`[DRY_RUN] Estimated chunks: ${estChunks}`);
    console.log(`[DRY_RUN] Estimated embedding tokens: ${estTokens}`);
    console.log(`[DRY_RUN] Estimated embedding cost (USD): ${embeddingCostUsd.toFixed(6)}`);
    console.log(`[DRY_RUN] Talks processed: ${processedTalks}`);
  } else {
    console.log(`Talks processed: ${processedTalks}`);
    console.log(`Total vectors upserted: ${vectorsUpserted}`);
    console.log(`Local ingested state now contains: ${ingested.size} talks (${STATE_PATH}).`);
  }
}

main().catch((e) => {
  console.error("Ingestion failed:", e);
  process.exit(1);
});
