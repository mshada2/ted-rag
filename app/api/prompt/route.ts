// app/api/prompt/route.ts
import { NextResponse } from "next/server";
import { answerQuestion, type Mode } from "@/lib/rag";

export const runtime = "nodejs";

function isMode(x: any): x is Mode {
  return x === "qa" || x === "title_list" || x === "title_speaker";
}

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({}));
    const question = String(body?.question ?? "").trim();
    const modeRaw = body?.mode;

    if (!question) {
      return NextResponse.json({ error: "Missing 'question' in JSON body." }, { status: 400 });
    }

    const mode: Mode | undefined = isMode(modeRaw) ? modeRaw : undefined;

    const result = await answerQuestion(question, mode);
    return NextResponse.json(result);
  } catch (e: any) {
    return NextResponse.json({ error: String(e?.message ?? e) }, { status: 500 });
  }
}
