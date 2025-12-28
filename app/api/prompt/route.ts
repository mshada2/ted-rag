// app/api/prompt/route.ts
import { NextResponse } from "next/server";
import { answerQuestion } from "@/lib/rag";

export const runtime = "nodejs";

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({}));
    const question = String(body?.question ?? "").trim();
    if (!question) {
      return NextResponse.json({ error: "Missing 'question' in JSON body." }, { status: 400 });
    }

    const result = await answerQuestion(question);
    return NextResponse.json(result);
  } catch (err: any) {
    return NextResponse.json(
      { error: "Server error", details: String(err?.message ?? err) },
      { status: 500 }
    );
  }
}
