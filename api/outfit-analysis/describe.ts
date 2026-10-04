/**
 * POST /api/outfit-analysis/describe — Vercel serverless adapter.
 * Mode 2: natural-language wardrobe description -> structured garments.
 *
 * ARCHITECTURE (local-first):
 *   Extraction is performed entirely by the deterministic dictionary parser
 *   (api/outfit-parser.ts via runDescribeAnalysis). This endpoint requires NO
 *   GEMINI_API_KEY, NO GOOGLE_API_KEY and makes NO external AI or network
 *   request — it works on Vercel with every AI-related env var unset.
 *
 * ROUTING NOTE (kept from the original fix):
 *   This file lives at api/outfit-analysis/describe.ts so Vercel deploys it at
 *   /api/outfit-analysis/describe — the URL the frontend already calls
 *   (api/outfit-describe.ts re-exports it as a legacy alias).
 */
import type { VercelRequest, VercelResponse } from "@vercel/node";
import { runDescribeAnalysis } from "../_lib/outfit-core.js";

export const maxDuration = 30;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST")
    return res.status(405).json({ success: false, error: "Method not allowed" });

  const body = (req.body && typeof req.body === "object" ? req.body : {}) as Record<
    string,
    unknown
  >;
  const description = typeof body.description === "string" ? body.description : "";
  const profileRaw = (body.profile && typeof body.profile === "object"
    ? body.profile
    : {}) as Record<string, unknown>;

  // Profile is optional: extraction is purely local; a saved season simply
  // enriches the later styling guidance.
  const profile =
    typeof profileRaw.season === "string" && profileRaw.season.trim()
      ? {
          season: profileRaw.season.trim(),
          undertone: typeof profileRaw.undertone === "string" ? profileRaw.undertone : undefined,
          skin_depth: typeof profileRaw.skin_depth === "string" ? profileRaw.skin_depth : undefined,
          chroma: typeof profileRaw.chroma === "string" ? profileRaw.chroma : undefined,
          contrast: typeof profileRaw.contrast === "string" ? profileRaw.contrast : undefined,
        }
      : undefined;

  if (!description.trim()) {
    return res.status(400).json({
      success: false,
      error: "EMPTY_DESCRIPTION",
      message: "Describe at least one garment.",
    });
  }
  if (description.length > 2000) {
    return res.status(400).json({
      success: false,
      error: "DESCRIPTION_TOO_LONG",
      message: "Please keep the description under 2000 characters.",
    });
  }

  try {
    const result = await runDescribeAnalysis({ description, profile });
    // HTTP 200 even when no garments found — the body carries success:false with
    // a meaningful message so the frontend can distinguish extraction failure
    // from a network or server error.
    return res.status(200).json(result);
  } catch (error: unknown) {
    console.error("Outfit describe error:", error);
    return res.status(500).json({
      success: false,
      error: "DESCRIBE_FAILED",
      message: "Could not interpret the description. Please try again.",
    });
  }
}
