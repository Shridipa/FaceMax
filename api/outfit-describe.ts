/**
 * POST /api/outfit-analysis/describe — Vercel serverless adapter.
 * Mode 2: natural-language wardrobe description -> structured garments.
 * Core logic lives in api/outfit-core.ts (shared with server.ts).
 */
import type { VercelRequest, VercelResponse } from "@vercel/node";
import { runDescribeAnalysis } from "./outfit-core";

export const maxDuration = 30;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ success: false, error: "Method not allowed" });

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({
      success: false,
      error: "Gemini API key not configured. Please set GEMINI_API_KEY in your environment.",
    });
  }

  const body = (req.body && typeof req.body === "object" ? req.body : {}) as Record<string, unknown>;
  const description = typeof body.description === "string" ? body.description : "";
  const profileRaw = (body.profile && typeof body.profile === "object" ? body.profile : {}) as Record<string, unknown>;
  const profile = {
    season: typeof profileRaw.season === "string" ? profileRaw.season : "",
    undertone: typeof profileRaw.undertone === "string" ? profileRaw.undertone : undefined,
    skin_depth: typeof profileRaw.skin_depth === "string" ? profileRaw.skin_depth : undefined,
    chroma: typeof profileRaw.chroma === "string" ? profileRaw.chroma : undefined,
    contrast: typeof profileRaw.contrast === "string" ? profileRaw.contrast : undefined,
  };

  if (!description.trim()) {
    return res.status(400).json({ success: false, error: "EMPTY_DESCRIPTION", message: "Describe at least one garment." });
  }
  if (description.length > 2000) {
    return res.status(400).json({ success: false, error: "DESCRIPTION_TOO_LONG", message: "Please keep the description under 2000 characters." });
  }
  if (!profile.season) {
    return res.status(400).json({ success: false, error: "NO_PROFILE", message: "Complete your FaceMaxx colour analysis to receive styling recommendations tailored to you." });
  }

  try {
    const result = await runDescribeAnalysis({ description, profile }, apiKey);
    return res.status(200).json(result);
  } catch (error: unknown) {
    console.error("Outfit describe error:", error);
    return res.status(500).json({
      success: false,
      error: "Could not interpret the description. Please try again.",
      details: error instanceof Error ? error.message : String(error),
    });
  }
}
