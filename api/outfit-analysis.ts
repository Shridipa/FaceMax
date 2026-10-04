/**
 * POST /api/outfit-analysis — Vercel serverless adapter.
 *
 * The frontend never supplies colour-profile facts for personalisation
 * decisions beyond echoing back the profile it already received from this
 * app's own analysis flow; the deterministic suitability engine runs
 * server-side in api/outfit-core.ts against that saved profile.
 */
import type { VercelRequest, VercelResponse } from "@vercel/node";
import { runOutfitAnalysis } from "./_lib/outfit-core.js";

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
  const image = typeof body.image === "string" ? body.image : "";
  const profileRaw = (body.profile && typeof body.profile === "object" ? body.profile : {}) as Record<string, unknown>;
  const profile = {
    season: typeof profileRaw.season === "string" ? profileRaw.season : "",
    undertone: typeof profileRaw.undertone === "string" ? profileRaw.undertone : undefined,
    skin_depth: typeof profileRaw.skin_depth === "string" ? profileRaw.skin_depth : undefined,
    chroma: typeof profileRaw.chroma === "string" ? profileRaw.chroma : undefined,
    contrast: typeof profileRaw.contrast === "string" ? profileRaw.contrast : undefined,
  };
  const occasion = typeof body.occasion === "string" ? body.occasion.slice(0, 80) : undefined;
  const stylePreference = typeof body.stylePreference === "string" ? body.stylePreference.slice(0, 80) : undefined;

  if (!image) {
    return res.status(400).json({ success: false, error: "Image data (base64) is required." });
  }
  if (!profile.season) {
    return res.status(400).json({
      success: false,
      error: "NO_PROFILE",
      message: "Complete your FaceMaxx colour analysis to receive styling recommendations tailored to you.",
    });
  }

  try {
    const result = await runOutfitAnalysis({ image, profile, occasion, stylePreference }, apiKey);
    return res.status(200).json(result);
  } catch (error: unknown) {
    console.error("Outfit analysis error:", error);
    return res.status(500).json({
      success: false,
      error: "Outfit analysis failed. Please try again.",
      details: error instanceof Error ? error.message : String(error),
    });
  }
}
