/**
 * POST /api/outfit-analysis/wardrobe — Vercel serverless adapter.
 * Mode 3: analyse a user-selected wardrobe outfit (deterministic core + AI styling).
 * Core logic lives in api/_lib/outfit-core.ts (shared with server.ts).
 */
import type { VercelRequest, VercelResponse } from "@vercel/node";
import { runWardrobeAnalysis } from "../_lib/outfit-core.js";

export const maxDuration = 30;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ success: false, error: "Method not allowed" });

  // trim() so a whitespace-only value is treated as unconfigured, never sent to Gemini
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey) {
    return res.status(500).json({
      success: false,
      error: "Gemini API key not configured. Please set GEMINI_API_KEY in your environment.",
    });
  }

  const body = (req.body && typeof req.body === "object" ? req.body : {}) as Record<string, unknown>;
  const garments = Array.isArray(body.garments) ? body.garments : [];
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

  if (!profile.season) {
    return res.status(400).json({ success: false, error: "NO_PROFILE", message: "Complete your FaceMaxx colour analysis to receive styling recommendations tailored to you." });
  }

  try {
    const result = await runWardrobeAnalysis({ garments, profile, occasion, stylePreference }, apiKey);
    return res.status(200).json(result);
  } catch (error: unknown) {
    console.error("Outfit wardrobe error:", error);
    return res.status(500).json({
      success: false,
      error: "Outfit analysis failed. Please try again.",
      details: error instanceof Error ? error.message : String(error),
    });
  }
}
