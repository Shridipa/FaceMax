/**
 * POST /api/outfit-analysis/compare — Vercel serverless adapter.
 * Deterministic comparison of 2-4 outfit options (no AI involved).
 * Core logic lives in api/outfit-core.ts (shared with server.ts).
 */
import type { VercelRequest, VercelResponse } from "@vercel/node";
import { compareOutfits } from "./outfit-core.js";

export const maxDuration = 10;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ success: false, error: "Method not allowed" });

  const body = (req.body && typeof req.body === "object" ? req.body : {}) as Record<string, unknown>;
  const options = Array.isArray(body.options) ? body.options : [];
  const profileRaw = (body.profile && typeof body.profile === "object" ? body.profile : {}) as Record<string, unknown>;
  const profile = {
    season: typeof profileRaw.season === "string" ? profileRaw.season : "",
    undertone: typeof profileRaw.undertone === "string" ? profileRaw.undertone : undefined,
    skin_depth: typeof profileRaw.skin_depth === "string" ? profileRaw.skin_depth : undefined,
    chroma: typeof profileRaw.chroma === "string" ? profileRaw.chroma : undefined,
    contrast: typeof profileRaw.contrast === "string" ? profileRaw.contrast : undefined,
  };

  if (!profile.season) {
    return res.status(400).json({ success: false, error: "NO_PROFILE", message: "Complete your FaceMaxx colour analysis to compare outfits against your palette." });
  }
  if (options.length < 2) {
    return res.status(400).json({ success: false, error: "NOT_ENOUGH_OPTIONS", message: "Add at least two outfit options to compare." });
  }

  try {
    return res.status(200).json(compareOutfits(options, profile));
  } catch (error: unknown) {
    console.error("Outfit compare error:", error);
    return res.status(500).json({
      success: false,
      error: "Comparison failed. Please try again.",
      details: error instanceof Error ? error.message : String(error),
    });
  }
}
