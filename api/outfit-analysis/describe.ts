/**
 * POST /api/outfit-analysis/describe — Vercel serverless adapter.
 * Mode 2: natural-language wardrobe description -> structured garments.
 *
 * ROOT CAUSE FIX:
 *   The old adapter lived at api/outfit-describe.ts which Vercel deploys as
 *   /api/outfit-describe.  The frontend called /api/outfit-analysis/describe,
 *   which did not exist on Vercel — every request got a 404, which the
 *   frontend's postJson helper converted to "Request failed. Please try again."
 *
 *   Moving the handler here (api/outfit-analysis/describe.ts) makes Vercel
 *   deploy it at the URL the frontend already expects.
 *
 * SECONDARY FIX:
 *   The old handler blocked offline extraction by returning 400 when
 *   profile.season was missing.  Offline garment extraction must work even if
 *   the user has not yet completed a colour analysis, so the season gate has
 *   been removed from this handler.  runDescribeAnalysis already handles the
 *   no-profile case gracefully (it extracts garments but omits season-specific
 *   styling tips).
 */
import type { VercelRequest, VercelResponse } from "@vercel/node";
import { runDescribeAnalysis } from "../outfit-core.js";

export const maxDuration = 30;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST")
    return res.status(405).json({ success: false, error: "Method not allowed" });

  // GEMINI_API_KEY is optional — offline extraction works without it.
  const apiKey = process.env.GEMINI_API_KEY || null;

  const body = (req.body && typeof req.body === "object" ? req.body : {}) as Record<
    string,
    unknown
  >;
  const description = typeof body.description === "string" ? body.description : "";
  const profileRaw = (body.profile && typeof body.profile === "object"
    ? body.profile
    : {}) as Record<string, unknown>;

  // Profile is optional for basic offline extraction; when present it enables
  // season-aware styling guidance from Gemini.
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
    const result = await runDescribeAnalysis({ description, profile }, apiKey);
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
