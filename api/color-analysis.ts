import type { VercelRequest, VercelResponse } from "@vercel/node";

/**
 * Colour analysis requires the Python CV/ML backend (OpenCV, numpy, sklearn).
 * This cannot run on Vercel's Node.js serverless environment.
 *
 * This stub returns a graceful response so the frontend can display
 * a helpful message instead of a 404 error.
 */
export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  // Return a structured response indicating the Python backend is needed
  return res.status(200).json({
    ok: false,
    error: "Colour analysis requires the Python CV backend which is not available in this deployment. The season analysis feature works fully when running locally with `npm run dev`.",
    fallback: true,
    season: { primary: "", secondary: "", confidence: 0 },
    axes: {
      undertone: { prediction: "", probabilities: {}, model: "unavailable" },
      skin_depth: { prediction: "", probabilities: {}, model: "unavailable" },
      chroma: { prediction: "", probabilities: {}, model: "unavailable" },
      contrast: { prediction: "", probabilities: {}, model: "unavailable" },
    },
    features: {
      skin_L: 0, skin_a: 0, skin_b: 0, ita: 0,
      hair_L: 0, eye_L: 0, contrast_proxy: 0,
      brightness: "unavailable", skin_pixels: 0,
    },
  });
}
