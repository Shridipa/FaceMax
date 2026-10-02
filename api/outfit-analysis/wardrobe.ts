/**
 * POST /api/outfit-analysis/wardrobe — Vercel serverless adapter.
 * Re-exports the wardrobe analysis handler from api/outfit-wardrobe.ts
 * so both /api/outfit-analysis/wardrobe and /api/outfit-wardrobe work.
 */
export { default, maxDuration } from "../outfit-wardrobe.js";
