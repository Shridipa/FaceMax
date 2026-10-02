/**
 * POST /api/outfit-analysis/compare — Vercel serverless adapter.
 * Re-exports the outfit comparison handler from api/outfit-compare.ts
 * so both /api/outfit-analysis/compare and /api/outfit-compare work.
 */
export { default, maxDuration } from "../outfit-compare.js";
