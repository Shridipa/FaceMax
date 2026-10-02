/**
 * POST /api/outfit-describe — Vercel serverless adapter (legacy alias).
 * Re-exports the describe handler from api/outfit-analysis/describe.ts
 * so both /api/outfit-describe and /api/outfit-analysis/describe work identically.
 */
export { default, maxDuration } from "./outfit-analysis/describe.js";
