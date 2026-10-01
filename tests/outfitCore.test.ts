/**
 * Outfit Studio core tests — deterministic engine only (no network, no AI key).
 *
 * The AI call itself is exercised through injected stub analysers so the
 * suite never depends on paid APIs (per the implementation spec).
 * Run: npm run test:outfit
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";
import {
  rgbToLab,
  hexToLab,
  deltaE76,
  chromaOf,
  nearestColourName,
  paletteForSeason,
  computeSuitability,
  validateGarments,
  validateAdvice,
  validateImageQuality,
  validateCombinations,
  fallbackCombinations,
  buildOutfitPrompt,
  runOutfitAnalysis,
  type OutfitAiPayload,
} from "../api/outfit-core";
import type { OutfitGarment } from "../src/types/outfit";

const DEEP_AUTUMN = "Deep Autumn";

function garment(hex: string, category = "Shirt", over: Partial<OutfitGarment> = {}): OutfitGarment {
  return {
    category,
    dominantColour: { name: nearestColourName(hex), hex },
    secondaryColour: null,
    pattern: "Solid",
    patternConfidence: 0.9,
    confidence: 0.9,
    ...over,
  };
}

/* -------------------------- colour math -------------------------- */

describe("CIELAB colour math", () => {
  test("white -> L*=100, a*=b*=0", () => {
    const lab = hexToLab("#FFFFFF")!;
    assert.ok(Math.abs(lab.L - 100) < 0.5);
    assert.ok(Math.abs(lab.a) < 0.5);
    assert.ok(Math.abs(lab.b) < 0.5);
  });

  test("black -> L*=0", () => {
    const lab = hexToLab("#000000")!;
    assert.ok(Math.abs(lab.L) < 0.5);
  });

  test("deltaE of identical colours is 0, white vs black is 100", () => {
    assert.equal(deltaE76("#ABCDEF", "#ABCDEF"), 0);
    assert.ok(Math.abs(deltaE76("#FFFFFF", "#000000")! - 100) < 0.5);
  });

  test("red is closer to dark red than to green (perceptual ordering)", () => {
    const toDarkRed = deltaE76("#FF0000", "#8B0000")!;
    const toGreen = deltaE76("#FF0000", "#00FF00")!;
    assert.ok(toDarkRed < toGreen);
  });

  test("chroma of a neutral is lower than a vivid colour", () => {
    assert.ok(chromaOf("#808080")! < chromaOf("#FF0000")!);
  });

  test("invalid hex returns null distances", () => {
    assert.equal(hexToLab("nope"), null);
    assert.equal(deltaE76("nope", "#FFFFFF"), null);
  });

  test("rgbToLab matches hexToLab", () => {
    const a = rgbToLab(255, 0, 0);
    const b = hexToLab("#FF0000")!;
    assert.ok(Math.abs(a.L - b.L) < 1e-9);
  });
});

describe("colour naming", () => {
  test("dark navy maps to Dark Navy", () => {
    assert.equal(nearestColourName("#1A1A3E"), "Dark Navy");
  });
  test("always returns a name", () => {
    assert.equal(typeof nearestColourName("#123456"), "string");
  });
});

describe("season palettes", () => {
  test("known season returns swatches", () => {
    const p = paletteForSeason(DEEP_AUTUMN)!;
    assert.ok(p.best.length > 0);
    assert.ok(p.neutrals.length > 0);
  });
  test("unknown season returns null", () => {
    assert.equal(paletteForSeason("Not A Season"), null);
  });
});

/* ------------------------ suitability engine ------------------------ */

describe("deterministic suitability", () => {
  const swatches = paletteForSeason(DEEP_AUTUMN)!;
  const profile = { season: DEEP_AUTUMN, undertone: "Warm", skin_depth: "Deep", chroma: "Muted", contrast: "High" };

  test("palette-exact outfit scores Strong palette harmony", () => {
    const g = swatches.best.slice(0, 2).map((s) => garment(s.hex));
    const s = computeSuitability(g, swatches, profile);
    assert.equal(s.verdict, "Strong palette harmony");
    assert.equal(s.dimensions.colourHarmony.value, "harmonious");
  });

  test("far-from-palette outfit never gets a harmony verdict above neutral", () => {
    const g = [garment("#00FF00"), garment("#FF00FF")]; // vivid spring-ish colours
    const s = computeSuitability(g, swatches, profile);
    assert.ok(["neutral", "contrasting"].includes(s.dimensions.colourHarmony.value));
    assert.ok(["Limited palette compatibility", "Mixed palette compatibility"].includes(s.verdict));
  });

  test("warm profile + cool garment flags temperature as contrasting", () => {
    const s = computeSuitability([garment("#0000FF")], swatches, { ...profile, undertone: "Warm" });
    assert.equal(s.dimensions.temperature.value, "contrasting");
  });

  test("warm profile + warm garment is compatible", () => {
    const s = computeSuitability([garment("#C19A6B")], swatches, { ...profile, undertone: "Warm" });
    assert.ok(["compatible", "neutral"].includes(s.dimensions.temperature.value));
  });

  test("deep profile + very light outfit flags depth", () => {
    const s = computeSuitability([garment("#FFFFFF")], swatches, { ...profile, skin_depth: "Deep" });
    assert.equal(s.dimensions.depth.value, "contrasting");
  });

  test("muted profile + vivid outfit flags chroma", () => {
    const s = computeSuitability([garment("#FF00FF")], swatches, { ...profile, chroma: "Muted" });
    assert.equal(s.dimensions.chroma.value, "contrasting");
  });

  test("missing axes yield unknown dimensions with reasons, never invented scores", () => {
    const s = computeSuitability([garment("#333333")], swatches, { season: DEEP_AUTUMN });
    assert.equal(s.dimensions.temperature.value, "unknown");
    assert.equal(s.dimensions.chroma.value, "unknown");
    assert.ok(s.dimensions.temperature.reason.length > 0);
  });

  test("no garments -> Insufficient image or profile information", () => {
    const s = computeSuitability([], swatches, profile);
    assert.equal(s.verdict, "Insufficient image or profile information");
  });

  test("null swatches -> Insufficient", () => {
    const s = computeSuitability([garment("#333333")], null, profile);
    assert.equal(s.verdict, "Insufficient image or profile information");
  });

  test("same input -> same verdict (determinism)", () => {
    const g = [garment("#6E1F2A"), garment("#3A2418")];
    const a = computeSuitability(g, swatches, profile);
    const b = computeSuitability(g.map((x) => ({ ...x })), swatches, profile);
    assert.equal(a.verdict, b.verdict);
    assert.deepEqual(a.dimensions, b.dimensions);
  });
});

/* ------------------------ AI output validation ------------------------ */

describe("AI payload validation", () => {
  test("garbage garments -> empty array", () => {
    assert.deepEqual(validateGarments("nope"), []);
    assert.deepEqual(validateGarments(42), []);
    assert.deepEqual(validateGarments([null, 3, "x"]), []);
  });

  test("garment fields are sanitised (hex fixed, names backfilled)", () => {
    const [g] = validateGarments([
      { category: "  Blazer  ", dominantColour: { hex: "zz#1a1a3e", name: "" }, confidence: 5, pattern: 7 },
    ]);
    assert.equal(g!.category, "Blazer");
    assert.equal(g!.dominantColour.hex, "#1A1A3E");
    assert.equal(g!.dominantColour.name, "Dark Navy");
    assert.equal(g!.confidence, 1); // clamped
    assert.equal(g!.pattern, "Solid"); // non-string -> default
  });

  test("advice validation filters non-strings and drops empty sets", () => {
    assert.deepEqual(validateAdvice({ footwear: ["white sneakers", 5, ""], currentOutfit: [] })!.footwear, ["white sneakers"]);
    assert.equal(validateAdvice({ footwear: [] }), null);
    assert.equal(validateAdvice("x"), null);
  });

  test("image quality: garbage -> null; numbers kept; flags filtered", () => {
    assert.equal(validateImageQuality(null), null);
    const q = validateImageQuality({ brightness: 0.4, contrast: "x", flags: ["dim light", 42] })!;
    assert.equal(q.brightness, 0.4);
    assert.equal(q.contrast, 0.5); // defaulted
    assert.deepEqual(q.flags, ["dim light"]);
  });

  test("malformed combinations -> deterministic fallbacks (2, distinct)", () => {
    const swatches = paletteForSeason(DEEP_AUTUMN)!;
    const combos = validateCombinations([{ broken: true }, "junk"], [garment("#6E1F2A", "Blazer")], swatches);
    assert.ok(combos.length >= 2);
    assert.notEqual(combos[0].name, combos[1].name);
    for (const c of combos) assert.ok(c.anchorGarment.includes("blazer"));
  });

  test("valid combinations pass through with sanitised hex; duplicates dropped", () => {
    const swatches = paletteForSeason(DEEP_AUTUMN)!;
    const raw = [
      { name: "LOOK 01 - Everyday", pieces: [{ item: "Trousers", colourName: "Camel", hex: "#C19A6B" }] },
      { name: "LOOK 01 - Everyday", pieces: [{ item: "Dup", hex: "#FFF000" }] }, // duplicate name
      { name: "LOOK 02 - Elevated", pieces: [{ item: "Layer", colourName: "Forest", hex: "#234B35" }] },
    ];
    const combos = validateCombinations(raw, [garment("#6E1F2A", "Shirt")], swatches);
    assert.equal(combos.length, 2);
    assert.equal(combos[0].pieces[0].hex, "#C19A6B");
  });

  test("fallback combinations use the season palette and keep the anchor", () => {
    const swatches = paletteForSeason(DEEP_AUTUMN)!;
    const combos = fallbackCombinations([garment("#6E1F2A", "Shirt")], swatches);
    assert.equal(combos.length, 2);
    const paletteHexes = new Set([...swatches.best, ...swatches.neutrals].map((s) => s.hex));
    for (const c of combos) {
      for (const p of c.palette) assert.ok(paletteHexes.has(p.hex));
    }
  });
});

describe("AI prompt", () => {
  test("embeds the real profile and the schema contract", () => {
    const prompt = buildOutfitPrompt({ season: DEEP_AUTUMN, undertone: "Warm" }, "work", "minimal");
    assert.ok(prompt.includes('"season":"Deep Autumn"') || prompt.includes("Deep Autumn"));
    assert.ok(prompt.includes("outfit_combinations"));
    assert.ok(prompt.includes("work"));
    assert.ok(prompt.includes("minimal"));
  });
});

/* ------------------------ orchestration (stub AI) ------------------------ */

const fullAiPayload = (): OutfitAiPayload => ({
  image_quality: { brightness: 0.6, contrast: 0.5, saturation: 0.5, sharpness_proxy: 0.8, flags: [] },
  garments: [
    { category: "Shirt", dominantColour: { name: "Burgundy", hex: "#6E1F2A" }, secondaryColour: null, pattern: "Solid", patternConfidence: 0.8, confidence: 0.9 },
    { category: "Jeans", dominantColour: { name: "Dark Navy", hex: "#1A1A3E" }, secondaryColour: null, pattern: "Solid", patternConfidence: 0.7, confidence: 0.8 },
  ],
  styling_advice: { currentOutfit: ["Keep the shirt as the focal point"], footwear: ["Dark leather sneakers"], accessories: ["A simple belt"], layering: [] },
  outfit_combinations: [
    { name: "LOOK 01 - Relaxed everyday", description: "d", anchorGarment: "your burgundy shirt", pieces: [{ item: "Trousers", colourName: "Camel", hex: "#C19A6B" }], footwear: "Sneakers", accessories: ["Watch"], rationale: "r", aesthetic: "Everyday" },
    { name: "LOOK 02 - Refined", description: "d", anchorGarment: "your burgundy shirt", pieces: [{ item: "Blazer", colourName: "Forest Green", hex: "#234B35" }], footwear: "Loafers", accessories: [], rationale: "r", aesthetic: "Occasion" },
    { name: "LOOK 03 - Tonal", description: "d", anchorGarment: "your burgundy shirt", pieces: [{ item: "Chinos", colourName: "Burgundy", hex: "#6E1F2A" }], footwear: "Espadrilles", accessories: [], rationale: "r", aesthetic: "Tonal" },
  ],
  limitations: [],
});

const PROFILE = { season: DEEP_AUTUMN, undertone: "Warm", skin_depth: "Deep", chroma: "Muted", contrast: "High" };

describe("runOutfitAnalysis orchestration", () => {
  test("no profile -> NO_PROFILE without calling the AI", async () => {
    let called = false;
    const res = await runOutfitAnalysis({ image: "data:image/jpeg;base64,x", profile: { season: "" } }, "key", async () => {
      called = true;
      return fullAiPayload();
    });
    assert.equal(called, false);
    assert.equal(res.success, false);
    assert.equal(res.error, "NO_PROFILE");
  });

  test("happy path: deterministic suitability + validated combinations", async () => {
    const res = await runOutfitAnalysis({ image: "data:image/jpeg;base64,x", profile: PROFILE }, "key", async () => fullAiPayload());
    assert.equal(res.success, true);
    assert.equal(res.garments.length, 2);
    assert.equal(res.outfit_combinations.length, 3);
    // Suitability is computed by THIS module, not echoed from the AI:
    assert.ok(res.suitability.reasons.length > 0);
    assert.equal(res.profile_used.source, "session_profile");
    assert.equal(res.profile_used.season, DEEP_AUTUMN);
    // every combination keeps the anchor garment reference
    for (const c of res.outfit_combinations) assert.ok(c.anchorGarment.length > 0);
  });

  test("AI garbage -> honest degradation, no fabricated results", async () => {
    const res = await runOutfitAnalysis({ image: "x", profile: PROFILE }, "key", async () => ({}) as OutfitAiPayload);
    assert.equal(res.success, false);
    assert.equal(res.garments.length, 0);
    assert.equal(res.suitability.verdict, "Insufficient image or profile information");
    assert.ok(res.limitations.some((l) => l.toLowerCase().includes("no garment")));
  });

  test("AI failure -> AI_SERVICE_FAILED with clear error", async () => {
    const res = await runOutfitAnalysis({ image: "x", profile: PROFILE }, "key", async () => {
      throw new Error("upstream timeout");
    });
    assert.equal(res.success, false);
    assert.equal(res.error, "AI_SERVICE_FAILED");
    assert.ok(res.suitability.reasons[0].includes("timeout"));
  });

  test("determinism: identical stub input -> identical suitability", async () => {
    const a = await runOutfitAnalysis({ image: "x", profile: PROFILE }, "key", async () => fullAiPayload());
    const b = await runOutfitAnalysis({ image: "x", profile: PROFILE }, "key", async () => fullAiPayload());
    assert.deepEqual(a.suitability, b.suitability);
    assert.deepEqual(a.colour_analysis, b.colour_analysis);
  });
});
