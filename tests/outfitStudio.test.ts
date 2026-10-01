/**
 * Outfit Studio expansion tests — score engine, garment inputs, describe /
 * wardrobe / compare orchestrators. All AI is stubbed; no network, no key.
 * Run: npm run test:outfit
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";
import {
  scoreFromSuitability,
  outfitConflicts,
  validateGarmentInputs,
  garmentInputsToOutfitGarments,
  runDescribeAnalysis,
  runWardrobeAnalysis,
  compareOutfits,
  buildWardrobePrompt,
  paletteForSeason,
  computeSuitability,
  parseDescriptionLocally,
  type OutfitAiPayload,
} from "../api/outfit-core";
import type { OutfitSuitability, GarmentInput } from "../src/types/outfit";

const DEEP_WINTER_PROFILE = {
  season: "Deep Winter",
  undertone: "cool",
  skin_depth: "deep",
  chroma: "bright",
  contrast: "high",
};

function garment(hex: string, category = "Shirt") {
  return {
    category,
    dominantColour: { name: "Test", hex },
    secondaryColour: null,
    pattern: "Solid",
    patternConfidence: 0.9,
    confidence: 0.9,
  };
}

function dim(value: OutfitSuitability["dimensions"]["colourHarmony"]["value"], reason = "test"): OutfitSuitability["dimensions"]["colourHarmony"] {
  return { value, reason };
}

function suitability(overrides: Partial<Record<string, ReturnType<typeof dim>>> = {}): OutfitSuitability {
  return {
    verdict: "Good palette harmony",
    dimensions: {
      colourHarmony: overrides.colourHarmony ?? dim("harmonious"),
      temperature: overrides.temperature ?? dim("compatible"),
      depth: overrides.depth ?? dim("compatible"),
      chroma: overrides.chroma ?? dim("compatible"),
      contrast: overrides.contrast ?? dim("compatible"),
      colourCombination: overrides.colourCombination ?? dim("compatible"),
    },
    reasons: ["test"],
  };
}

/* ================================================================== */
/*  Deterministic score                                                */
/* ================================================================== */

describe("scoreFromSuitability", () => {
  test("all dimensions known -> weighted 0-100 score", () => {
    const s = scoreFromSuitability(suitability());
    // 0.3*100 + 0.2*80*4 dims... harmony=100, others=80
    // (0.3*100 + 0.8*80*... ) weights: .3 + .2 + .2 + .15 + .15 = 1.0
    // = 30 + 16 + 16 + 12 + 12 = 86
    assert.equal(s.score, 86);
    assert.equal(s.breakdown.length, 5);
    assert.ok(s.note.includes("not scientifically validated"));
  });

  test("unknown dimensions are excluded and weights renormalised", () => {
    const s = scoreFromSuitability(suitability({
      temperature: dim("unknown"),
      depth: dim("unknown"),
      chroma: dim("unknown"),
      contrast: dim("unknown"),
    }));
    // only harmony known: 100
    assert.equal(s.score, 100);
    const unknown = s.breakdown.filter((b) => b.percent === null);
    assert.equal(unknown.length, 4);
  });

  test("contrasting harmony lowers the score", () => {
    const high = scoreFromSuitability(suitability()).score as number;
    const low = scoreFromSuitability(suitability({ colourHarmony: dim("contrasting") })).score as number;
    assert.ok(low < high);
  });

  test("all dimensions unknown -> honest null score", () => {
    const s = scoreFromSuitability(suitability({
      colourHarmony: dim("unknown"), temperature: dim("unknown"), depth: dim("unknown"),
      chroma: dim("unknown"), contrast: dim("unknown"),
    }));
    assert.equal(s.score, null);
  });
});

/* ================================================================== */
/*  Outfit-internal conflicts                                          */
/* ================================================================== */

describe("outfitConflicts", () => {
  test("intermediate-distance pair is flagged", () => {
    // dark navy vs burgundy: \u0394E76 \u2248 45, inside the awkward 20-55 band
    const conflicts = outfitConflicts([garment("#1A1A3E", "Shirt"), garment("#6E1F2A", "Trousers")]);
    assert.ok(conflicts.length >= 1);
    assert.ok(conflicts[0].includes("clash"));
  });

  test("tonal pair is not flagged", () => {
    // near-identical navies: \u0394E76 well below 20
    const conflicts = outfitConflicts([garment("#1A1A3E", "Shirt"), garment("#1C1C40", "Trousers")]);
    assert.equal(conflicts.length, 0);
  });

  test("single garment has no conflicts", () => {
    assert.equal(outfitConflicts([garment("#FF0000")]).length, 0);
  });
});

/* ================================================================== */
/*  Garment inputs                                                     */
/* ================================================================== */

describe("validateGarmentInputs", () => {
  test("keeps valid entries and backfills colour names", () => {
    const out = validateGarmentInputs([
      { category: "Shirt", hex: "#1A1A3E" },
      { category: "Trousers", hex: "#F5F5DC", colourName: "Beige", pattern: "Striped", fit: "wide-leg" },
    ]);
    assert.equal(out.length, 2);
    assert.equal(out[0].colourName, "Dark Navy");
    assert.equal(out[0].owned, true);
    assert.equal(out[1].pattern, "Striped");
    assert.equal(out[1].fit, "wide-leg");
  });

  test("drops entries without category or valid hex", () => {
    const out = validateGarmentInputs([
      { category: "", hex: "#000000" },
      { category: "Shirt", hex: "blue" },
      { category: "Shirt", hex: "#0000FF" },
    ]);
    assert.equal(out.length, 1);
    assert.equal(out[0].hex, "#0000FF");
  });

  test("caps at 12 garments", () => {
    const many = Array.from({ length: 20 }, (_, i) => ({ category: `Item${i}`, hex: "#0000FF" }));
    assert.equal(validateGarmentInputs(many).length, 12);
  });

  test("garmentInputsToOutfitGarments marks user colours as ground truth", () => {
    const inputs: GarmentInput[] = validateGarmentInputs([{ category: "Shirt", hex: "#0000FF" }]);
    const garments = garmentInputsToOutfitGarments(inputs);
    assert.equal(garments[0].confidence, 1);
    assert.equal(garments[0].pattern, "Solid");
  });
});

/* ================================================================== */
/*  Mode 2 — describe clothes                                          */
/* ================================================================== */

describe("runDescribeAnalysis", () => {
  test("no profile -> NO_PROFILE without calling the interpreter", async () => {
    let called = false;
    const res = await runDescribeAnalysis({ description: "a black shirt", profile: undefined as never }, null, async () => {
      called = true;
      return { garments: [] };
    });
    assert.equal(res.success, false);
    assert.equal(res.error, "NO_PROFILE");
    assert.equal(called, false);
  });

  test("empty description -> EMPTY_DESCRIPTION", async () => {
    const res = await runDescribeAnalysis({ description: "  ", profile: DEEP_WINTER_PROFILE }, "key");
    assert.equal(res.error, "EMPTY_DESCRIPTION");
  });

  test("interpreter extracts and sanitises garments + assumptions", async () => {
    const res = await runDescribeAnalysis(
      { description: "black oversized shirt and beige trousers", profile: DEEP_WINTER_PROFILE },
      "key",
      async () => ({
        garments: [
          { category: "Shirt", hex: "#1A1A3E", colourName: "Dark Navy", fit: "oversized" },
          { category: "Trousers", hex: "not-a-hex" }, // dropped
          { category: "Sweater", hex: "#F5F5DC", colourName: "   " },
          "garbage",
        ],
        assumptions: ["assumed dark green means forest green", "", 42],
      }),
    );
    assert.equal(res.success, true);
    assert.equal(res.garments.length, 2);
    assert.equal(res.garments[0].fit, "oversized");
    assert.equal(res.garments[1].colourName.length > 0, true); // backfilled
    assert.deepEqual(res.assumptions, ["assumed dark green means forest green"]);
  });

  test("interpreter failure + parsable text -> offline fallback discloses the AI error", async () => {
    const res = await runDescribeAnalysis(
      { description: "a black shirt", profile: DEEP_WINTER_PROFILE },
      null,
      async () => { throw new Error("boom"); },
    );
    assert.equal(res.success, true); // offline dictionary saved it
    assert.equal(res.garments[0].colourName, "Black");
    assert.ok(res.assumptions.some((a) => a.includes("boom")));
  });

  test("interpreter failure + unparsable text -> honest AI_SERVICE_FAILED", async () => {
    const res = await runDescribeAnalysis(
      { description: "my comfy favourites", profile: DEEP_WINTER_PROFILE },
      null,
      async () => { throw new Error("boom"); },
    );
    assert.equal(res.success, false);
    assert.ok(res.limitations.some((l) => l.includes("boom")));
    assert.ok(res.limitations.some((l) => l.includes("GEMINI_API_KEY")));
  });

  test("no garments found -> success:false with guidance", async () => {
    const res = await runDescribeAnalysis(
      { description: "something nice", profile: DEEP_WINTER_PROFILE },
      "key",
      async () => ({ garments: ["nope"] }),
    );
    assert.equal(res.success, false);
    assert.ok(res.limitations[0].includes("No recognisable garment"));
  });

  test("AI unavailable + parsable description -> offline dictionary fallback with disclosure", async () => {
    const res = await runDescribeAnalysis(
      { description: "I have a yellow cardigan and hot pink jeans", profile: DEEP_WINTER_PROFILE },
      null,
      async () => { throw new Error("GEMINI_API_KEY is not configured."); },
    );
    assert.equal(res.success, true);
    assert.equal(res.garments.length, 2);
    const cardigan = res.garments.find((g) => g.category === "Cardigan");
    const jeans = res.garments.find((g) => g.category === "Jeans");
    assert.ok(cardigan && cardigan.hex === "#FFD700");
    assert.ok(jeans && jeans.hex === "#FF69B4");
    assert.ok(res.assumptions.some((a) => a.includes("built-in colour dictionary")));
  });

  test("AI unavailable + unparsable description -> honest error naming the key", async () => {
    const res = await runDescribeAnalysis(
      { description: "my favourite comfy things", profile: DEEP_WINTER_PROFILE },
      null,
      async () => { throw new Error("GEMINI_API_KEY is not configured."); },
    );
    assert.equal(res.success, false);
    assert.ok(res.limitations.some((l) => l.includes("GEMINI_API_KEY")));
  });

  test("AI found nothing + parsable description -> offline fallback", async () => {
    const res = await runDescribeAnalysis(
      { description: "a striped navy shirt", profile: DEEP_WINTER_PROFILE },
      "key",
      async () => ({ garments: [] }),
    );
    assert.equal(res.success, true);
    assert.equal(res.garments[0].category, "Shirt");
    assert.equal(res.garments[0].pattern, "Striped");
  });
});

describe("parseDescriptionLocally (offline dictionary)", () => {
  test("yellow cardigan + hot pink jeans", () => {
    const { garments } = parseDescriptionLocally("I have a yellow cardigan and hot pink jeans");
    assert.equal(garments.length, 2);
    assert.equal(garments[0].category, "Cardigan");
    assert.equal(garments[0].colourName, "Yellow");
    assert.equal(garments[0].hex, "#FFD700");
    assert.equal(garments[1].category, "Jeans");
    assert.equal(garments[1].hex, "#FF69B4");
    assert.equal(garments[1].colourName, "Hot Pink");
  });

  test("multi-word colour beats shorter contained phrase (hot pink, not pink)", () => {
    const { garments } = parseDescriptionLocally("hot pink jeans");
    assert.equal(garments[0].colourName, "Hot Pink");
  });

  test("pattern and fit captured", () => {
    const { garments } = parseDescriptionLocally("oversized striped navy shirt");
    assert.equal(garments[0].fit, "oversized");
    assert.equal(garments[0].pattern, "Striped");
  });

  test("garment without colour stays visible with unknown colour, never invented", () => {
    const { garments, assumptions } = parseDescriptionLocally("a nice cardigan");
    assert.equal(garments.length, 1);
    assert.equal(garments[0].category, "Cardigan");
    assert.equal(garments[0].hex, ""); // unknown, never defaulted
    assert.equal(garments[0].colourName, "Unknown");
    assert.ok(assumptions.some((a) => a.includes("could not tell the colour")));
  });

  test("indian wardrobe vocabulary (kurti, saree)", () => {
    const { garments } = parseDescriptionLocally("a maroon kurti with gold embroidery, a green saree");
    assert.ok(garments.some((g) => g.category === "Kurti"));
    assert.ok(garments.some((g) => g.category === "Saree"));
  });

  test("dedupes identical items", () => {
    const { garments } = parseDescriptionLocally("a black shirt, black shirt");
    assert.equal(garments.length, 1);
  });
});

/* ================================================================== */
/*  Mode 3 — wardrobe analysis                                         */
/* ================================================================== */

const WARDROBE_GARMENTS = [
  { category: "Shirt", colourName: "Dark Navy", hex: "#1A1A3E" },
  { category: "Trousers", colourName: "Cream", hex: "#FFFDD0" },
];

function okPayload(): OutfitAiPayload {
  return {
    image_quality: null,
    garments: [],
    styling_advice: {
      currentOutfit: ["Keep the shirt as the focal point."],
      footwear: ["White minimal sneakers."],
      accessories: ["Slim silver watch."],
      layering: [],
    },
    outfit_combinations: [
      {
        name: "LOOK 01 - Everyday",
        description: "Practical everyday look.",
        anchorGarment: "the dark navy shirt",
        pieces: [
          { item: "Dark navy shirt", colourName: "Dark Navy", hex: "#1A1A3E", owned: true },
          { item: "Cream trousers", colourName: "Cream", hex: "#FFFDD0", owned: true },
          { item: "White sneakers", colourName: "White", hex: "#FFFFFF", owned: false },
        ],
        footwear: "White minimal sneakers",
        accessories: ["Slim silver watch"],
        rationale: "High contrast suits a Deep Winter profile.",
        aesthetic: "Everyday",
      },
      {
        name: "LOOK 02 - Elevated",
        description: "Polished alternative.",
        anchorGarment: "the dark navy shirt",
        pieces: [{ item: "Dark navy shirt", colourName: "Dark Navy", hex: "#1A1A3E", owned: true }],
        footwear: "Black loafers",
        accessories: ["Leather belt"],
        rationale: "Deep, cool tones match the palette.",
        aesthetic: "Occasion",
      },
    ],
  };
}

describe("runWardrobeAnalysis", () => {
  test("no profile -> NO_PROFILE without calling the analyser", async () => {
    let called = false;
    const res = await runWardrobeAnalysis(
      { garments: WARDROBE_GARMENTS, profile: undefined as never },
      null,
      async () => { called = true; return okPayload(); },
    );
    assert.equal(res.error, "NO_PROFILE");
    assert.equal(called, false);
  });

  test("no valid garments -> NO_GARMENTS", async () => {
    const res = await runWardrobeAnalysis({ garments: [{ category: "", colourName: "x", hex: "#0000FF" }], profile: DEEP_WINTER_PROFILE }, "key");
    assert.equal(res.error, "NO_GARMENTS");
  });

  test("AI failure still returns a SUCCESS with deterministic verdict + fallback looks", async () => {
    const res = await runWardrobeAnalysis(
      { garments: WARDROBE_GARMENTS, profile: DEEP_WINTER_PROFILE },
      null,
      async () => { throw new Error("offline"); },
    );
    assert.equal(res.success, true);
    assert.equal(res.input_mode, "wardrobe");
    assert.ok(res.suitability.verdict.length > 0);
    assert.ok(res.score && res.score.score !== null);
    assert.equal(res.styling_advice, null);
    assert.ok(res.outfit_combinations.length >= 2); // deterministic fallbacks
    assert.ok(res.limitations.some((l) => l.includes("AI styling suggestions are unavailable")));
  });

  test("happy path: deterministic core + validated AI advice/combos", async () => {
    const res = await runWardrobeAnalysis(
      { garments: WARDROBE_GARMENTS, profile: DEEP_WINTER_PROFILE },
      "key",
      async () => okPayload(),
    );
    assert.equal(res.success, true);
    assert.equal(res.garments.length, 2);
    assert.equal(res.garments[0].confidence, 1);
    assert.ok(res.styling_advice);
    assert.equal(res.outfit_combinations.length, 2);
    // owned flag preserved through validation
    const combo = res.outfit_combinations[0];
    assert.equal(combo.pieces[0].owned, true);
    assert.equal(combo.pieces[2].owned, false);
  });

  test("prompt embeds owned garments and profile, never asks AI to invent colours", async () => {
    const prompt = buildWardrobePrompt(DEEP_WINTER_PROFILE, validateGarmentInputs(WARDROBE_GARMENTS), "Office", "minimal");
    assert.ok(prompt.includes("Dark Navy"));
    assert.ok(prompt.includes("[OWNED]"));
    assert.ok(prompt.includes("Deep Winter"));
    assert.ok(prompt.includes("Office"));
    assert.ok(prompt.includes("owned\": false"));
  });
});

/* ================================================================== */
/*  Comparison mode                                                    */
/* ================================================================== */

describe("compareOutfits", () => {
  test("no profile -> NO_PROFILE", () => {
    const res = compareOutfits([], undefined as never);
    assert.equal(res.error, "NO_PROFILE");
  });

  test("palette-matching option scores higher than clashing option", () => {
    const swatches = paletteForSeason("Deep Winter");
    assert.ok(swatches);
    const good = swatches!.best[0]?.hex ?? "#1A1A3E";
    const res = compareOutfits(
      [
        { label: "Option A", garments: [{ category: "Shirt", colourName: "Good", hex: good }, { category: "Trousers", colourName: "N2", hex: swatches!.neutrals[0]?.hex ?? "#FFFFFF" }] },
        { label: "Option B", garments: [{ category: "Shirt", colourName: "Loud", hex: "#00FF00" }, { category: "Trousers", colourName: "Warm", hex: "#FFA500" }] },
      ],
      DEEP_WINTER_PROFILE,
    );
    assert.equal(res.success, true);
    assert.equal(res.options.length, 2);
    const [a, b] = res.options;
    assert.ok((a.score.score as number) > (b.score.score as number));
    assert.ok(res.summary!.includes("Option A"));
    assert.ok(res.summary!.includes("not which look is best"));
  });

  test("honest fallback summary when scores are not computable", () => {
    const res = compareOutfits(
      [{ label: "A", garments: [] }, { label: "B", garments: [] }],
      DEEP_WINTER_PROFILE,
    );
    assert.ok(res.summary!.includes("Not enough measurable colour data"));
    assert.equal(res.success, true);
  });

  test("more than 4 options are truncated and invalid garments dropped", () => {
    const res = compareOutfits(
      [
        { label: "A", garments: [{ category: "Shirt", colourName: "N", hex: "#1A1A3E" }] },
        { label: "B", garments: [{ category: "Shirt", colourName: "N", hex: "#1A1A3E" }] },
        { label: "C", garments: [{ category: "Shirt", colourName: "N", hex: "#1A1A3E" }] },
        { label: "D", garments: [{ category: "Shirt", colourName: "N", hex: "#1A1A3E" }] },
        { label: "E", garments: [{ category: "Shirt", colourName: "N", hex: "#1A1A3E" }] },
      ],
      DEEP_WINTER_PROFILE,
    );
    assert.equal(res.options.length, 4);
  });

  test("deterministic: identical inputs give identical outputs", () => {
    const options = [
      { label: "A", garments: [{ category: "Shirt", colourName: "Navy", hex: "#1A1A3E" }] },
      { label: "B", garments: [{ category: "Trousers", colourName: "Cream", hex: "#FFFDD0" }] },
    ];
    assert.deepEqual(compareOutfits(options, DEEP_WINTER_PROFILE), compareOutfits(options, DEEP_WINTER_PROFILE));
  });

  test("suitability integrates with the deterministic engine end-to-end", () => {
    const garments = garmentInputsToOutfitGarments(
      validateGarmentInputs([{ category: "Shirt", colourName: "Dark Navy", hex: "#1A1A3E" }]),
    );
    const s = computeSuitability(garments, paletteForSeason("Deep Winter"), DEEP_WINTER_PROFILE);
    assert.ok(["Strong palette harmony", "Good palette harmony"].includes(s.verdict));
  });
});
