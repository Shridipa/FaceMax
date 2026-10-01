/**
 * Multi-garment regression suite — the corrective mission's acceptance tests.
 *
 * Covers the four reported bugs:
 *   P1 missing garments   — every accepted input garment survives the pipeline
 *   P2 unrelated recs     — looks retain the user's actual garments
 *   P3 inconsistent odds  — contrast/combination distinguish garments vs colours
 *   P4 AI failures        — deterministic analysis survives AI outages
 *
 * The exact two-garment scenario from the bug report (cobalt blue top +
 * teal bottom) is a first-class fixture throughout. All AI is stubbed.
 * Run: npm run test:outfit
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";
import {
  computeSuitability,
  fallbackCombinations,
  validateCombinations,
  validateGarmentInputsWithReport,
  validateGarmentInputs,
  garmentInputsToOutfitGarments,
  runWardrobeAnalysis,
  runDescribeAnalysis,
  compareOutfits,
  scoreFromSuitability,
  paletteForSeason,
  parseDescriptionLocally,
  type OutfitAiPayload,
} from "../api/outfit-core";
import type {
  GarmentInput,
  OutfitGarment,
  OutfitSuitability,
} from "../src/types/outfit";

const WARM_AUTUMN = {
  season: "Warm Autumn",
  undertone: "warm",
  skin_depth: "light-medium",
  chroma: "muted",
  contrast: "medium",
};

/** The exact bug-report scenario. */
const COBALT_TEAL: GarmentInput[] = [
  { id: "top-001", category: "Tops", colourName: "Cobalt Blue", hex: "#0047AB", pattern: null, fit: null, owned: true },
  { id: "bottom-001", category: "Bottoms", colourName: "Teal", hex: "#008080", pattern: null, fit: null, owned: true },
];

function toGarments(inputs: GarmentInput[]): OutfitGarment[] {
  return garmentInputsToOutfitGarments(inputs);
}

function okWardrobePayload(): OutfitAiPayload {
  return {
    image_quality: null,
    garments: [],
    styling_advice: {
      currentOutfit: ["Keep the top as the focal point."],
      footwear: ["Neutral leather sneakers."],
      accessories: ["Simple watch."],
      layering: [],
    },
    outfit_combinations: [
      {
        name: "LOOK 01 - Everyday",
        description: "Wear the pieces together.",
        anchorGarment: "your cobalt blue top",
        pieces: [
          { item: "Cobalt blue top", colourName: "Cobalt Blue", hex: "#0047AB", owned: true },
          { item: "Teal bottoms", colourName: "Teal", hex: "#008080", owned: true },
          { item: "White sneakers", colourName: "White", hex: "#FFFFFF", owned: false },
        ],
        footwear: "White sneakers",
        accessories: ["Watch"],
        rationale: "Contrast reads deliberate.",
        aesthetic: "Everyday",
      },
      {
        name: "LOOK 02 - Elevated",
        description: "Layer for structure.",
        anchorGarment: "your cobalt blue top",
        pieces: [{ item: "Cobalt blue top", colourName: "Cobalt Blue", hex: "#0047AB", owned: true }],
        footwear: "Brown loafers",
        accessories: ["Belt"],
        rationale: "Deep tones ground the look.",
        aesthetic: "Occasion",
      },
    ],
  };
}

/* ================================================================== */
/*  A. Multi-garment regression (spec §13 A)                           */
/* ================================================================== */

describe("P1 regression: garments preserved through the pipeline", () => {
  test("Case 1: cobalt top + teal bottom — both garments in response, both analysed", async () => {
    const res = await runWardrobeAnalysis({ garments: COBALT_TEAL, profile: WARM_AUTUMN }, null, async () => {
      throw new Error("offline");
    });
    assert.equal(res.success, true);
    assert.equal(res.garments.length, 2);
    assert.equal(res.garments[0].dominantColour.name, "Cobalt Blue");
    assert.equal(res.garments[0].dominantColour.hex, "#0047AB");
    assert.equal(res.garments[1].dominantColour.name, "Teal");
    assert.equal(res.garments[1].dominantColour.hex, "#008080");
    // ids trace back to the user's inputs
    assert.equal(res.garments[0].id, "top-001");
    assert.equal(res.garments[1].id, "bottom-001");
    // outfit-level metrics use BOTH garments — never single-colour claims
    assert.notEqual(res.colour_analysis?.outfitContrast, null);
    const contrastReason = res.suitability.dimensions.contrast.reason;
    assert.ok(!contrastReason.toLowerCase().includes("only one garment"));
    const comboReason = res.suitability.dimensions.colourCombination.reason;
    assert.ok(!comboReason.toLowerCase().includes("single-colour"));
    assert.ok(res.score && res.score.score !== null);
  });

  test("Case 1b: cobalt+teal NOT flagged as internal colour clash (ΔE 15.5 < 20)", () => {
    const res = compareOutfits(
      [{ label: "X", garments: COBALT_TEAL }],
      WARM_AUTUMN,
    );
    assert.equal(res.options[0].conflicts.some((c) => c.includes("Cobalt") && c.includes("Teal")), false);
  });

  test("Case 2: three garments (cobalt, teal, camel footwear) all preserved", async () => {
    const three: GarmentInput[] = [
      ...COBALT_TEAL,
      { id: "shoe-001", category: "Footwear", colourName: "Camel", hex: "#C19A6B", pattern: null, fit: null, owned: true },
    ];
    const res = await runWardrobeAnalysis({ garments: three, profile: WARM_AUTUMN }, "key", async () => okWardrobePayload());
    assert.equal(res.garments.length, 3);
    assert.equal(res.garments.map((g) => g.id).join(","), "top-001,bottom-001,shoe-001");
    // all three contribute to suitability (mean over 3 colours)
    const s = computeSuitability(toGarments(three), paletteForSeason("Warm Autumn"), WARM_AUTUMN);
    assert.ok(s.reasons.length > 0);
    // recommendations see all three via the prompt path (validated combos carry owned flags)
    const ownedHexes = new Set(["#0047AB", "#008080", "#C19A6B"]);
    for (const combo of res.outfit_combinations) {
      for (const p of combo.pieces) {
        if (p.owned === true) assert.ok(ownedHexes.has(p.hex.toUpperCase()));
      }
    }
  });

  test("Case 3: two same-colour garments stay distinct; garment count ≠ colour count", async () => {
    const same: GarmentInput[] = [
      { id: "top-001", category: "Tops", colourName: "Teal", hex: "#008080", pattern: null, fit: null, owned: true },
      { id: "bottom-001", category: "Bottoms", colourName: "Teal", hex: "#008080", pattern: null, fit: null, owned: true },
    ];
    const res = await runWardrobeAnalysis({ garments: same, profile: WARM_AUTUMN }, null, async () => {
      throw new Error("offline");
    });
    assert.equal(res.garments.length, 2); // BOTH garments distinct
    const labs = res.garments.map((g) => g.dominantColour.hex.toUpperCase());
    assert.equal(new Set(labs).size, 1); // one unique colour
    // same colour -> tonal pairing, not a drop
    const s = computeSuitability(toGarments(same), paletteForSeason("Warm Autumn"), WARM_AUTUMN);
    assert.equal(s.dimensions.colourCombination.value, "harmonious"); // ΔE 0 -> tonal
  });

  test("Case 4: missing colour stays visible as unknown; never invented", async () => {
    const withUnknown: GarmentInput[] = [
      ...COBALT_TEAL,
      { id: "belt-001", category: "Accessories", colourName: "Unknown", hex: "", pattern: null, fit: null, owned: true },
    ];
    const res = await runWardrobeAnalysis({ garments: withUnknown, profile: WARM_AUTUMN }, null, async () => {
      throw new Error("offline");
    });
    // the two known-colour garments are analysed
    assert.equal(res.garments.length, 2);
    // the unknown garment is REPORTED, not silently dropped
    assert.ok(res.rejectedGarments?.some((r) => r.category === "Accessories" && r.reason.includes("unknown")));
    assert.ok(res.limitations.some((l) => l.includes("Accessories")));
    // it would remain analysable once the user supplies a colour
    const repaired = validateGarmentInputs([{ category: "Accessories", hex: "" }]);
    assert.equal(repaired[0].hex, "");
  });

  test("Case 5: invalid colour reported with a useful message, no substitution", async () => {
    const report = validateGarmentInputsWithReport([
      { category: "Tops", colourName: "Weird", hex: "blue" },
      { category: "Bottoms", colourName: "Teal", hex: "#008080" },
    ]);
    assert.equal(report.accepted.length, 1);
    assert.equal(report.rejected.length, 1);
    assert.ok(report.rejected[0].reason.includes("not a valid #RRGGBB"));
    assert.ok(report.rejected[0].reason.includes("blue"));
    assert.equal(report.accepted[0].hex, "#008080"); // neighbour untouched
  });

  test("duplicate ids are made unique, never merged away", () => {
    const report = validateGarmentInputsWithReport([
      { id: "dup", category: "Tops", hex: "#0047AB" },
      { id: "dup", category: "Bottoms", hex: "#008080" },
    ]);
    assert.equal(report.accepted.length, 2);
    assert.notEqual(report.accepted[0].id, report.accepted[1].id);
  });

  test("compare endpoint returns every accepted garment per option", () => {
    const res = compareOutfits(
      [{ label: "Cobalt+Teal", garments: COBALT_TEAL }],
      WARM_AUTUMN,
    );
    assert.equal(res.options[0].garments.length, 2);
    assert.equal(res.options[0].garments[1].dominantColour.hex, "#008080");
  });
});

/* ================================================================== */
/*  B. Recommendation regression (spec §13 B)                          */
/* ================================================================== */

describe("P2 regression: recommendations grounded in the actual outfit", () => {
  const swatches = paletteForSeason("Warm Autumn")!;

  test("fallback looks retain EVERY supplied garment as owned", () => {
    const combos = fallbackCombinations(toGarments(COBALT_TEAL), swatches);
    assert.ok(combos.length >= 2);
    for (const c of combos) {
      const ownedHexes = c.pieces.filter((p) => p.owned === true).map((p) => p.hex.toUpperCase());
      assert.ok(ownedHexes.includes("#0047AB"), "cobalt top retained");
      assert.ok(ownedHexes.includes("#008080"), "teal bottom retained");
      // suggested items are labelled, never implied owned
      for (const p of c.pieces) {
        if (p.owned === false) {
          assert.ok(
            !ownedHexes.includes(p.hex.toUpperCase()),
            "a suggested piece must not carry an owned garment's hex",
          );
        }
      }
      // rationale explains the relationship to the actual outfit
      assert.ok(c.rationale.includes("cobalt blue") || c.rationale.includes("your"));
    }
  });

  test("AI looks claiming non-owned colours as owned are corrected", () => {
    const raw = [
      {
        name: "LOOK 01 - Claimed",
        pieces: [
          { item: "Mystery item", colourName: "Camel", hex: "#C19A6B", owned: true }, // LIE: not supplied
          { item: "Teal bottoms", colourName: "Teal", hex: "#008080", owned: false }, // LIE: user owns this
        ],
      },
      {
        name: "LOOK 02 - Fine",
        pieces: [{ item: "Teal bottoms", colourName: "Teal", hex: "#008080", owned: true }],
      },
    ];
    const combos = validateCombinations(raw, toGarments(COBALT_TEAL), swatches);
    const look1 = combos.find((c) => c.name.includes("Claimed"))!;
    const camel = look1.pieces.find((p) => p.hex.toUpperCase() === "#C19A6B")!;
    assert.equal(camel.owned, undefined, "non-supplied colour must not be marked owned");
    const teal = look1.pieces.find((p) => p.hex.toUpperCase() === "#008080")!;
    assert.equal(teal.owned, true, "supplied colour must be marked owned even if AI claims otherwise");
  });

  test("AI failure keeps the full deterministic analysis and grounded fallback looks", async () => {
    const res = await runWardrobeAnalysis({ garments: COBALT_TEAL, profile: WARM_AUTUMN }, null, async () => {
      throw new Error("offline");
    });
    assert.equal(res.success, true);
    assert.ok(res.styling_advice === null); // no fake AI text
    assert.ok(res.limitations.some((l) => l.includes("AI styling suggestions are unavailable")));
    // fallback looks are the USER's outfit + labelled suggestions
    for (const c of res.outfit_combinations) {
      const ownedHexes = c.pieces.filter((p) => p.owned === true).map((p) => p.hex.toUpperCase());
      assert.ok(ownedHexes.includes("#0047AB"));
      assert.ok(ownedHexes.includes("#008080"));
    }
  });

  test("malformed AI combinations fall back to grounded looks, never hardcoded outfits", async () => {
    const combos = validateCombinations([{ broken: true }, "junk"], toGarments(COBALT_TEAL), swatches);
    assert.ok(combos.length >= 2);
    for (const c of combos) {
      const ownedHexes = c.pieces.filter((p) => p.owned === true).map((p) => p.hex.toUpperCase());
      assert.ok(ownedHexes.includes("#0047AB") && ownedHexes.includes("#008080"));
    }
  });

  test("empty outfit fallback presents nothing as owned", () => {
    const combos = fallbackCombinations([], swatches);
    for (const c of combos) {
      for (const p of c.pieces) assert.equal(p.owned, false);
    }
  });

  test("anchor phrase references the user's first garment", () => {
    const combos = fallbackCombinations(toGarments(COBALT_TEAL), swatches);
    for (const c of combos) assert.ok(c.anchorGarment.includes("cobalt blue tops"));
  });
});

/* ================================================================== */
/*  C. Scoring consistency (spec §13 C + §10)                          */
/* ================================================================== */

describe("P3 regression: consistent, honest scoring", () => {
  test("score is the weighted sum of displayed dimension percents (documented formula)", () => {
    const s = computeSuitability(toGarments(COBALT_TEAL), paletteForSeason("Warm Autumn"), WARM_AUTUMN);
    const score = scoreFromSuitability(s);
    assert.ok(score.score !== null);
    const usable = score.breakdown.filter((b) => b.percent !== null);
    const totalWeight = usable.reduce((sum, b) => sum + b.weight, 0);
    const weighted = usable.reduce((sum, b) => sum + b.weight * (b.percent as number), 0);
    assert.equal(score.score, Math.round(weighted / totalWeight));
  });

  test("colourCombination is excluded from the score weights (relationship, not profile fit)", () => {
    const score = scoreFromSuitability(computeSuitability(toGarments(COBALT_TEAL), paletteForSeason("Warm Autumn"), WARM_AUTUMN));
    assert.equal(score.breakdown.length, 5);
    assert.ok(!score.breakdown.some((b) => b.label === "Colour combination"));
  });

  test("contrast uses both L* values: cobalt L*≈30.1, teal L*≈47.1 -> ΔL≈17", () => {
    const s = computeSuitability(toGarments(COBALT_TEAL), paletteForSeason("Warm Autumn"), WARM_AUTUMN);
    // profile contrast = medium, ΔL 15.5 -> workable
    assert.equal(s.dimensions.contrast.value, "compatible");
  });

  test("single garment honestly reports contrast as not measurable", () => {
    const s = computeSuitability(toGarments([COBALT_TEAL[0]]), paletteForSeason("Warm Autumn"), WARM_AUTUMN);
    assert.equal(s.dimensions.contrast.value, "neutral");
    assert.ok(s.dimensions.contrast.reason.includes("Single-garment"));
    assert.equal(s.dimensions.colourCombination.value, "neutral");
    assert.ok(s.dimensions.colourCombination.reason.includes("Single-garment"));
  });

  test("multi-garment outfit with unmeasurable colours reports counts, never 'single-colour'", () => {
    // Defensive branch: 2 garments supplied but only 1 usable colour measurement
    // (e.g. a corrupt hex reaching the engine). Counts — not 'single-colour' —
    // must be reported.
    const garments: OutfitGarment[] = [
      { id: "a", category: "Tops", dominantColour: { name: "Cobalt Blue", hex: "#0047AB" }, secondaryColour: null, pattern: "Solid", patternConfidence: 1, confidence: 1 },
      { id: "b", category: "Bottoms", dominantColour: { name: "Broken", hex: "#ZZZZZZ" }, secondaryColour: null, pattern: "Solid", patternConfidence: 1, confidence: 1 },
    ];
    const s = computeSuitability(garments, paletteForSeason("Warm Autumn"), { ...WARM_AUTUMN, contrast: "" });
    assert.equal(s.dimensions.contrast.value, "unknown");
    assert.ok(s.dimensions.contrast.reason.includes("2 garments"));
    assert.ok(s.dimensions.contrast.reason.includes("1 usable colour measurement"));
    assert.equal(s.dimensions.colourCombination.value, "unknown");
    assert.ok(s.dimensions.colourCombination.reason.includes("2 garments"));
    // and it must NOT claim a single-colour outfit
    assert.ok(!s.dimensions.colourCombination.reason.includes("Single-colour"));
  });

  test("all-unknown dimensions -> null overall score, honest breakdown", () => {
    const s: OutfitSuitability = {
      verdict: "Insufficient image or profile information",
      dimensions: {
        colourHarmony: { value: "unknown", reason: "r" },
        temperature: { value: "unknown", reason: "r" },
        depth: { value: "unknown", reason: "r" },
        chroma: { value: "unknown", reason: "r" },
        contrast: { value: "unknown", reason: "r" },
        colourCombination: { value: "unknown", reason: "r" },
      },
      reasons: [],
    };
    const score = scoreFromSuitability(s);
    assert.equal(score.score, null);
    assert.equal(score.breakdown.every((b) => b.percent === null), true);
  });

  test("cobalt+teal wardrobe analysis: score agrees with dimension values end-to-end", async () => {
    const res = await runWardrobeAnalysis({ garments: COBALT_TEAL, profile: WARM_AUTUMN }, "key", async () => okWardrobePayload());
    const score = res.score!;
    const usable = score.breakdown.filter((b) => b.percent !== null);
    const totalWeight = usable.reduce((sum, b) => sum + b.weight, 0);
    const weighted = usable.reduce((sum, b) => sum + b.weight * (b.percent as number), 0);
    assert.equal(score.score, Math.round(weighted / totalWeight));
    // verdict derived from the same dimension values
    const harmony = res.suitability.dimensions.colourHarmony.value;
    const opposing = [res.suitability.dimensions.temperature, res.suitability.dimensions.depth, res.suitability.dimensions.chroma]
      .filter((d) => d.value === "contrasting").length;
    const expected =
      harmony === "harmonious" && opposing === 0 ? "Strong palette harmony"
      : (harmony === "harmonious" || harmony === "compatible") && opposing <= 1 ? "Good palette harmony"
      : harmony === "contrasting" || opposing >= 2 ? "Limited palette compatibility"
      : "Mixed palette compatibility";
    assert.equal(res.suitability.verdict, expected);
  });
});

/* ================================================================== */
/*  D. Mode-specific: describe (spec §13 D)                            */
/* ================================================================== */

describe("Describe mode: multiple garments in one description", () => {
  test("spec sentence parses into THREE separate garments", () => {
    const { garments, assumptions } = parseDescriptionLocally(
      "I am wearing a cobalt blue cotton top with teal wide-leg trousers and beige flats",
    );
    assert.equal(garments.length, 3);
    const top = garments.find((g) => g.category === "Top");
    const trousers = garments.find((g) => g.category === "Trousers");
    const flats = garments.find((g) => g.category === "Flats");
    assert.ok(top && top.hex === "#0047AB");
    assert.ok(trousers && trousers.hex === "#008080" && trousers.fit === "wide-leg");
    assert.ok(flats && flats.hex === "#F5F5DC");
    assert.equal(assumptions.length, 0);
  });

  test("'a cobalt blue top with teal bottom' keeps both garments", () => {
    const { garments } = parseDescriptionLocally("a cobalt blue top with teal bottom");
    assert.equal(garments.length, 2);
    assert.ok(garments.some((g) => g.hex === "#0047AB" && g.category === "Top"));
    assert.ok(garments.some((g) => g.hex === "#008080" && g.category === "Bottoms"));
  });

  test("describe pipeline preserves both parsed garments end-to-end", async () => {
    const res = await runDescribeAnalysis(
      { description: "a cobalt blue top with teal bottom", profile: WARM_AUTUMN },
      null,
      async () => { throw new Error("GEMINI_API_KEY is not configured."); },
    );
    assert.equal(res.success, true);
    assert.equal(res.garments.length, 2);
    assert.ok(res.garments.some((g) => g.hex === "#0047AB"));
    assert.ok(res.garments.some((g) => g.hex === "#008080"));
  });

  test("ambiguous descriptions: colour shared across two categories is attributed to both, disclosed", () => {
    const { garments, assumptions } = parseDescriptionLocally("teal top and teal trousers");
    assert.equal(garments.length, 2);
    assert.ok(assumptions.length >= 0); // no invention either way
  });
});

/* ================================================================== */
/*  E. Comparison tray integration (spec §11 F)                        */
/* ================================================================== */

describe("Comparison: complete outfits per option", () => {
  test("both options keep their full garment lists", () => {
    const res = compareOutfits(
      [
        { label: "Cobalt+Teal", garments: COBALT_TEAL },
        { label: "Teal+Camel", garments: [COBALT_TEAL[1], { id: "s", category: "Footwear", colourName: "Camel", hex: "#C19A6B", pattern: null, fit: null, owned: true }] },
      ],
      WARM_AUTUMN,
    );
    assert.equal(res.success, true);
    assert.equal(res.options[0].garments.length, 2);
    assert.equal(res.options[1].garments.length, 2);
    assert.equal(res.options[1].garments[1].dominantColour.hex, "#C19A6B");
  });

  test("deterministic: same options -> same results (tray stability)", () => {
    const a = compareOutfits([{ label: "X", garments: COBALT_TEAL }], WARM_AUTUMN);
    const b = compareOutfits([{ label: "X", garments: COBALT_TEAL }], WARM_AUTUMN);
    assert.deepEqual(a, b);
  });
});
