/**
 * Describe Clothes — local deterministic parser acceptance tests.
 *
 * Spec coverage:
 *  - multi-garment extraction (never stops at the first garment)
 *  - colour <-> garment association by phrase proximity (no cross-leak)
 *  - category normalisation + colour aliases
 *  - attribute (fit/pattern) extraction scoped to the right garment
 *  - honesty: nothing invented; unknown colours stay unknown; ambiguous
 *    phrases surface as editable placeholders
 *  - determinism, stable ids, dedupe
 *  - regression: "cobalt blue top and teal bottom" yields BOTH garments and
 *    a multi-colour analysis end-to-end
 *  - architecture: the describe path makes NO external AI/network call and
 *    needs NO API key
 *
 * Run: npm run test:outfit
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  parseDescriptionLocally,
  runDescribeAnalysis,
  runWardrobeAnalysis,
  garmentInputsToOutfitGarments,
  computeSuitability,
} from "../api/outfit-core";

const PROFILE = {
  season: "Warm Autumn",
  undertone: "warm",
  skin_depth: "light-medium",
  chroma: "muted",
  contrast: "medium",
};

function summary(result: ReturnType<typeof parseDescriptionLocally>) {
  return result.garments.map((g) => ({
    category: g.category,
    colourName: g.colourName,
    hex: g.hex,
    fit: g.fit ?? null,
    pattern: g.pattern ?? null,
  }));
}

/* ================================================================== */
/*  Required test cases 1-13                                            */
/* ================================================================== */

describe("parser acceptance tests", () => {
  test("1. 'black shirt and blue jeans' -> 2 garments", () => {
    const r = parseDescriptionLocally("black shirt and blue jeans");
    assert.equal(r.garments.length, 2);
    assert.equal(r.garments[0].category, "Shirt");
    assert.equal(r.garments[0].colourName, "Black");
    assert.equal(r.garments[1].category, "Jeans");
    assert.equal(r.garments[1].colourName, "Blue");
  });

  test("2. spec sentence -> 3 garments with colours/categories/attributes", () => {
    const r = parseDescriptionLocally(
      "cobalt blue oversized shirt, teal wide-leg trousers and beige sneakers",
    );
    assert.equal(r.garments.length, 3);
    assert.deepEqual(summary(r), [
      { category: "Shirt", colourName: "Cobalt Blue", hex: "#0047AB", fit: "oversized", pattern: null },
      { category: "Trousers", colourName: "Teal", hex: "#008080", fit: "wide-leg", pattern: null },
      { category: "Sneakers", colourName: "Beige", hex: "#F5F5DC", fit: null, pattern: null },
    ]);
    assert.equal(r.assumptions.length, 0);
  });

  test("3. 'navy blazer over a white shirt with charcoal trousers' -> 3 garments", () => {
    const r = parseDescriptionLocally("navy blazer over a white shirt with charcoal trousers");
    assert.equal(r.garments.length, 3);
    assert.deepEqual(
      summary(r).map((g) => [g.category, g.colourName]),
      [
        ["Blazer", "Navy"],
        ["Shirt", "White"],
        ["Trousers", "Charcoal"],
      ],
    );
  });

  test("4. 'cream blouse, burgundy midi skirt and brown loafers' -> 3 garments", () => {
    const r = parseDescriptionLocally("cream blouse, burgundy midi skirt and brown loafers");
    assert.equal(r.garments.length, 3);
    const skirt = r.garments.find((g) => g.category === "Skirt")!;
    assert.equal(skirt.colourName, "Burgundy");
    assert.equal(skirt.fit, "midi");
    assert.ok(r.garments.some((g) => g.category === "Blouse" && g.colourName === "Cream"));
    assert.ok(r.garments.some((g) => g.category === "Loafers" && g.colourName === "Brown"));
  });

  test("5. case-insensitive input behaves identically", () => {
    const lower = parseDescriptionLocally("cobalt blue shirt and teal trousers");
    const upper = parseDescriptionLocally("COBALT BLUE SHIRT AND TEAL TROUSERS");
    assert.equal(upper.garments.length, 2);
    assert.deepEqual(summary(upper), summary(lower));
  });

  test("6. plural handling: 'blue jeans' identifies Jeans", () => {
    const r = parseDescriptionLocally("blue jeans");
    assert.equal(r.garments.length, 1);
    assert.equal(r.garments[0].category, "Jeans");
    assert.equal(r.garments[0].colourName, "Blue");
  });

  test("7. colour alias: 'navy blue shirt' normalises to Navy", () => {
    const r = parseDescriptionLocally("navy blue shirt");
    assert.equal(r.garments[0].colourName, "Navy");
    assert.equal(r.garments[0].hex, "#1A1A3E");
  });

  test("8. multiple conjunctions -> 4 garments", () => {
    const r = parseDescriptionLocally("black shirt, blue jeans, white sneakers and brown belt");
    assert.equal(r.garments.length, 4);
    assert.deepEqual(
      summary(r).map((g) => [g.category, g.colourName]),
      [
        ["Shirt", "Black"],
        ["Jeans", "Blue"],
        ["Sneakers", "White"],
        ["Belt", "Brown"],
      ],
    );
  });

  test("9. attributes: 'oversized cobalt blue shirt'", () => {
    const r = parseDescriptionLocally("oversized cobalt blue shirt");
    assert.equal(r.garments[0].category, "Shirt");
    assert.equal(r.garments[0].colourName, "Cobalt Blue");
    assert.equal(r.garments[0].fit, "oversized");
  });

  test("10. fit stays on its own garment ('wide-leg' never leaks to the shirt)", () => {
    const r = parseDescriptionLocally("cobalt blue oversized shirt and teal wide-leg trousers");
    const shirt = r.garments.find((g) => g.category === "Shirt")!;
    const trousers = r.garments.find((g) => g.category === "Trousers")!;
    assert.equal(trousers.fit, "wide-leg");
    assert.equal(shirt.fit, "oversized");
    assert.notEqual(shirt.fit, "wide-leg");
  });

  test("11. 'I like blue colours.' invents no garment", () => {
    const r = parseDescriptionLocally("I like blue colours.");
    assert.equal(r.garments.length, 0);
  });

  test("12. empty input -> validation error upstream", async () => {
    assert.equal(parseDescriptionLocally("   ").garments.length, 0);
    const res = await runDescribeAnalysis({ description: "  " });
    assert.equal(res.success, false);
    assert.equal(res.error, "EMPTY_DESCRIPTION");
  });

  test("13. duplicate phrase dedupes; genuinely distinct garments stay", () => {
    assert.equal(parseDescriptionLocally("black shirt and black shirt").garments.length, 1);
    const distinct = parseDescriptionLocally("black shirt and blue shirt");
    assert.equal(distinct.garments.length, 2);
  });
});

/* ================================================================== */
/*  Honesty: nothing invented (spec 15/16/45/46)                        */
/* ================================================================== */

describe("honesty guarantees", () => {
  test("'something blue' surfaces as an ambiguous editable item, never an invented shirt", () => {
    const r = parseDescriptionLocally("I have something blue that I usually wear with jeans");
    const placeholder = r.garments.find((g) => g.category === "Garment");
    assert.ok(placeholder, "colour-only phrase must become an editable placeholder");
    assert.equal(placeholder!.colourName, "Blue");
    assert.ok(r.assumptions.some((a) => a.includes("could not tell the garment type")));
    // must NOT claim a specific category the user never named
    assert.ok(!r.garments.some((g) => g.category === "Shirt" || g.category === "Top"));
    // the jeans mentioned in the same sentence are still extracted
    assert.ok(r.garments.some((g) => g.category === "Jeans"));
  });

  test("unmodellable colour compound stays Unknown, never silently mapped", () => {
    const r = parseDescriptionLocally("a sweater in a very unusual shade of blue-green");
    assert.equal(r.garments.length, 1);
    assert.equal(r.garments[0].category, "Sweater");
    assert.equal(r.garments[0].colourName, "Unknown");
    assert.equal(r.garments[0].hex, "");
    assert.ok(r.assumptions.some((a) => a.includes("could not tell the colour")));
  });

  test("'black shirt and trousers' does NOT over-generalise the colour", () => {
    const r = parseDescriptionLocally("black shirt and trousers");
    assert.equal(r.garments.length, 2);
    const trousers = r.garments.find((g) => g.category === "Trousers")!;
    assert.equal(trousers.colourName, "Unknown");
    assert.ok(r.assumptions.some((a) => a.includes("could not tell the colour")));
  });

  test("'black shirt and blue trousers' keeps the colours separate", () => {
    const r = parseDescriptionLocally("black shirt and blue trousers");
    const shirt = r.garments.find((g) => g.category === "Shirt")!;
    const trousers = r.garments.find((g) => g.category === "Trousers")!;
    assert.equal(shirt.hex, "#000000");
    assert.equal(trousers.hex, "#0047AB");
  });

  test("raw categories are normalised (tee/tshirt/t-shirt -> T-Shirt, pants -> Trousers)", () => {
    for (const word of ["tee", "tshirt", "t-shirt", "t shirt"]) {
      const r = parseDescriptionLocally(`a black ${word}`);
      assert.equal(r.garments[0].category, "T-Shirt", word);
    }
    assert.equal(parseDescriptionLocally("a blue pair of pants").garments[0].category, "Trousers");
    assert.equal(parseDescriptionLocally("brown trainers").garments[0].category, "Sneakers");
  });

  test("pattern attaches to the right garment ('maroon kurti with gold embroidery')", () => {
    const r = parseDescriptionLocally("a maroon kurti with gold embroidery");
    assert.equal(r.garments.length, 1);
    assert.equal(r.garments[0].category, "Kurti");
    assert.equal(r.garments[0].colourName, "Maroon");
    assert.equal(r.garments[0].pattern, "Embroidered");
    // the modifier's colour word must not become a phantom garment
    assert.ok(!r.garments.some((g) => g.category === "Garment"));
  });
});

/* ================================================================== */
/*  Regression: the reported cobalt + teal production bug (spec 33)     */
/* ================================================================== */

describe("regression: cobalt blue top + teal bottom", () => {
  test("parser keeps both garments with their colours", () => {
    const r = parseDescriptionLocally("cobalt blue top and teal bottom");
    assert.equal(r.garments.length, 2);
    assert.equal(r.garments[0].hex, "#0047AB");
    assert.equal(r.garments[0].category, "Top");
    assert.equal(r.garments[1].hex, "#008080");
    assert.equal(r.garments[1].category, "Bottoms");
  });

  test("end-to-end: both garments reach the analysis engine; never 'single-colour'", async () => {
    const { garments } = parseDescriptionLocally("cobalt blue top and teal bottom");
    const res = await runWardrobeAnalysis(
      { garments, profile: PROFILE },
      null,
      async () => {
        throw new Error("offline — AI must not be needed for the colour verdict");
      },
    );
    assert.equal(res.success, true);
    assert.equal(res.garments.length, 2);
    assert.equal(res.garments[0].dominantColour.hex, "#0047AB");
    assert.equal(res.garments[1].dominantColour.hex, "#008080");
    // outfit contrast is computed from BOTH colours
    assert.ok(
      res.colour_analysis?.outfitContrast !== null &&
        res.colour_analysis?.outfitContrast !== undefined,
    );
    // multi-colour outfit must never be reported as single-colour
    const comboReason = res.suitability.dimensions.colourCombination.reason;
    assert.ok(!/single-colour/i.test(comboReason));

    // direct engine check: 2 valid colour measurements, honest contrast verdict
    const direct = computeSuitability(garmentInputsToOutfitGarments(garments), null, PROFILE);
    assert.ok(!direct.dimensions.contrast.reason.toLowerCase().includes("single"));
    assert.ok(!direct.dimensions.colourCombination.reason.toLowerCase().includes("single"));
  });
});

/* ================================================================== */
/*  Architecture: local-only, no AI, deterministic, fast (spec 34/35)   */
/* ================================================================== */

describe("architecture guarantees", () => {
  test("parser module contains no network/AI primitives", () => {
    const src = readFileSync(
      fileURLToPath(new URL("../api/outfit-parser.ts", import.meta.url)),
      "utf8",
    );
    const forbidden = [
      "generateContent",
      "GoogleGenAI",
      "Gemini",
      "fetch(",
      "axios",
      "XMLHttpRequest",
      "require(",
      "process.env",
      "http://",
      "https://",
    ];
    for (const token of forbidden) {
      assert.ok(!src.includes(token), `parser must not reference ${token}`);
    }
  });

  test("describe path in outfit-core has no Gemini text-mode call left", () => {
    const src = readFileSync(
      fileURLToPath(new URL("../api/outfit-core.ts", import.meta.url)),
      "utf8",
    );
    assert.ok(!src.includes("buildDescribePrompt"));
    assert.ok(!src.includes("callDescribeAi"));
    assert.ok(!src.includes("DescribeInterpreter"));
    // the image-mode Gemini integration stays untouched
    assert.ok(src.includes("callOutfitAi"));
  });

  test("parses 100 multi-garment descriptions well under interactive latency", () => {
    const start = Date.now();
    for (let i = 0; i < 100; i++) {
      parseDescriptionLocally(
        "I own a cobalt blue oversized shirt, teal wide-leg trousers and beige sneakers.",
      );
    }
    const elapsed = Date.now() - start;
    assert.ok(elapsed < 500, `parsing took too long: ${elapsed}ms for 100 descriptions`);
  });

  test("deterministic: identical input -> identical output", () => {
    const a = parseDescriptionLocally("a cream blouse, burgundy midi skirt and brown loafers");
    const b = parseDescriptionLocally("a cream blouse, burgundy midi skirt and brown loafers");
    assert.deepEqual(a, b);
  });

  test("stable unique ids for every extracted garment", () => {
    const { garments } = parseDescriptionLocally("black shirt, blue jeans, white sneakers and brown belt");
    assert.equal(new Set(garments.map((g) => g.id)).size, garments.length);
    assert.ok(garments.every((g) => g.id.length > 0));
  });
});

/* ================================================================== */
/*  Expanded dictionaries (dupatta, anorak, plurals, fabric colours)    */
/* ================================================================== */

describe("expanded dictionaries", () => {
  test("Indian wardrobe: dupatta with modifier", () => {
    const r = parseDescriptionLocally("a red dupatta with gold embroidery");
    assert.equal(r.garments.length, 1);
    assert.equal(r.garments[0].category, "Dupatta");
    assert.equal(r.garments[0].colourName, "Red");
    assert.equal(r.garments[0].pattern, "Embroidered");
    assert.ok(!r.garments.some((g) => g.category === "Garment"));
  });

  test("global wardrobe: anorak + gloves in one sentence", () => {
    const r = parseDescriptionLocally("an olive anorak and grey gloves");
    assert.equal(r.garments.length, 2);
    assert.ok(r.garments.some((g) => g.category === "Anorak" && g.colourName === "Olive Green"));
    assert.ok(r.garments.some((g) => g.category === "Gloves" && g.colourName === "Grey"));
  });

  test("plural garments match without dedicated plural entries", () => {
    for (const [phrase, category] of [
      ["blue tops", "Top"],
      ["black t-shirts", "T-Shirt"],
      ["white dresses", "Dress"],
      ["grey cardigans", "Cardigan"],
      ["blue dungarees", "Dungarees"],
    ] as [string, string][]) {
      const r = parseDescriptionLocally(phrase);
      assert.equal(r.garments.length, 1, phrase);
      assert.equal(r.garments[0].category, category, phrase);
    }
  });

  test("singular footwear forms map to the plural category", () => {
    assert.equal(parseDescriptionLocally("a white sneaker").garments[0].category, "Sneakers");
    assert.equal(parseDescriptionLocally("brown boots").garments[0].category, "Boots");
    assert.equal(parseDescriptionLocally("black sandal").garments[0].category, "Sandals");
  });

  test("'salwar kameez' is one compound garment, not two", () => {
    const r = parseDescriptionLocally("green salwar kameez");
    assert.equal(r.garments.length, 1);
    assert.equal(r.garments[0].category, "Salwar Kameez");
    assert.equal(r.garments[0].colourName, "Green");
  });

  test("new fabric/fashion colours parse", () => {
    assert.equal(parseDescriptionLocally("a saffron kurta").garments[0].colourName, "Saffron");
    assert.equal(parseDescriptionLocally("a peacock blue saree").garments[0].colourName, "Peacock Blue");
    assert.equal(parseDescriptionLocally("oxblood boots").garments[0].colourName, "Oxblood");
    assert.equal(parseDescriptionLocally("a moss blouse").garments[0].colourName, "Moss");
    assert.equal(parseDescriptionLocally("a champagne gown").garments[0].colourName, "Champagne");
  });

  test("scarf aliases (muffler, scarves) and multi-word compound colours stay longest-first", () => {
    assert.equal(parseDescriptionLocally("a red muffler").garments[0].category, "Scarf");
    assert.equal(parseDescriptionLocally("two black scarves").garments[0].category, "Scarf");
    assert.equal(parseDescriptionLocally("midnight blue shirt").garments[0].colourName, "Midnight Blue");
    assert.equal(parseDescriptionLocally("cornflower blue shirt").garments[0].colourName, "Cornflower Blue");
  });

  test("plain colours still win inside their phrases (no regression)", () => {
    assert.equal(parseDescriptionLocally("navy blue shirt").garments[0].colourName, "Navy");
    assert.equal(parseDescriptionLocally("hot pink jeans").garments[0].colourName, "Hot Pink");
  });
});
