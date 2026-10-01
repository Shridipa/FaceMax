/**
 * SSR render tests — the real Outfit Studio page and mode components are
 * rendered with react-dom/server under stubbed sessionStorage/localStorage.
 * These catch runtime render crashes that typecheck cannot (and, unlike a
 * browser, run deterministically in CI).
 * Run: npm run test:outfit
 */
import { test, describe, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import React from "react";
import { renderToString } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import OutfitStudio from "../src/pages/OutfitStudio";
import { DescribeMode } from "../src/components/outfit/DescribeMode";
import { WardrobeMode } from "../src/components/outfit/WardrobeMode";
import { ComparisonView } from "../src/components/outfit/ComparisonView";
import type { CompareResponse } from "../src/types/outfit";

/* ---------------- browser-storage stubs ---------------- */

const ORIGINAL_SESSION = globalThis.sessionStorage;
const ORIGINAL_LOCAL = globalThis.localStorage;

function makeStorage(initial: Record<string, string> = {}) {
  const store = new Map(Object.entries(initial));
  return {
    getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
    clear: () => void store.clear(),
  };
}

const SAVED_ANALYSIS = JSON.stringify({
  season: { primary: "Deep Winter" },
  axes: {
    undertone: { prediction: "cool" },
    skin_depth: { prediction: "deep" },
    chroma: { prediction: "bright" },
    contrast: { prediction: "high" },
  },
});

beforeEach(() => {
  Object.defineProperty(globalThis, "sessionStorage", { configurable: true, value: makeStorage() });
  Object.defineProperty(globalThis, "localStorage", { configurable: true, value: makeStorage() });
});
afterEach(() => {
  Object.defineProperty(globalThis, "sessionStorage", { configurable: true, value: ORIGINAL_SESSION });
  Object.defineProperty(globalThis, "localStorage", { configurable: true, value: ORIGINAL_LOCAL });
});

const e = React.createElement;
/** SSR inserts <!-- --> between dynamic and static text; strip for text assertions. */
const render = (node: React.ReactNode) =>
  renderToString(e(MemoryRouter, { initialEntries: ["/outfit-studio"] }, node)).replace(/<!-- -->/g, "");
const renderPlain = (node: React.ReactNode) => renderToString(node).replace(/<!-- -->/g, "");

/* ---------------- page ---------------- */

describe("OutfitStudio page (SSR)", () => {
  test("renders the missing-profile empty state when no analysis is saved", () => {
    const html = render(e(OutfitStudio));
    assert.ok(html.includes("Outfit Studio"));
    assert.ok(html.includes("Complete your FaceMaxx colour analysis"));
    assert.ok(html.includes("Start colour analysis"));
  });

  test("renders the full three-mode studio when a profile exists", () => {
    Object.defineProperty(globalThis, "sessionStorage", {
      configurable: true,
      value: makeStorage({ fmax_analysis: SAVED_ANALYSIS }),
    });
    const html = render(e(OutfitStudio));
    assert.ok(html.includes("Outfit Studio"));
    // mode tabs
    assert.ok(html.includes("Photograph"));
    assert.ok(html.includes("Describe Clothes"));
    assert.ok(html.includes("My Wardrobe"));
    // profile banner personalisation
    assert.ok(html.includes("Deep Winter"));
    assert.ok(html.includes("Recommendations personalised with your saved analysis"));
    // photo mode is the default content
    assert.ok(html.includes("Upload an outfit photo"));
    assert.ok(html.includes("Choose file"));
    assert.ok(html.includes("Use camera"));
  });
});

/* ---------------- mode components ---------------- */

describe("DescribeMode (SSR)", () => {
  test("renders label, textarea, examples; context controls appear only after extraction", () => {
    const html = renderPlain(
      e(DescribeMode, {
        profile: { season: "Deep Winter" },
        onAnalyse: () => {},
        analyzing: false,
      }),
    );
    assert.ok(html.includes("Describe the clothes you own"));
    assert.ok(html.includes("Tell me what you have in your wardrobe"));
    assert.ok(html.includes("Extract my garments"));
    assert.ok(html.includes("black oversized shirt"));
    // occasion/style controls are gated behind a successful extraction (correct UX);
    // the interactive review/edit flow is exercised by the browser QA, since SSR
    // cannot simulate textarea input events.
    assert.ok(!html.includes("Generate My Outfit Ideas"));
  });
});

describe("WardrobeMode (SSR)", () => {
  test("renders empty state when nothing is stored", () => {
    const html = renderToString(e(WardrobeMode, { onAnalyse: () => {}, analyzing: false }));
    assert.ok(html.includes("Your wardrobe is empty"));
    assert.ok(html.includes("Add your first item"));
    assert.ok(html.includes("Saved in this browser only"));
  });

  test("renders saved items, search, filter and analysis action", () => {
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: makeStorage({
        fmax_wardrobe_v1: JSON.stringify([
          { id: "w-1", category: "Tops", colourName: "Navy", hex: "#1A1A3E", pattern: null, fit: null },
          { id: "w-2", category: "Bottoms", colourName: "Cream", hex: "#FFFDD0", pattern: null, fit: "wide-leg" },
        ]),
      }),
    });
    const html = renderPlain(e(WardrobeMode, { onAnalyse: () => {}, analyzing: false }));
    assert.ok(html.includes("Navy Tops"));
    assert.ok(html.includes("Cream Bottoms"));
    assert.ok(html.includes("wide-leg"));
    assert.ok(html.includes("Analyse Selected Outfit"));
    assert.ok(html.includes("0 selected")); // nothing ticked yet; selection is user-driven
  });
});

describe("ComparisonView (SSR)", () => {
  test("renders option cards with score, advantages and conflicts", () => {
    const comparison: CompareResponse = {
      success: true,
      options: [
        {
          label: "Navy + cream",
          garments: [],
          score: { score: 86, breakdown: [], note: "test note" },
          suitability: { verdict: "Good palette harmony", dimensions: {} as never, reasons: [] },
          palette: [{ name: "Dark Navy", hex: "#1A1A3E" }],
          advantages: ["Palette compatibility is high."],
          conflicts: ["Some tension between accents."],
          limitations: [],
        },
      ],
      summary: "Navy + cream scores highest on documented palette-compatibility rules (86/100).",
      evidence_class: "DETERMINISTIC PALETTE MATCH + AI STYLING (NOT SCIENTIFICALLY VALIDATED)",
      profile_used: { season: "Deep Winter", available_axes: ["season"], source: "session_profile" },
    };
    const html = renderToString(
      e(ComparisonView, { comparison, comparing: false, compareError: null, onClear: () => {} }),
    );
    assert.ok(html.includes("Compare outfit options"));
    assert.ok(html.includes("Navy + cream"));
    assert.ok(html.includes("86"));
    assert.ok(html.includes("Works well"));
    assert.ok(html.includes("Colour tensions"));
    assert.ok(html.includes("not which look is"));
  });
});
