/**
 * Outfit Studio client-service tests — profile retrieval and API access.
 *
 * sessionStorage and fetch are stubbed so the suite runs in plain Node with
 * no browser and no network. Run: npm run test:outfit
 */
import { test, describe, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { loadSavedProfile, analyseOutfit, describeClothes } from "../src/services/outfit";
import type { ColourAnalysisResponse } from "../src/types/api";

/* ------------------------------------------------------------------ */
/*  Global stubs                                                       */
/* ------------------------------------------------------------------ */

const ORIGINAL_SESSION = globalThis.sessionStorage;
const ORIGINAL_FETCH = globalThis.fetch;

function installSessionStorage(store: Map<string, string>) {
  Object.defineProperty(globalThis, "sessionStorage", {
    configurable: true,
    value: {
      getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
      setItem: (k: string, v: string) => void store.set(k, v),
      removeItem: (k: string) => void store.delete(k),
      clear: () => void store.clear(),
    },
  });
}

function installFetch(responder: () => Response) {
  globalThis.fetch = (async () => responder()) as typeof fetch;
}

beforeEach(() => {
  installSessionStorage(new Map());
});
afterEach(() => {
  Object.defineProperty(globalThis, "sessionStorage", { configurable: true, value: ORIGINAL_SESSION });
  globalThis.fetch = ORIGINAL_FETCH;
});

/* ------------------------------------------------------------------ */
/*  Fixtures                                                           */
/* ------------------------------------------------------------------ */

const SAVED_ANALYSIS = {
  season: { primary: "Soft Summer" },
  axes: {
    undertone: { prediction: "cool" },
    skin_depth: { prediction: "light" },
    chroma: { prediction: "muted" },
    contrast: { prediction: "low" },
  },
} as unknown as ColourAnalysisResponse;

/* ------------------------------------------------------------------ */
/*  loadSavedProfile                                                   */
/* ------------------------------------------------------------------ */

describe("loadSavedProfile", () => {
  test("returns null profile when nothing is saved", () => {
    const { profile, raw } = loadSavedProfile();
    assert.equal(profile, null);
    assert.equal(raw, null);
  });

  test("returns null profile on corrupted JSON", () => {
    installSessionStorage(new Map([["fmax_analysis", "{not json"]]));
    const { profile, raw } = loadSavedProfile();
    assert.equal(profile, null);
    assert.equal(raw, null);
  });

  test("returns null profile when the saved analysis has no season", () => {
    installSessionStorage(new Map([["fmax_analysis", JSON.stringify({ season: {} })]]));
    const { profile, raw } = loadSavedProfile();
    assert.equal(profile, null);
    assert.notEqual(raw, null); // raw object still echoed for the UI
  });

  test("maps a full saved analysis into the flat profile", () => {
    installSessionStorage(new Map([["fmax_analysis", JSON.stringify(SAVED_ANALYSIS)]]));
    const { profile, raw } = loadSavedProfile();
    assert.deepEqual(profile, {
      season: "Soft Summer",
      undertone: "cool",
      skin_depth: "light",
      chroma: "muted",
      contrast: "low",
    });
    assert.equal((raw as ColourAnalysisResponse | null)?.season?.primary, "Soft Summer");
  });
});

/* ------------------------------------------------------------------ */
/*  analyseOutfit                                                      */
/* ------------------------------------------------------------------ */

describe("analyseOutfit", () => {
  test("POSTs image + profile to /api/outfit-analysis and returns the parsed body", async () => {
    let capturedUrl = "";
    let capturedBody: Record<string, unknown> = {};
    globalThis.fetch = (async (url: string, init?: RequestInit) => {
      capturedUrl = String(url);
      capturedBody = JSON.parse(String(init?.body));
      return new Response(
        JSON.stringify({ success: true, outfit_combinations: [{ name: "Look 01" }] }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }) as typeof fetch;

    const res = await analyseOutfit(
      "data:image/jpeg;base64,abc",
      { season: "Soft Summer", undertone: "cool" },
      "Work & office",
      "minimal",
    );

    assert.equal(capturedUrl, "/api/outfit-analysis");
    assert.equal(capturedBody.image, "data:image/jpeg;base64,abc");
    assert.deepEqual(capturedBody.profile, { season: "Soft Summer", undertone: "cool" });
    assert.equal(capturedBody.occasion, "Work & office");
    assert.equal(capturedBody.stylePreference, "minimal");
    assert.deepEqual(res, { success: true, outfit_combinations: [{ name: "Look 01" }] });
  });

  test("omits optional context fields when not provided", async () => {
    let capturedBody: Record<string, unknown> = {};
    globalThis.fetch = (async (_url: string, init?: RequestInit) => {
      capturedBody = JSON.parse(String(init?.body));
      return new Response(JSON.stringify({ success: true }), { status: 200 });
    }) as typeof fetch;

    await analyseOutfit("data:image/jpeg;base64,abc", { season: "Soft Summer" });
    assert.ok(!("occasion" in capturedBody));
    assert.ok(!("stylePreference" in capturedBody));
  });

  test("maps structured API errors to err.code (NO_PROFILE)", async () => {
    installFetch(() =>
      new Response(
        JSON.stringify({ success: false, error: "NO_PROFILE", message: "Complete your FaceMaxx colour analysis." }),
        { status: 400 },
      ),
    );

    await assert.rejects(
      analyseOutfit("data:image/jpeg;base64,abc", { season: "" } as never),
      (err: Error & { code?: string }) => err.code === "NO_PROFILE" && err.message.includes("Complete your FaceMaxx"),
    );
  });

  test("falls back to a friendly message when the error body is not JSON", async () => {
    installFetch(() => new Response("<html>Bad Gateway</html>", { status: 502 }));

    await assert.rejects(
      analyseOutfit("data:image/jpeg;base64,abc", { season: "Soft Summer" }),
      (err: Error & { code?: string }) =>
        err.code === undefined && err.message.includes("Outfit analysis failed"),
    );
  });
});

/* ------------------------------------------------------------------ */
/*  describeClothes                                                    */
/* ------------------------------------------------------------------ */

describe("describeClothes", () => {
  test("POSTs description + profile to /api/outfit-analysis/describe", async () => {
    let capturedUrl = "";
    let capturedBody: Record<string, unknown> = {};
    globalThis.fetch = (async (url: string, init?: RequestInit) => {
      capturedUrl = String(url);
      capturedBody = JSON.parse(String(init?.body));
      return new Response(
        JSON.stringify({
          success: true,
          garments: [{ category: "Shirt", colourName: "Black", hex: "#000000" }],
          assumptions: [],
          limitations: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }) as typeof fetch;

    const res = await describeClothes(
      "black oversized shirt",
      { season: "Dark Winter" },
    );

    assert.equal(capturedUrl, "/api/outfit-analysis/describe");
    assert.equal(capturedBody.description, "black oversized shirt");
    assert.deepEqual(capturedBody.profile, { season: "Dark Winter" });
    assert.equal(res.success, true);
    assert.equal(res.garments.length, 1);
    assert.equal(res.garments[0].category, "Shirt");
  });

  test("throws structured error on non-ok HTTP responses", async () => {
    installFetch(() =>
      new Response(
        JSON.stringify({ success: false, error: "EMPTY_DESCRIPTION", message: "Describe at least one garment." }),
        { status: 400 },
      ),
    );

    await assert.rejects(
      describeClothes("", { season: "Dark Winter" }),
      (err: Error & { code?: string }) =>
        err.code === "EMPTY_DESCRIPTION" && err.message.includes("Describe at least one garment"),
    );
  });
});

