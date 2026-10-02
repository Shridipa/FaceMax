/**
 * Outfit Studio client — profile retrieval + API access.
 *
 * The saved FaceMaxx colour profile lives in sessionStorage under
 * "fmax_analysis" (written by the existing Analyze/Results flow, read by the
 * existing Results page). No new persistence mechanism is introduced; the
 * user never repeats their colour analysis to use Outfit Studio.
 */
import type { ColourAnalysisResponse } from "../types/api";
import type {
  OutfitAnalysisRequest, OutfitAnalysisResponse, DescribeResponse,
  CompareOptionInput, CompareResponse,
} from "../types/outfit";

const BASE_URL = "";

export function loadSavedProfile(): { profile: OutfitAnalysisRequest["profile"] | null; raw: ColourAnalysisResponse | null } {
  try {
    const stored = sessionStorage.getItem("fmax_analysis");
    if (!stored) return { profile: null, raw: null };
    const raw = JSON.parse(stored) as ColourAnalysisResponse;
    if (!raw || !raw.season?.primary) return { profile: null, raw };
    return {
      profile: {
        season: raw.season.primary,
        undertone: raw.axes?.undertone?.prediction,
        skin_depth: raw.axes?.skin_depth?.prediction,
        chroma: raw.axes?.chroma?.prediction,
        contrast: raw.axes?.contrast?.prediction,
      },
      raw,
    };
  } catch {
    return { profile: null, raw: null };
  }
}

export async function analyseOutfit(
  imageBase64: string,
  profile: NonNullable<OutfitAnalysisRequest["profile"]>,
  occasion?: string,
  stylePreference?: string,
): Promise<OutfitAnalysisResponse> {
  const response = await fetch(`${BASE_URL}/api/outfit-analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageBase64, profile, occasion, stylePreference }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const err = new Error(errorData.message || errorData.error || "Outfit analysis failed. Please try again.") as Error & { code?: string };
    err.code = errorData.error;
    throw err;
  }
  return response.json();
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    const err = new Error("Could not connect to the styling service. Please check your internet connection.") as Error & { code?: string };
    err.code = "NETWORK_ERROR";
    throw err;
  }

  if (!response.ok) {
    const errorData = (await response.json().catch(() => ({}))) as Record<string, unknown>;
    let message = typeof errorData.message === "string" ? errorData.message : "";
    if (!message && typeof errorData.error === "string") {
      message = errorData.error;
    }
    if (!message) {
      if (response.status === 404) {
        message = "Styling service endpoint not found (404).";
      } else if (response.status >= 500) {
        message = `Styling server encountered an error (${response.status}). Please try again shortly.`;
      } else if (response.status === 429) {
        message = "Too many requests. Please wait a moment before trying again.";
      } else {
        message = `Request failed (${response.status}). Please try again.`;
      }
    }
    const err = new Error(message) as Error & { code?: string; status?: number };
    if (typeof errorData.error === "string") err.code = errorData.error;
    err.status = response.status;
    throw err;
  }
  return response.json() as Promise<T>;
}

/* ------------------------------------------------------------------ */
/*  Mode 2 — describe clothes                                          */
/* ------------------------------------------------------------------ */

export async function describeClothes(
  description: string,
  profile: NonNullable<OutfitAnalysisRequest["profile"]>,
): Promise<DescribeResponse> {
  return postJson<DescribeResponse>("/api/outfit-analysis/describe", { description, profile });
}

/* ------------------------------------------------------------------ */
/*  Mode 3 — wardrobe outfit analysis                                  */
/* ------------------------------------------------------------------ */

export type WardrobeGarments = {
  id?: string;
  category: string; colourName: string; hex: string; pattern?: string | null; fit?: string | null;
}[];

export async function analyseWardrobeOutfit(
  garments: WardrobeGarments,
  profile: NonNullable<OutfitAnalysisRequest["profile"]>,
  occasion?: string,
  stylePreference?: string,
): Promise<OutfitAnalysisResponse> {
  return postJson<OutfitAnalysisResponse>("/api/outfit-analysis/wardrobe", {
    garments, profile, occasion, stylePreference,
  });
}

/* ------------------------------------------------------------------ */
/*  Comparison mode                                                    */
/* ------------------------------------------------------------------ */

export async function compareOutfitsApi(
  options: CompareOptionInput[],
  profile: NonNullable<OutfitAnalysisRequest["profile"]>,
): Promise<CompareResponse> {
  return postJson<CompareResponse>("/api/outfit-analysis/compare", { options, profile });
}

/* ------------------------------------------------------------------ */
/*  Wardrobe persistence (localStorage — no account system exists)     */
/* ------------------------------------------------------------------ */

export interface WardrobeItem {
  id: string;
  category: string;
  colourName: string;
  hex: string;
  pattern?: string | null;
  fit?: string | null;
}

const WARDROBE_KEY = "fmax_wardrobe_v1";
const MAX_WARDROBE_ITEMS = 100;

export function loadWardrobe(): WardrobeItem[] {
  try {
    const stored = localStorage.getItem(WARDROBE_KEY);
    if (!stored) return [];
    const parsed = JSON.parse(stored);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (i): i is WardrobeItem =>
          i && typeof i === "object" &&
          typeof i.id === "string" &&
          typeof i.category === "string" &&
          typeof i.hex === "string" && /^#[0-9a-fA-F]{6}$/.test(i.hex),
      )
      .slice(0, MAX_WARDROBE_ITEMS);
  } catch {
    return [];
  }
}

export function saveWardrobe(items: WardrobeItem[]): void {
  try {
    localStorage.setItem(WARDROBE_KEY, JSON.stringify(items.slice(0, MAX_WARDROBE_ITEMS)));
  } catch {
    // storage full or unavailable — wardrobe simply stays in memory this session
  }
}
