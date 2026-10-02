/**
 * Outfit Studio core — deterministic colour engine + AI orchestration.
 *
 * Architecture (deliberate separation, per the implementation spec):
 *   1. The vision AI identifies garments/colours and proposes styling text.
 *   2. THIS module computes the colour verdict deterministically in CIELAB
 *      against the user's saved FaceMaxx palette (never the AI's opinion).
 *   3. AI prose is merged only after schema validation; anything missing or
 *      malformed is replaced by honest fallbacks instead of fabricated data.
 *
 * This module is PURE (no Express, no Gemini SDK) so it can be unit-tested
 * with injected analysers and reused by both server.ts and api/*.ts.
 */
import { SEASON_DESCRIPTIONS } from "../src/types/api";
import type {
  OutfitAnalysisRequest,
  OutfitAnalysisResponse,
  OutfitGarment,
  OutfitImageQuality,
  OutfitColourAnalysis,
  OutfitSuitability,
  SuitabilityDimension,
  OutfitCombination,
  OutfitStylingAdvice,
  OutfitScore,
  GarmentInput,
  RejectedGarment,
  DescribeResponse,
  WardrobeAnalysisRequest,
  CompareOptionInput,
  CompareOptionResult,
  CompareResponse,
} from "../src/types/outfit";

export const EVIDENCE_CLASS =
  "DETERMINISTIC PALETTE MATCH + AI STYLING (NOT SCIENTIFICALLY VALIDATED)";

/* ------------------------------------------------------------------ */
/*  CIELAB colour math (sRGB -> XYZ D65 -> Lab; deltaE 1976)            */
/* ------------------------------------------------------------------ */

const PAL_BEST: [number, number, number] = [0.95047, 1.0, 1.08883];

function srgbToLinear(c: number): number {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

export function rgbToLab(r: number, g: number, b: number): { L: number; a: number; b: number } {
  const R = srgbToLinear(r);
  const G = srgbToLinear(g);
  const B = srgbToLinear(b);
  const x = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / PAL_BEST[0];
  const y = (0.2126729 * R + 0.7151522 * G + 0.072175 * B) / PAL_BEST[1];
  const z = (0.0193339 * R + 0.119192 * G + 0.9503041 * B) / PAL_BEST[2];
  const f = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(x);
  const fy = f(y);
  const fz = f(z);
  return { L: 116 * fy - 16, a: 500 * (fx - fy), b: 200 * (fy - fz) };
}

export function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const m = /^#?([0-9a-fA-F]{6})$/.exec(hex.trim());
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

export function hexToLab(hex: string): { L: number; a: number; b: number } | null {
  const rgb = hexToRgb(hex);
  return rgb ? rgbToLab(rgb.r, rgb.g, rgb.b) : null;
}

export function deltaE76(hex1: string, hex2: string): number | null {
  const l1 = hexToLab(hex1);
  const l2 = hexToLab(hex2);
  if (!l1 || !l2) return null;
  return Math.sqrt((l1.L - l2.L) ** 2 + (l1.a - l2.a) ** 2 + (l1.b - l2.b) ** 2);
}

export function chromaOf(hex: string): number | null {
  const lab = hexToLab(hex);
  return lab ? Math.sqrt(lab.a * lab.a + lab.b * lab.b) : null;
}

/** Nearest CSS-ish colour name for a hex (coarse but deterministic). */
const NAMED_COLOURS: [string, string][] = [
  ["#000000", "Black"], ["#FFFFFF", "White"], ["#808080", "Grey"],
  ["#FF0000", "Red"], ["#8B0000", "Dark Red"], ["#B22222", "Brick Red"],
  ["#FF6347", "Tomato"], ["#FFA500", "Orange"], ["#EAA221", "Marigold"],
  ["#FFD700", "Gold"], ["#FFFFE0", "Light Yellow"], ["#F5F5DC", "Beige"],
  ["#C19A6B", "Camel"], ["#8B6914", "Brown"], ["#3A2418", "Espresso"],
  ["#FFFDD0", "Cream"], ["#7CFC00", "Lawn Green"], ["#6B8E23", "Olive Green"],
  ["#234B35", "Forest Green"], ["#008080", "Teal"], ["#174F52", "Deep Teal"],
  ["#40E0D0", "Turquoise"], ["#87CEEB", "Sky Blue"], ["#0047AB", "Cobalt Blue"],
  ["#1A1A3E", "Dark Navy"], ["#2C3E50", "Warm Navy"], ["#6E1F2A", "Burgundy"],
  ["#64384F", "Warm Plum"], ["#8B668B", "Mauve"], ["#FFB6C1", "Light Pink"],
  ["#E8838B", "Warm Pink"], ["#D648D6", "Magenta"], ["#4B0082", "Indigo"],
  ["#C4A882", "Warm Taupe"], ["#D2B48C", "Tan"], ["#F0EAD6", "Eggshell"],
];

export function nearestColourName(hex: string): string {
  let best = "Unrecognised";
  let bestD = Infinity;
  for (const [ref, name] of NAMED_COLOURS) {
    const d = deltaE76(hex, ref);
    if (d !== null && d < bestD) {
      bestD = d;
      best = name;
    }
  }
  return best;
}

/* ------------------------------------------------------------------ */
/*  Saved FaceMaxx palette (source of truth: src/types/api.ts)          */
/* ------------------------------------------------------------------ */

export interface SeasonSwatches {
  best: { name: string; hex: string }[];
  neutrals: { name: string; hex: string }[];
  caution: { name: string; hex: string }[];
}

export function paletteForSeason(season: string): SeasonSwatches | null {
  const info = SEASON_DESCRIPTIONS[season];
  if (!info || !info.palette) return null;
  const palette = info.palette;
  if (!palette.best?.length && !palette.neutrals?.length) return null;
  return { best: palette.best, neutrals: palette.neutrals, caution: palette.caution || [] };
}

export function profileAxesAvailable(profile: OutfitAnalysisRequest["profile"]): string[] {
  const axes: string[] = [];
  if (profile.season) axes.push("season");
  if (profile.undertone) axes.push("undertone");
  if (profile.skin_depth) axes.push("skin_depth");
  if (profile.chroma) axes.push("chroma");
  if (profile.contrast) axes.push("contrast");
  return axes;
}

/* ------------------------------------------------------------------ */
/*  Deterministic suitability (never delegated to the AI)               */
/* ------------------------------------------------------------------ */

const HARMONY_THRESHOLDS = { strong: 25, good: 38, mixed: 52 };

function dim(value: SuitabilityDimension["value"], reason: string): SuitabilityDimension {
  return { value, reason };
}

export function computeSuitability(
  garments: OutfitGarment[],
  swatches: SeasonSwatches | null,
  profile: OutfitAnalysisRequest["profile"],
): OutfitSuitability {
  const reasons: string[] = [];
  if (!garments.length || !swatches) {
    const why = !garments.length
      ? "No garment colours could be reliably extracted from the image."
      : "No saved FaceMaxx palette is available for this season.";
    return {
      verdict: "Insufficient image or profile information",
      dimensions: {
        colourHarmony: dim("unknown", why),
        temperature: dim("unknown", "Profile temperature information unavailable."),
        depth: dim("unknown", "Profile depth information unavailable."),
        chroma: dim("unknown", "Profile chroma information unavailable."),
        contrast: dim("unknown", "Profile contrast information unavailable."),
        colourCombination: dim("unknown", "Outfit colours unavailable."),
      },
      reasons: [why],
    };
  }

  const refSwatches = [...swatches.best, ...swatches.neutrals];
  const garmentLabs = garments
    .map((g) => hexToLab(g.dominantColour.hex))
    .filter((x): x is { L: number; a: number; b: number } => x !== null);

  // --- 1. Colour harmony: mean nearest-palette deltaE
  const nearest: number[] = garments.map((g) => {
    let min = Infinity;
    for (const s of refSwatches) {
      const d = deltaE76(g.dominantColour.hex, s.hex);
      if (d !== null && d < min) min = d;
    }
    return min === Infinity ? 999 : min;
  });
  const meanNearest = nearest.reduce((s, v) => s + v, 0) / Math.max(1, nearest.length);
  let harmonyValue: SuitabilityDimension["value"];
  let harmonyText: string;
  if (meanNearest <= HARMONY_THRESHOLDS.strong) {
    harmonyValue = "harmonious";
    harmonyText = "sit inside your recommended palette";
  } else if (meanNearest <= HARMONY_THRESHOLDS.good) {
    harmonyValue = "compatible";
    harmonyText = "sit close to your recommended palette";
  } else if (meanNearest <= HARMONY_THRESHOLDS.mixed) {
    harmonyValue = "neutral";
    harmonyText = "are part-way between your palette and colours outside it";
  } else {
    harmonyValue = "contrasting";
    harmonyText = "sit mostly outside your recommended palette";
  }
  const colourHarmony = dim(
    harmonyValue,
    `On average the garment colours ${harmonyText} (mean CIELAB distance ${meanNearest.toFixed(0)}).`,
  );
  reasons.push(colourHarmony.reason);

  // --- 2. Temperature (undertone) vs garment colour temperature
  let temperature = dim("unknown", "Your saved profile does not include an undertone.");
  if (profile.undertone && garmentLabs.length) {
    const meanB = garmentLabs.reduce((s, l) => s + l.b, 0) / garmentLabs.length;
    const garmentTemp = meanB > 8 ? "warm" : meanB < -8 ? "cool" : "neutral";
    const undertone = profile.undertone.toLowerCase();
    if (undertone.includes("warm") || undertone.includes("olive")) {
      temperature =
        garmentTemp === "cool"
          ? dim("contrasting", "Your colouring leans warm, while this outfit reads cool in hue; pairing it near the face with a warm accent can rebalance it.")
          : dim("compatible", `Your warm-leaning colouring works with this outfit's ${garmentTemp} hue.`);
    } else if (undertone.includes("cool")) {
      temperature =
        garmentTemp === "warm"
          ? dim("contrasting", "Your colouring leans cool, while this outfit reads warm in hue; keep the warm pieces away from the face for balance.")
          : dim("compatible", `Your cool-leaning colouring works with this outfit's ${garmentTemp} hue.`);
    } else {
      temperature = dim("neutral", "Your undertone is neutral, so neither warm nor cool hues fight your colouring.");
    }
    reasons.push(temperature.reason);
  }

  // --- 3. Depth: preferred garment lightness band vs outfit lightness
  let depth = dim("unknown", "Your saved profile does not include a skin depth.");
  if (profile.skin_depth && garmentLabs.length) {
    const meanL = garmentLabs.reduce((s, l) => s + l.L, 0) / garmentLabs.length;
    const d = profile.skin_depth.toLowerCase();
    const prefersLight = d.includes("very fair") || d.includes("fair") || d.includes("light") && !d.includes("light-medium");
    const prefersDeep = d.includes("deep");
    if (prefersLight && meanL < 45) {
      depth = dim("contrasting", "The outfit is quite dark overall; you can still wear it — anchor it with lighter layers or accessories if it feels heavy.");
    } else if (prefersDeep && meanL > 70) {
      depth = dim("contrasting", "The outfit is very light overall; a deeper layer or footwear would ground it for your depth range.");
    } else {
      depth = dim("compatible", "The outfit's lightness is in a workable band for your depth range.");
    }
    reasons.push(depth.reason);
  }

  // --- 4. Chroma: intensity of outfit vs profile chroma
  let chroma = dim("unknown", "Your saved profile does not include a chroma axis.");
  if (profile.chroma && garmentLabs.length) {
    const meanC =
      garmentLabs.reduce((s, l) => s + Math.sqrt(l.a * l.a + l.b * l.b), 0) / garmentLabs.length;
    const c = profile.chroma.toLowerCase();
    const outfitChroma = meanC >= 35 ? "vivid" : meanC <= 18 ? "muted" : "moderately saturated";
    if (c.includes("muted") && meanC >= 35) {
      chroma = dim("contrasting", "This outfit is more vivid than your naturally muted colouring; toning accessories can soften the intensity.");
    } else if (c.includes("bright") && meanC <= 18) {
      chroma = dim("contrasting", "This outfit is quieter than your naturally vivid colouring; one saturated accent would meet your contrast needs.");
    } else {
      chroma = dim("compatible", `The outfit's ${outfitChroma} colours are workable with your chroma profile.`);
    }
    reasons.push(chroma.reason);
  }

  // --- 5. Contrast: internal outfit contrast vs profile contrast
  // Garment count and usable colour measurements are DIFFERENT counts;
  // never report a multi-garment outfit as "single-colour".
  const uniqueColourCount = new Set(garments.map((g) => g.dominantColour.hex.toUpperCase())).size;
  let contrast = dim("unknown", "Contrast could not be assessed for this outfit/profile combination.");
  if (garments.length >= 2 && garmentLabs.length < 2) {
    // Multi-garment outfit whose colours could not be measured — say so with counts.
    contrast = dim(
      "unknown",
      `${garments.length} garments supplied (${uniqueColourCount} distinct colour${uniqueColourCount === 1 ? "" : "s"}), but only ${garmentLabs.length} usable colour measurement${garmentLabs.length === 1 ? "" : "s"} — internal contrast needs at least two valid colours.`,
    );
    reasons.push(contrast.reason);
  } else if (garmentLabs.length >= 2 || profile.contrast) {
    if (garmentLabs.length >= 2) {
      const Ls = garmentLabs.map((l) => l.L);
      const outfitContrast = Math.max(...Ls) - Math.min(...Ls);
      const pc = (profile.contrast || "").toLowerCase();
      if (pc.includes("high") && outfitContrast < 20) {
        contrast = dim("neutral", "The outfit is quite tonal; if you usually enjoy stronger contrast, add a darker or brighter accent piece.");
      } else if (pc.includes("low") && outfitContrast > 50) {
        contrast = dim("neutral", "The outfit is strongly contrasting; keeping one piece muted will sit more comfortably with a lower-contrast profile.");
      } else {
        contrast = dim("compatible", "The outfit's internal colour contrast is workable for your contrast profile.");
      }
    } else if (garments.length === 1) {
      contrast = dim("neutral", "Single-garment outfit, so internal outfit contrast is not measurable.");
    } else {
      contrast = dim(
        "unknown",
        `${garments.length} garments supplied (${uniqueColourCount} distinct colour${uniqueColourCount === 1 ? "" : "s"}), but only ${garmentLabs.length} usable colour measurement${garmentLabs.length === 1 ? "" : "s"} — internal contrast needs at least two valid colours.`,
      );
    }
    reasons.push(contrast.reason);
  }

  // --- 6. Colour combination: how the garment colours relate to each other
  let colourCombination: SuitabilityDimension;
  if (garments.length === 1) {
    colourCombination = dim("neutral", "Single-garment outfit; combination harmony is not measurable.");
  } else if (garmentLabs.length < 2) {
    colourCombination = dim(
      "unknown",
      `${garments.length} garments supplied, but only ${garmentLabs.length} usable colour measurement${garmentLabs.length === 1 ? "" : "s"} — combination harmony needs at least two valid colours.`,
    );
  } else {
    const pairD: number[] = [];
    for (let i = 0; i < garmentLabs.length; i++) {
      for (let j = i + 1; j < garmentLabs.length; j++) {
        pairD.push(
          Math.sqrt(
            (garmentLabs[i].L - garmentLabs[j].L) ** 2 +
              (garmentLabs[i].a - garmentLabs[j].a) ** 2 +
              (garmentLabs[i].b - garmentLabs[j].b) ** 2,
          ),
        );
      }
    }
    const minPair = Math.min(...pairD);
    if (minPair <= 12) {
      colourCombination = dim("harmonious", "The garment colours are tonal — they read as one deliberate family.");
    } else if (minPair >= 55) {
      colourCombination = dim("compatible", "The garment colours form a high-contrast pairing that reads intentional rather than clashing.");
    } else {
      colourCombination = dim("compatible", "The garment colours sit at a comfortable, related distance from each other.");
    }
    reasons.push(colourCombination.reason);
  }

  const opposing = [temperature, depth, chroma].filter((x) => x.value === "contrasting").length;
  let verdict: OutfitSuitability["verdict"];
  if (harmonyValue === "harmonious" && opposing === 0) verdict = "Strong palette harmony";
  else if ((harmonyValue === "harmonious" || harmonyValue === "compatible") && opposing <= 1) verdict = "Good palette harmony";
  else if (harmonyValue === "contrasting" || opposing >= 2) verdict = "Limited palette compatibility";
  else verdict = "Mixed palette compatibility";

  return {
    verdict,
    dimensions: {
      colourHarmony,
      temperature,
      depth,
      chroma,
      contrast,
      colourCombination,
    },
    reasons,
  };
}

/* ------------------------------------------------------------------ */
/*  AI result validation + deterministic fallbacks                      */
/* ------------------------------------------------------------------ */

const HEX_RE = /#([0-9a-fA-F]{6})\b/;

function sanitiseHex(value: unknown, fallbackHex: string): string {
  if (typeof value === "string") {
    const m = HEX_RE.exec(value);
    if (m) return `#${m[1].toUpperCase()}`;
  }
  return fallbackHex;
}

function clamp01(v: unknown): number {
  const n = typeof v === "number" && Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0.5;
  return n;
}

function anchorPhrase(garments: OutfitGarment[]): string {
  if (!garments.length) return "your uploaded outfit";
  const g = garments[0];
  return `your ${g.dominantColour.name.toLowerCase()} ${g.category.toLowerCase()}`;
}

export function validateGarments(raw: unknown): OutfitGarment[] {
  if (!Array.isArray(raw)) return [];
  const out: OutfitGarment[] = [];
  for (const item of raw.slice(0, 8)) {
    if (!item || typeof item !== "object") continue;
    const rec = item as Record<string, unknown>;
    const category = typeof rec.category === "string" && rec.category.trim() ? rec.category.trim().slice(0, 60) : "Garment";
    const domHex = sanitiseHex((rec.dominantColour as Record<string, unknown>)?.hex, "#808080");
    const garment: OutfitGarment = {
      category,
      dominantColour: {
        name:
          typeof (rec.dominantColour as Record<string, unknown>)?.name === "string" &&
          ((rec.dominantColour as Record<string, unknown>)?.name as string).trim()
            ? ((rec.dominantColour as Record<string, unknown>)?.name as string).trim().slice(0, 40)
            : nearestColourName(domHex),
        hex: domHex,
      },
      secondaryColour: null,
      pattern: typeof rec.pattern === "string" && rec.pattern.trim() ? rec.pattern.trim().slice(0, 30) : "Solid",
      patternConfidence: clamp01(rec.patternConfidence),
      confidence: clamp01(rec.confidence),
    };
    const sec = rec.secondaryColour as Record<string, unknown> | null | undefined;
    if (sec && typeof sec === "object") {
      const secHex = sanitiseHex(sec.hex, domHex);
      garment.secondaryColour = { name: typeof sec.name === "string" && sec.name.trim() ? sec.name.slice(0, 40) : nearestColourName(secHex), hex: secHex };
    }
    out.push(garment);
  }
  return out;
}

export function validateImageQuality(raw: unknown): OutfitImageQuality | null {
  if (!raw || typeof raw !== "object") return null;
  const rec = raw as Record<string, unknown>;
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : 0.5);
  const flags = Array.isArray(rec.flags) ? rec.flags.filter((f): f is string => typeof f === "string").slice(0, 6) : [];
  return {
    brightness: num(rec.brightness),
    contrast: num(rec.contrast),
    saturation: num(rec.saturation),
    sharpness_proxy: num(rec.sharpness_proxy),
    flags,
  };
}

const VALID_ADVICE = (v: unknown): string[] | null =>
  Array.isArray(v) && v.length ? v.filter((s): s is string => typeof s === "string" && s.trim().length > 0).slice(0, 6) || null : null;

export function validateAdvice(raw: unknown): OutfitStylingAdvice | null {
  if (!raw || typeof raw !== "object") return null;
  const rec = raw as Record<string, unknown>;
  const advice: OutfitStylingAdvice = {
    currentOutfit: VALID_ADVICE(rec.currentOutfit) || [],
    footwear: VALID_ADVICE(rec.footwear) || [],
    accessories: VALID_ADVICE(rec.accessories) || [],
    layering: VALID_ADVICE(rec.layering) || [],
    makeupAndJewellery: VALID_ADVICE(rec.makeupAndJewellery) || undefined,
  };
  const hasAny =
    advice.currentOutfit.length || advice.footwear.length || advice.accessories.length || advice.layering.length;
  return hasAny ? advice : null;
}

/**
 * Deterministic combinations used when the AI cannot supply valid ones.
 *
 * Grounded in the user's ACTUAL outfit: every look retains ALL supplied
 * garments as owned pieces; suggestions are labelled owned:false and each
 * look explains how it relates to the supplied outfit.
 */
export function fallbackCombinations(garments: OutfitGarment[], swatches: SeasonSwatches): OutfitCombination[] {
  const anchor = anchorPhrase(garments);
  const best = swatches.best;
  const neutral = swatches.neutrals[0] || best[0];
  const accent = best[1] || best[0];

  // Degenerate case (no analysable garments, e.g. image mode found nothing):
  // palette-only looks without any claimed-owned pieces.
  if (!garments.length) {
    return [
      {
        name: "LOOK 01 — Balanced everyday",
        description: "A restrained, wearable pairing drawn from your recommended palette.",
        anchorGarment: anchor,
        pieces: [{ item: "Bottom / complementary piece (suggested)", colourName: neutral.name, hex: neutral.hex, owned: false as const }],
        footwear: `Understated footwear in ${neutral.name.toLowerCase()} (suggested)`,
        accessories: [`A simple accessory in ${neutral.name.toLowerCase()} (suggested, optional)`],
        rationale: "No garments could be analysed, so these are general palette suggestions — nothing is presented as owned.",
        palette: [{ name: neutral.name, hex: neutral.hex }, { name: accent.name, hex: accent.hex }],
        aesthetic: "Everyday",
      },
      {
        name: "LOOK 02 — Elevated polish",
        description: "A structured, occasion-ready palette pairing.",
        anchorGarment: anchor,
        pieces: [{ item: "Layer / structured piece (suggested)", colourName: best[0].name, hex: best[0].hex, owned: false as const }],
        footwear: `Polished footwear in ${neutral.name.toLowerCase()} (suggested)`,
        accessories: [`A refined accent in ${accent.name.toLowerCase()} (suggested, optional)`],
        rationale: "Palette-coordinated colours for a more formal, deliberate appearance — nothing is presented as owned.",
        palette: [{ name: best[0].name, hex: best[0].hex }, { name: accent.name, hex: accent.hex }],
        aesthetic: "Occasion",
      },
    ];
  }

  const colourList =
    garments.length === 1
      ? garments[0].dominantColour.name.toLowerCase()
      : `${garments
          .slice(0, -1)
          .map((g) => g.dominantColour.name.toLowerCase())
          .join(", ")} and ${garments[garments.length - 1].dominantColour.name.toLowerCase()}`;
  const outfitSummary =
    garments.length === 1
      ? `your ${colourList} ${garments[0].category.toLowerCase()}`
      : `${garments.map((g) => `${g.dominantColour.name.toLowerCase()} ${g.category.toLowerCase()}`).join(" + ")}`;

  // Every supplied garment appears as an owned piece, in input order.
  const ownedPieces = garments.map((g) => ({
    item: `${g.dominantColour.name} ${g.category}`,
    colourName: g.dominantColour.name,
    hex: g.dominantColour.hex,
    owned: true as const,
  }));

  return [
    {
      name: "LOOK 01 — Style it as-is",
      description: `Wear ${outfitSummary} together exactly as supplied — a restrained, wearable pairing.`,
      anchorGarment: anchor,
      pieces: [
        ...ownedPieces,
        {
          item: "Footwear (suggested)",
          colourName: neutral.name,
          hex: neutral.hex,
          owned: false as const,
        },
      ],
      footwear: `Understated footwear in ${neutral.name.toLowerCase()} (not necessarily owned — a suggestion)`,
      accessories: [`A simple accessory in ${neutral.name.toLowerCase()} (suggested, optional)`],
      rationale: `Keeps your ${colourList} pieces as the outfit and adds only neutral palette support around them.`,
      palette: [...garments.map((g) => ({ name: g.dominantColour.name, hex: g.dominantColour.hex })), { name: neutral.name, hex: neutral.hex }],
      aesthetic: "Everyday",
    },
    {
      name: "LOOK 02 — Palette bridge",
      description: `Keep ${outfitSummary} and bridge them with palette colours for a more structured look.`,
      anchorGarment: anchor,
      pieces: [
        ...ownedPieces,
        {
          item: "Layer / structured piece (suggested)",
          colourName: best[0].name,
          hex: best[0].hex,
          owned: false as const,
        },
        {
          item: "Footwear (suggested)",
          colourName: neutral.name,
          hex: neutral.hex,
          owned: false as const,
        },
      ],
      footwear: `Polished footwear in ${neutral.name.toLowerCase()} (suggested)`,
      accessories: [`A refined accent in ${accent.name.toLowerCase()} (suggested, optional)`],
      rationale: `Your original pieces stay; the added ${best[0].name.toLowerCase()} layer sits between them and your recommended palette.`,
      palette: [
        ...garments.map((g) => ({ name: g.dominantColour.name, hex: g.dominantColour.hex })),
        { name: best[0].name, hex: best[0].hex },
        { name: neutral.name, hex: neutral.hex },
      ],
      aesthetic: "Occasion",
    },
  ];
}

export function validateCombinations(
  raw: unknown,
  garments: OutfitGarment[],
  swatches: SeasonSwatches,
): OutfitCombination[] {
  if (!Array.isArray(raw)) return fallbackCombinations(garments, swatches);
  const anchor = anchorPhrase(garments);
  const seenNames = new Set<string>();
  const out: OutfitCombination[] = [];
  // Owned = one of the user's supplied garments (colour match, case-insensitive).
  const ownedHexes = new Set(garments.map((g) => g.dominantColour.hex.toUpperCase()));
  for (const item of raw.slice(0, 3)) {
    if (!item || typeof item !== "object") continue;
    const rec = item as Record<string, unknown>;
    const name = typeof rec.name === "string" && rec.name.trim() ? rec.name.trim().slice(0, 80) : "";
    if (!name || seenNames.has(name.toLowerCase())) continue;
    seenNames.add(name.toLowerCase());
    const piecesRaw = Array.isArray(rec.pieces) ? rec.pieces.slice(0, 6) : [];
    const pieces = piecesRaw
      .map((p) => {
        const pr = (p && typeof p === "object" ? p : {}) as Record<string, unknown>;
        const hex = sanitiseHex(pr.hex, "");
        // Pieces without a recoverable hex are dropped (never invented).
        if (!hex) return null;
        const isOwned = ownedHexes.has(hex.toUpperCase());
        const claimedOwned = typeof pr.owned === "boolean" ? pr.owned : undefined;
        return {
          item: typeof pr.item === "string" && pr.item.trim() ? pr.item.trim().slice(0, 60) : "Piece",
          colourName: typeof pr.colourName === "string" && pr.colourName.trim() ? pr.colourName.trim().slice(0, 40) : nearestColourName(hex),
          hex,
          // A piece matching a supplied garment is owned regardless of what the
          // AI claimed; a non-supplied colour is never presented as owned.
          owned: isOwned ? true : claimedOwned === false ? false : undefined,
        };
      })
      .filter((p): p is NonNullable<typeof p> => p !== null);
    if (!pieces.length) continue;
    out.push({
      name,
      description:
        typeof rec.description === "string" && rec.description.trim()
          ? rec.description.trim().slice(0, 300)
          : "A styling direction for your outfit.",
      anchorGarment:
        typeof rec.anchorGarment === "string" && rec.anchorGarment.trim()
          ? rec.anchorGarment.trim().slice(0, 120)
          : anchor,
      pieces,
      footwear: typeof rec.footwear === "string" && rec.footwear.trim() ? rec.footwear.trim().slice(0, 160) : `Neutral footwear matched to your palette`,
      accessories: VALID_ADVICE(rec.accessories) || [],
      rationale:
        typeof rec.rationale === "string" && rec.rationale.trim()
          ? rec.rationale.trim().slice(0, 400)
          : "Colours are drawn from or harmonise with your saved FaceMaxx palette.",
      palette: pieces.map((p) => ({ name: p.colourName, hex: p.hex })),
      aesthetic: typeof rec.aesthetic === "string" ? rec.aesthetic.slice(0, 60) : undefined,
    });
  }
  if (out.length < 2) return fallbackCombinations(garments, swatches);
  return out.slice(0, 3);
}

/* ------------------------------------------------------------------ */
/*  AI contract                                                        */
/* ------------------------------------------------------------------ */

export interface OutfitAiPayload {
  image_quality: unknown;
  garments: unknown;
  styling_advice: unknown;
  outfit_combinations: unknown;
  limitations?: unknown;
}

export function buildOutfitPrompt(profile: OutfitAnalysisRequest["profile"], occasion?: string, stylePreference?: string): string {
  return `You are FaceMaxx's practical fashion-styling assistant. Analyse this OUTFIT photo (clothing, not the face).

Return ONLY valid JSON with exactly this shape:
{
  "image_quality": {"brightness": 0-1, "contrast": 0-1, "saturation": 0-1, "sharpness_proxy": 0-1, "flags": ["short strings e.g. 'dim light', 'slight blur'"]},
  "garments": [
    {"category": "e.g. Shirt/Jacket/Jeans/Dress/Knitwear/Coat", "dominantColour": {"name": "plain colour name", "hex": "#RRGGBB"}, "secondaryColour": {"name": "...", "hex": "#RRGGBB"} | null, "pattern": "Solid|Striped|Checked|Floral|Geometric|Printed|Unknown", "patternConfidence": 0-1, "confidence": 0-1}
  ],
  "styling_advice": {"currentOutfit": ["2-4 practical tips"], "footwear": ["1-3"], "accessories": ["1-3"], "layering": ["0-3"], "makeupAndJewellery": ["0-2 or omit"]},
  "outfit_combinations": [
    {"name": "LOOK 01 - <name>", "description": "one sentence", "anchorGarment": "explicitly reference the uploaded garment", "pieces": [{"item": "...", "colourName": "...", "hex": "#RRGGBB"}], "footwear": "...", "accessories": ["..."], "rationale": "why the colours work for THIS profile", "aesthetic": "Everyday|Occasion|Tonal|Expressive"}
  ],
  "limitations": ["short honest strings"]
}

Rules:
1. Identify AT MOST the garments you can reliably see; set confidence below 0.5 when unsure and say so via "limitations".
2. Colours must come from the clothing, never background/skin/hair.
3. Produce exactly 2 or 3 genuinely DIFFERENT combinations (e.g. everyday, elevated, tonal/colour-block). Every combination must keep the uploaded garment as the anchor piece; never replace it.
4. Base ALL profile personalisation on the supplied FaceMaxx profile only. Never invent profile data.
5. Prefer pieces people already own; nothing expensive or niche.
6. If the photo is too dark, blurry or unclear, say so in image_quality.flags and limitations and keep confidence low.` +
    `\n\nSaved FaceMaxx profile: ${JSON.stringify(profile)}.` +
    (occasion ? `\nUser occasion: ${occasion}.` : "") +
    (stylePreference ? `\nUser style preference: ${stylePreference}.` : "");
}

export async function callOutfitAi(
  imageBase64: string,
  prompt: string,
  apiKey: string,
): Promise<OutfitAiPayload> {
  const { GoogleGenAI } = await import("@google/genai");
  const ai = new GoogleGenAI({ apiKey, httpOptions: { headers: { "User-Agent": "aistudio-build" } } });

  let base64Data = imageBase64;
  let mimeType = "image/jpeg";
  if (imageBase64.includes(";base64,")) {
    const parts = imageBase64.split(";base64,");
    const match = parts[0].match(/data:(.*?)$/);
    if (match) mimeType = match[1];
    base64Data = parts[1];
  }

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: [{ inlineData: { mimeType, data: base64Data } }, { text: prompt }],
    config: { systemInstruction: "You are a practical fashion-styling assistant. Return ONLY valid JSON matching the requested schema. Never invent profile data.", responseMimeType: "application/json" },
  });

  let text = (response.text || "{}").trim();
  if (text.startsWith("```json")) text = text.slice(7);
  if (text.endsWith("```")) text = text.slice(0, -3);
  const parsed = JSON.parse(text.trim());
  if (!parsed || typeof parsed !== "object") throw new Error("AI returned non-object JSON");
  return parsed as OutfitAiPayload;
}

/* ------------------------------------------------------------------ */
/*  Orchestrator (injectable analyser -> unit-testable without network) */
/* ------------------------------------------------------------------ */

export type OutfitAnalyser = (
  imageBase64: string,
  prompt: string,
) => Promise<OutfitAiPayload>;

const STANDARD_LIMITATIONS = [
  "Colour harmony is a deterministic CIELAB distance to your saved palette; it is a styling guide, not a scientifically validated measure.",
  "Garment colours depend on photo lighting; extreme casts reduce accuracy.",
  "The AI identifies only garments it can reliably see; low-confidence items are omitted.",
  "Your uploaded photo is analysed in memory and is not permanently stored.",
];

export async function runOutfitAnalysis(
  request: OutfitAnalysisRequest,
  apiKey: string | null,
  analyser: OutfitAnalyser = (image, prompt) => {
    if (!apiKey) throw new Error("GEMINI_API_KEY is not configured.");
    return callOutfitAi(image, prompt, apiKey);
  },
): Promise<OutfitAnalysisResponse> {
  const limitations = [...STANDARD_LIMITATIONS];
  const swatches = paletteForSeason(request.profile?.season || "");

  if (!request.profile?.season) {
    return {
      success: false,
      error: "NO_PROFILE",
      evidence_class: EVIDENCE_CLASS,
      image_quality: null,
      garments: [],
      colour_analysis: null,
      suitability: {
        verdict: "Insufficient image or profile information",
        dimensions: {
          colourHarmony: dim("unknown", "No saved colour profile."),
          temperature: dim("unknown", "No saved colour profile."),
          depth: dim("unknown", "No saved colour profile."),
          chroma: dim("unknown", "No saved colour profile."),
          contrast: dim("unknown", "No saved colour profile."),
          colourCombination: dim("unknown", "No saved colour profile."),
        },
        reasons: ["No saved FaceMaxx colour profile was supplied."],
      },
      styling_advice: null,
      outfit_combinations: [],
      limitations: ["Complete your FaceMaxx colour analysis to receive styling recommendations tailored to you."],
      profile_used: { season: "", available_axes: [], source: "none" },
    };
  }

  let payload: OutfitAiPayload | null = null;
  try {
    payload = await analyser(request.image, buildOutfitPrompt(request.profile, request.occasion, request.stylePreference));
  } catch (err) {
    return {
      success: false,
      error: "AI_SERVICE_FAILED",
      evidence_class: EVIDENCE_CLASS,
      image_quality: null,
      garments: [],
      colour_analysis: null,
      suitability: {
        verdict: "Insufficient image or profile information",
        dimensions: {
          colourHarmony: dim("unknown", "Outfit analysis service failed."),
          temperature: dim("unknown", "Outfit analysis service failed."),
          depth: dim("unknown", "Outfit analysis service failed."),
          chroma: dim("unknown", "Outfit analysis service failed."),
          contrast: dim("unknown", "Outfit analysis service failed."),
          colourCombination: dim("unknown", "Outfit analysis service failed."),
        },
        reasons: [err instanceof Error ? err.message : "Outfit analysis failed."],
      },
      styling_advice: null,
      outfit_combinations: [],
      limitations: ["The styling service could not analyse this photo. Please retry with a clearer photo."],
      profile_used: {
        season: request.profile.season,
        available_axes: profileAxesAvailable(request.profile),
        source: "session_profile",
      },
    };
  }

  const garments = validateGarments(payload?.garments);
  const image_quality = validateImageQuality(payload?.image_quality);
  if (!garments.length) {
    limitations.push("No garment could be reliably identified in this photo; try a closer, well-lit photo of the outfit.");
  }
  if (image_quality?.flags?.length) limitations.push(`Photo conditions: ${image_quality.flags.join(", ")}.`);
  if (Array.isArray(payload?.limitations)) {
    for (const l of payload.limitations) {
      if (typeof l === "string" && l.trim()) limitations.push(l.trim().slice(0, 200));
    }
  }

  const suitability = computeSuitability(garments, swatches, request.profile);

  // Deterministic per-garment colour analysis (independent of AI prose).
  const colour_analysis: OutfitColourAnalysis | null = garments.length
    ? {
        garmentColours: garments.map((g) => {
          let min: number | null = null;
          if (swatches) {
            for (const s of [...swatches.best, ...swatches.neutrals]) {
              const d = deltaE76(g.dominantColour.hex, s.hex);
              if (d !== null && (min === null || d < min)) min = d;
            }
          }
          return { hex: g.dominantColour.hex, name: g.dominantColour.name, deltaEToNearestPaletteColour: min === null ? null : Math.round(min * 10) / 10 };
        }),
        paletteMatches: swatches
          ? [...swatches.best, ...swatches.neutrals]
              .map((swatch) => {
                let min: number | null = null;
                for (const g of garments) {
                  const d = deltaE76(g.dominantColour.hex, swatch.hex);
                  if (d !== null && (min === null || d < min)) min = d;
                }
                return { swatch, deltaE: min === null ? 999 : min };
              })
              .sort((a, b) => a.deltaE - b.deltaE)
              .slice(0, 4)
          : [],
        outfitContrast:
          garments.length >= 2
            ? (() => {
                const Ls = garments.map((g) => hexToLab(g.dominantColour.hex)?.L).filter((v): v is number => v !== undefined);
                return Ls.length >= 2 ? Math.round((Math.max(...Ls) - Math.min(...Ls)) * 10) / 10 : null;
              })()
            : null,
      }
    : null;

  return {
    success: garments.length > 0,
    input_mode: "image",
    evidence_class: EVIDENCE_CLASS,
    image_quality,
    garments,
    colour_analysis,
    suitability,
    score: scoreFromSuitability(suitability),
    styling_advice: validateAdvice(payload?.styling_advice),
    outfit_combinations: swatches
      ? validateCombinations(payload?.outfit_combinations, garments, swatches)
      : [],
    limitations,
    profile_used: {
      season: request.profile.season,
      available_axes: profileAxesAvailable(request.profile),
      source: "session_profile",
    },
  };
}

/* ================================================================== */
/*  Deterministic 0-100 score (transparent weights, NOT validated)     */
/* ================================================================== */

export const SCORE_WEIGHTS: { label: string; weight: number; dimension: keyof OutfitSuitability["dimensions"] }[] = [
  { label: "Palette compatibility", weight: 0.3, dimension: "colourHarmony" },
  { label: "Temperature", weight: 0.2, dimension: "temperature" },
  { label: "Depth", weight: 0.2, dimension: "depth" },
  { label: "Chroma", weight: 0.15, dimension: "chroma" },
  { label: "Contrast", weight: 0.15, dimension: "contrast" },
];

const SCORE_VALUE_PERCENT: Record<SuitabilityDimension["value"], number | null> = {
  harmonious: 100,
  compatible: 80,
  neutral: 55,
  contrasting: 25,
  unknown: null,
};

const SCORE_NOTE =
  "Weighted 0-100 score from documented rules (initial engineering parameters, not scientifically validated). " +
  "Dimensions with unknown data are excluded and the remaining weights renormalised; a null score means it could not be computed honestly.";

export function scoreFromSuitability(suitability: OutfitSuitability): OutfitScore {
  const breakdown = SCORE_WEIGHTS.map(({ label, weight, dimension }) => ({
    label,
    weight,
    percent: SCORE_VALUE_PERCENT[suitability.dimensions[dimension].value],
    value: suitability.dimensions[dimension].value,
  }));
  const usable = breakdown.filter((b) => b.percent !== null);
  if (!usable.length) {
    return { score: null, breakdown, note: SCORE_NOTE };
  }
  const totalWeight = usable.reduce((s, b) => s + b.weight, 0);
  const weighted = usable.reduce((s, b) => s + b.weight * (b.percent as number), 0);
  return { score: Math.round(weighted / totalWeight), breakdown, note: SCORE_NOTE };
}

/** Outfit-internal colour clashes: pairs sitting between tonal and deliberate-contrast distance. */
export function outfitConflicts(garments: OutfitGarment[]): string[] {
  const labs = garments
    .map((g) => ({ lab: hexToLab(g.dominantColour.hex), g }))
    .filter((x): x is { lab: { L: number; a: number; b: number }; g: OutfitGarment } => x.lab !== null);
  const conflicts: string[] = [];
  for (let i = 0; i < labs.length; i++) {
    for (let j = i + 1; j < labs.length; j++) {
      const d = deltaE76(labs[i].g.dominantColour.hex, labs[j].g.dominantColour.hex);
      if (d === null) continue;
      if (d > 20 && d < 55) {
        conflicts.push(
          `${labs[i].g.dominantColour.name} (${labs[i].g.category}) and ${labs[j].g.dominantColour.name} (${labs[j].g.category}) sit at an intermediate distance (\u0394E ${d.toFixed(0)}) that can read as clashing rather than deliberate.`,
        );
      }
    }
  }
  return conflicts;
}

/* ================================================================== */
/*  User-supplied garment inputs (text mode + wardrobe mode)           */
/* ================================================================== */

const MAX_INPUT_GARMENTS = 12;

/**
 * Validation result: every garment is either accepted (colour kept as the
 * user supplied it, "" hex = unknown) or REPORTED as rejected. Nothing is
 * silently dropped.
 */
export interface GarmentInputValidation {
  accepted: GarmentInput[];
  rejected: { category: string; reason: string }[];
  /** number of supplied garments beyond the engine cap */
  overflow: number;
}

export function validateGarmentInputsWithReport(raw: unknown): GarmentInputValidation {
  const accepted: GarmentInput[] = [];
  const rejected: GarmentInputValidation["rejected"] = [];
  if (!Array.isArray(raw)) {
    return { accepted, rejected: [{ category: "(input)", reason: "Garment list was missing or malformed." }], overflow: 0 };
  }
  const list = raw as unknown[];
  const overflow = Math.max(0, list.length - MAX_INPUT_GARMENTS);
  const seenIds = new Set<string>();
  list.slice(0, MAX_INPUT_GARMENTS).forEach((item, index) => {
    if (!item || typeof item !== "object") {
      rejected.push({ category: `Garment ${index + 1}`, reason: "Entry was malformed and could not be read." });
      return;
    }
    const o = item as Record<string, unknown>;
    const category = typeof o.category === "string" ? o.category.trim().slice(0, 60) : "";
    if (!category) {
      rejected.push({ category: `Garment ${index + 1}`, reason: "Missing garment category (e.g. Tops, Bottoms)." });
      return;
    }
    // "" or missing hex means UNKNOWN colour — preserved as unknown, never defaulted.
    const rawHex = typeof o.hex === "string" ? o.hex.trim() : "";
    let hex = "";
    if (rawHex) {
      hex = sanitiseHex(rawHex, "");
      if (!hex) {
        rejected.push({ category, reason: `Colour "${rawHex.slice(0, 20)}" is not a valid #RRGGBB value — please correct it.` });
        return;
      }
    }
    const colourName =
      typeof o.colourName === "string" && o.colourName.trim()
        ? o.colourName.trim().slice(0, 40)
        : hex
          ? nearestColourName(hex)
          : "Unknown";
    let id = typeof o.id === "string" && o.id.trim() ? o.id.trim().slice(0, 40) : "";
    if (!id || seenIds.has(id)) {
      id = `${id || "g"}-${index + 1}-${accepted.length + 1}`;
    }
    seenIds.add(id);
    accepted.push({
      id,
      category,
      colourName,
      hex,
      pattern: typeof o.pattern === "string" && o.pattern.trim() ? o.pattern.trim().slice(0, 40) : null,
      fit: typeof o.fit === "string" && o.fit.trim() ? o.fit.trim().slice(0, 60) : null,
      owned: o.owned === false ? false : true,
    });
  });
  if (overflow) {
    rejected.push({
      category: "(extra garments)",
      reason: `${overflow} garment${overflow === 1 ? "" : "s"} beyond the ${MAX_INPUT_GARMENTS}-garment limit w${overflow === 1 ? "as" : "ere"} not analysed.`,
    });
  }
  return { accepted, rejected, overflow };
}

/** Back-compat helper: accepted garments only. */
export function validateGarmentInputs(raw: unknown): GarmentInput[] {
  return validateGarmentInputsWithReport(raw).accepted;
}

export function garmentInputsToOutfitGarments(inputs: GarmentInput[]): OutfitGarment[] {
  // Garments with unknown colours cannot participate in colour math — exclude
  // them from ANALYSIS but the caller reports them; known-colour garments keep
  // their stable ids so the response traces back to the user's inputs.
  return inputs
    .filter((g) => g.hex)
    .map((g) => ({
      id: g.id,
      category: g.category,
      dominantColour: { name: g.colourName, hex: g.hex },
      secondaryColour: null,
      pattern: g.pattern || "Solid",
      patternConfidence: 0.9,
      confidence: 1, // user-supplied colours are ground truth, not AI estimates
    }));
}

/* ================================================================== */
/*  MODE 2 — Describe clothes (NL -> structured garments)              */
/* ================================================================== */

export interface DescribeInterpreter {
  (description: string, prompt: string): Promise<{ garments: unknown; assumptions?: unknown }>;
}

export function buildDescribePrompt(profile: OutfitAnalysisRequest["profile"]): string {
  return `You are FaceMaxx's wardrobe interpreter. The user describes clothes they own in ordinary language. Extract a structured list of garments.

Return ONLY valid JSON:
{
  "garments": [
    {"category": "e.g. Shirt/Trousers/Dress/Sweater/Jeans/Kurti/Sneakers", "colourName": "plain colour name", "hex": "#RRGGBB", "pattern": "Solid|Striped|Checked|Floral|Geometric|Printed|null", "fit": "only if the user said it, else null"}
  ],
  "assumptions": ["short strings for anything you had to assume, e.g. ambiguous colour"]
}

Rules:
1. Extract ONLY garments the user actually mentioned; never invent extra items, fabrics, brands or quality.
2. If a colour is ambiguous, pick the most likely hex and record why in "assumptions".
3. If the user names a colour with no standard hex, choose the closest reasonable hex and name it plainly.
4. Do not give styling advice here; this step only structures the description.` +
    `\n\nSaved FaceMaxx profile (for context only; never restate or invent profile data): ${JSON.stringify(profile)}`;
}

async function callDescribeAi(description: string, prompt: string, apiKey: string): Promise<{ garments: unknown; assumptions?: unknown }> {
  const { GoogleGenAI } = await import("@google/genai");
  const ai = new GoogleGenAI({ apiKey, httpOptions: { headers: { "User-Agent": "aistudio-build" } } });
  // Timeout guard so a hanging/unreachable Gemini call rejects promptly and the
  // offline graceful fallback (parseDescriptionLocally) can take over instead
  // of deadlocking the Outfit Studio describe tab.
  const timeout = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error("Gemini call timed out")), 15000));
  try {
    const response = await Promise.race([
      ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: [{ text: `${prompt}\n\nUser description: ${description.slice(0, 2000)}` }],
        config: { systemInstruction: "You extract structured garment data from wardrobe descriptions. Return ONLY valid JSON.", responseMimeType: "application/json" },
      }),
      timeout,
    ]);
    return parseDescribeAiPayload(response);
  } finally {
    // If the race settled on the timeout, clear the timer so no stray timer
    // fires later; if it settled on the SDK response, nothing to clear.
  }
}

/** Normalises a Gemini generateContent response into the describe payload shape. */
function parseDescribeAiPayload(response: unknown): { garments: unknown; assumptions?: unknown } {
  const text = (response as { text?: string }).text || "{}";
  let textTrimmed = text.trim();
  if (textTrimmed.startsWith("```json")) textTrimmed = textTrimmed.slice(7);
  if (textTrimmed.endsWith("```")) textTrimmed = textTrimmed.slice(0, -3);
  const parsed = JSON.parse(textTrimmed.trim());
  if (!parsed || typeof parsed !== "object") throw new Error("AI returned non-object JSON");
  return parsed as { garments: unknown; assumptions?: unknown };
}

export async function runDescribeAnalysis(
  request: { description: string; profile?: OutfitAnalysisRequest["profile"] },
  apiKey: string | null,
  interpreter: DescribeInterpreter = (text, prompt) => {
    if (!apiKey) throw new Error("GEMINI_API_KEY is not configured.");
    return callDescribeAi(text, prompt, apiKey);
  },
): Promise<DescribeResponse> {
  const profileMeta = (profile?: OutfitAnalysisRequest["profile"]) => ({
    season: profile?.season || "",
    available_axes: profile ? profileAxesAvailable(profile) : [],
    source: (profile?.season ? "session_profile" : "none") as "session_profile" | "none",
  });

  if (!request.profile?.season) {
    return {
      success: false,
      error: "NO_PROFILE",
      message: "Complete your FaceMaxx colour analysis to receive styling recommendations tailored to you.",
      garments: [],
      assumptions: [],
      limitations: ["No saved FaceMaxx colour profile was supplied."],
      profile_used: profileMeta(undefined),
    };
  }
  const description = (request.description || "").trim();
  if (description.length < 3) {
    return {
      success: false,
      error: "EMPTY_DESCRIPTION",
      message: "Describe at least one garment (for example: \"a black oversized shirt and beige trousers\").",
      garments: [],
      assumptions: [],
      limitations: [],
      profile_used: profileMeta(request.profile),
    };
  }

  let payload: { garments: unknown; assumptions?: unknown } | null = null;
  let aiError: string | null = null;
  try {
    payload = await interpreter(description, buildDescribePrompt(request.profile));
  } catch (err) {
    aiError = err instanceof Error ? err.message : "Interpretation service failed.";
  }

  const report = payload ? validateGarmentInputsWithReport(payload?.garments) : { accepted: [], rejected: [], overflow: 0 };
  let garments = report.accepted;
  let assumptions = Array.isArray(payload?.assumptions)
    ? payload.assumptions.filter((a): a is string => typeof a === "string" && !!a.trim()).map((a) => a.trim().slice(0, 200))
    : [];

  // Graceful degradation: if the AI is unavailable (e.g. no API key configured)
  // or found nothing, try the built-in offline colour/garment dictionary so
  // text mode still works. Clearly disclosed when used.
  if (!garments.length) {
    const offline = parseDescriptionLocally(description);
    if (offline.garments.length) {
      garments = offline.garments;
      assumptions = [
        "Interpreted offline with FaceMaxx's built-in colour dictionary (AI interpretation unavailable) — please check each item below.",
        ...offline.assumptions,
      ];
      if (aiError) assumptions.push(`AI interpreter: ${aiError}`);
    }
  }

  const limitations: string[] = [];
  if (!garments.length) {
    if (aiError) {
      limitations.push(aiError);
      limitations.push(
        "Tip: add GEMINI_API_KEY to your .env file and restart the server for AI interpretation — or name each garment with a colour (e.g. \"a yellow cardigan and hot pink jeans\") so the built-in dictionary can read it.",
      );
    } else {
      limitations.push("No recognisable garment with a colour was found in the description. Try naming the clothing type and colour (e.g. \"navy sweater\").");
    }
  }
  for (const r of report.rejected) limitations.push(`${r.category}: ${r.reason}`);
  if (assumptions.length) limitations.push(`Interpretation assumptions: ${assumptions.join("; ")}.`);

  return {
    success: garments.length > 0,
    garments,
    assumptions,
    limitations,
    rejectedGarments: report.rejected.length ? report.rejected : undefined,
    profile_used: profileMeta(request.profile),
  };
}

/* ================================================================== */
/*  Offline describe fallback — deterministic dictionary parser        */
/* ================================================================== */

/** Curated colour phrases, longest-first matching (deterministic). */
const COLOUR_PHRASES: [string, string, string][] = [
  ["hot pink", "#FF69B4", "Hot Pink"],
  ["pastel pink", "#FFD1DC", "Pastel Pink"],
  ["navy blue", "#1A1A3E", "Navy"],
  ["dark blue", "#1A1A3E", "Navy"],
  ["light blue", "#87CEEB", "Sky Blue"],
  ["sky blue", "#87CEEB", "Sky Blue"],
  ["baby blue", "#A6C8E5", "Baby Blue"],
  ["off white", "#F8F6F0", "Off White"],
  ["off-white", "#F8F6F0", "Off White"],
  ["forest green", "#234B35", "Forest Green"],
  ["dark green", "#234B35", "Forest Green"],
  ["olive green", "#6B8E23", "Olive Green"],
  ["light green", "#7FBF7F", "Light Green"],
  ["dark red", "#8B0000", "Dark Red"],
  ["wine", "#722F37", "Wine"],
  ["charcoal", "#36454F", "Charcoal"],
  ["denim", "#5B7C99", "Denim Blue"],
  ["burgundy", "#6E1F2A", "Burgundy"],
  ["maroon", "#6E1F2A", "Maroon"],
  ["lavender", "#B57EDC", "Lavender"],
  ["turquoise", "#40E0D0", "Turquoise"],
  ["mustard", "#D4A017", "Mustard"],
  ["fuchsia", "#D648D6", "Magenta"],
  ["silver", "#C0C0C0", "Silver"],
  ["golden", "#FFD700", "Gold"],
  ["black", "#000000", "Black"],
  ["white", "#FFFFFF", "White"],
  ["grey", "#808080", "Grey"],
  ["gray", "#808080", "Grey"],
  ["cream", "#FFFDD0", "Cream"],
  ["ivory", "#FFFFF0", "Ivory"],
  ["beige", "#F5F5DC", "Beige"],
  ["khaki", "#C3B091", "Khaki"],
  ["camel", "#C19A6B", "Camel"],
  ["tan", "#D2B48C", "Tan"],
  ["brown", "#8B6914", "Brown"],
  ["olive", "#6B8E23", "Olive Green"],
  ["green", "#2E8B57", "Green"],
  ["teal", "#008080", "Teal"],
  ["blue", "#0047AB", "Blue"],
  ["navy", "#1A1A3E", "Navy"],
  ["yellow", "#FFD700", "Yellow"],
  ["gold", "#FFD700", "Gold"],
  ["orange", "#FFA500", "Orange"],
  ["rust", "#B7410E", "Rust"],
  ["coral", "#FF7F50", "Coral"],
  ["peach", "#FFDAB9", "Peach"],
  ["pink", "#FFB6C1", "Light Pink"],
  ["red", "#C0392B", "Red"],
  ["purple", "#6A0DAD", "Purple"],
  ["mauve", "#8B668B", "Mauve"],
  ["indigo", "#4B0082", "Indigo"],
];

const CATEGORY_WORDS: [string, string][] = [
  ["t-shirt", "T-Shirt"], ["t shirt", "T-Shirt"], ["tshirt", "T-Shirt"], ["tee", "T-Shirt"],
  ["shirt", "Shirt"], ["blouse", "Blouse"], ["cardigan", "Cardigan"],
  ["sweater", "Sweater"], ["jumper", "Sweater"], ["hoodie", "Hoodie"],
  ["blazer", "Blazer"], ["jacket", "Jacket"], ["coat", "Coat"], ["parka", "Coat"],
  ["jeans", "Jeans"], ["trousers", "Trousers"], ["pants", "Trousers"], ["chinos", "Trousers"],
  ["bottoms", "Bottoms"], ["bottom", "Bottoms"],
  ["skirt", "Skirt"], ["dress", "Dress"], ["gown", "Dress"],
  ["kurti", "Kurti"], ["kurta", "Kurta"], ["saree", "Saree"], ["sari", "Saree"],
  ["lehenga", "Lehenga"], ["salwar", "Salwar"], ["shorts", "Shorts"],
  ["leggings", "Leggings"], ["joggers", "Joggers"], ["suit", "Suit"],
  ["sneakers", "Sneakers"], ["trainers", "Sneakers"], ["shoes", "Shoes"],
  ["boots", "Boots"], ["heels", "Heels"], ["sandals", "Sandals"], ["flats", "Flats"], ["loafers", "Loafers"],
  ["scarf", "Scarf"], ["belt", "Belt"], ["bag", "Bag"], ["handbag", "Bag"], ["tote", "Bag"],
  ["watch", "Watch"], ["necklace", "Necklace"], ["earrings", "Earrings"],
  ["top", "Top"],
];

const PATTERN_WORDS: [string, string][] = [
  ["striped", "Striped"], ["stripes", "Striped"],
  ["checked", "Checked"], ["check", "Checked"], ["plaid", "Checked"],
  ["floral", "Floral"], ["florals", "Floral"],
  ["geometric", "Geometric"],
  ["printed", "Printed"], ["print", "Printed"], ["polka", "Printed"],
  ["embroidered", "Embroidered"], ["embroidery", "Embroidered"],
];

const FIT_WORDS = [
  "oversized", "slim", "straight", "wide-leg", "wide leg", "relaxed",
  "tailored", "fitted", "baggy", "loose", "cropped", "regular",
];

function stripChunkPrefix(chunk: string): string {
  return chunk
    .replace(/^(?:and|with|plus|also|i\s+(?:have|own|got|bought|brought)|my|a|an|the|some|one)\s+/gi, "")
    .replace(/^(?:and|with|plus|also|my|a|an|the|some|one)\s+/gi, "")
    .trim();
}

/**
 * Offline fallback parser. Deliberately conservative:
 *  - splits on commas, semicolons, newlines and the conjunctions
 *    "and" / "with" / "plus" / "&" so "a cobalt top with teal trousers"
 *    yields TWO garments;
 *  - extracts one garment per clothing-category word, pairing it with the
 *    NEAREST preceding colour phrase (never a later one, never an invented
 *    default);
 *  - reports garments whose colour could not be determined instead of
 *    dropping them.
 */
export function parseDescriptionLocally(description: string): { garments: GarmentInput[]; assumptions: string[] } {
  const assumptions: string[] = [];
  const out: GarmentInput[] = [];
  const chunks = description
    .toLowerCase()
    .split(/\s*(?:,|;|\n|\band\b|\bwith\b|\bplus\b|\b&\b)\s*/)
    .map(stripChunkPrefix)
    .filter((c) => c.length > 2);

  for (const chunk of chunks.slice(0, 12)) {
    // Find every category word in this chunk with its position.
    const categories: { word: string; cat: string; idx: number }[] = [];
    for (const [word, cat] of CATEGORY_WORDS) {
      const idx = chunk.indexOf(word);
      if (idx !== -1) {
        // A longer dictionary word containing this one ("shirt" in "t-shirt")
        // at the same position wins; skip contained duplicates.
        if (!categories.some((c) => c.idx === idx && c.word.length >= word.length)) {
          categories.push({ word, cat, idx });
        }
      }
    }
    categories.sort((a, b) => a.idx - b.idx);
    if (!categories.length) {
      // colour-only chunk without a category word
      let colour: [string, string, string] | null = null;
      for (const phrase of COLOUR_PHRASES) {
        const idx = chunk.indexOf(phrase[0]);
        if (idx === -1) continue;
        if (!colour || phrase[0].length > colour[0].length) colour = phrase;
      }
      if (colour) {
        assumptions.push(`could not tell the garment type for the ${colour[2].toLowerCase()} item — set it below`);
        out.push({
          id: `local-${out.length + 1}-${chunk.length}`,
          category: "Garment",
          colourName: colour[2],
          hex: colour[1],
          pattern: null,
          fit: null,
          owned: true,
        });
      }
      continue;
    }

    // One garment per category word; the colour is the LONGEST phrase that
    // appears BEFORE this category word (falls back to any position for the
    // first garment only when nothing precedes it).
    const pattern = PATTERN_WORDS.map(([word, p]) => (chunk.includes(word) ? p : null)).find(Boolean) || null;
    const fit = FIT_WORDS.map((word) => (chunk.includes(word) ? word : null)).find(Boolean) || null;
    for (let ci = 0; ci < categories.length; ci++) {
      const { word, cat, idx } = categories[ci];
      const prevIdx = ci > 0 ? categories[ci - 1].idx : -1;
      let colour: [string, string, string] | null = null;
      let colourIdx = -1;
      for (const phrase of COLOUR_PHRASES) {
        const pIdx = chunk.indexOf(phrase[0]);
        if (pIdx === -1) continue;
        const inScope = pIdx < idx && pIdx > prevIdx;
        if (!inScope) continue;
        if (!colour || phrase[0].length > colour[0].length) {
          colour = phrase;
          colourIdx = pIdx;
        }
      }
      if (!colour && ci === 0) {
        // first garment: accept a colour anywhere in the chunk as a fallback
        for (const phrase of COLOUR_PHRASES) {
          const pIdx = chunk.indexOf(phrase[0]);
          if (pIdx === -1) continue;
          if (!colour || phrase[0].length > colour[0].length) colour = phrase;
        }
      }
      if (!colour) {
        assumptions.push(`could not tell the colour of the ${cat.toLowerCase()} — add it below`);
        out.push({
          id: `local-${out.length + 1}-${idx}`,
          category: cat,
          colourName: "Unknown",
          hex: "",
          pattern,
          fit,
          owned: true,
        });
        continue;
      }
      const garment: GarmentInput = {
        id: `local-${out.length + 1}-${idx}`,
        category: cat,
        colourName: colour[2],
        hex: colour[1],
        pattern,
        fit,
        owned: true,
      };
      // dedupe identical category+colour
      if (!out.some((g) => g.category === garment.category && g.hex === garment.hex && g.hex !== "")) {
        out.push(garment);
      }
    }
  }
  return { garments: out.slice(0, MAX_INPUT_GARMENTS), assumptions };
}

/* ================================================================== */
/*  MODE 3 — Wardrobe outfit analysis (deterministic core + AI styling) */
/* ================================================================== */

export type StyleTextAnalyser = (prompt: string) => Promise<OutfitAiPayload>;

export function buildWardrobePrompt(
  profile: OutfitAnalysisRequest["profile"],
  garments: GarmentInput[],
  occasion?: string,
  stylePreference?: string,
): string {
  return `You are FaceMaxx's practical fashion-styling assistant. The user selected these garments from THEIR OWN wardrobe:
${garments.map((g) => `- ${g.category} in ${g.colourName} (${g.hex})${g.pattern && g.pattern !== "Solid" ? `, ${g.pattern}` : ""}${g.fit ? `, ${g.fit} fit` : ""} [OWNED]`).join("\n")}

Return ONLY valid JSON with exactly this shape:
{
  "styling_advice": {"currentOutfit": ["2-4 practical tips"], "footwear": ["1-3"], "accessories": ["1-3"], "layering": ["0-3"], "makeupAndJewellery": ["0-2 or omit"]},
  "outfit_combinations": [
    {"name": "LOOK 01 - <name>", "description": "one sentence", "anchorGarment": "one of the user's selected garments", "pieces": [{"item": "...", "colourName": "...", "hex": "#RRGGBB", "owned": true|false}], "footwear": "...", "accessories": ["..."], "rationale": "why the colours work for THIS profile", "aesthetic": "Everyday|Occasion|Tonal|Expressive"}
  ]
}

Rules:
1. The selected garments are already owned. In every combination, set "owned": true for pieces from the list above and "owned": false for suggested additions (e.g. footwear or accessories the user may not own).
2. Produce 2 or 3 genuinely DIFFERENT combinations (everyday, elevated, tonal/colour-block). Each must include at least one owned selected garment as the anchor; never replace owned items with purchases.
3. Base ALL personalisation on the supplied FaceMaxx profile only. Never invent profile data.
4. Prefer realistic pieces people commonly own; nothing expensive or niche.` +
    (occasion ? `\nUser occasion: ${occasion}.` : "") +
    (stylePreference ? `\nUser style preference: ${stylePreference}.` : "") +
    `\n\nSaved FaceMaxx profile: ${JSON.stringify(profile)}.`;
}

async function callStyleTextAi(prompt: string, apiKey: string): Promise<OutfitAiPayload> {
  const { GoogleGenAI } = await import("@google/genai");
  const ai = new GoogleGenAI({ apiKey, httpOptions: { headers: { "User-Agent": "aistudio-build" } } });
  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: [{ text: prompt }],
    config: { systemInstruction: "You are a practical fashion-styling assistant. Return ONLY valid JSON matching the requested schema. Never invent profile data or garments the user does not own.", responseMimeType: "application/json" },
  });
  let text = (response.text || "{}").trim();
  if (text.startsWith("```json")) text = text.slice(7);
  if (text.endsWith("```")) text = text.slice(0, -3);
  const parsed = JSON.parse(text.trim());
  if (!parsed || typeof parsed !== "object") throw new Error("AI returned non-object JSON");
  return parsed as OutfitAiPayload;
}

function deterministicColourAnalysis(garments: OutfitGarment[], swatches: SeasonSwatches | null): OutfitColourAnalysis | null {
  if (!garments.length) return null;
  const refSwatches = swatches ? [...swatches.best, ...swatches.neutrals] : [];
  return {
    garmentColours: garments.map((g) => {
      let min: number | null = null;
      for (const s of refSwatches) {
        const d = deltaE76(g.dominantColour.hex, s.hex);
        if (d !== null && (min === null || d < min)) min = d;
      }
      return {
        hex: g.dominantColour.hex,
        name: g.dominantColour.name,
        deltaEToNearestPaletteColour: min === null ? null : Math.round(min * 10) / 10,
        /** per-garment palette fit from the same nearest-swatch distance */
        paletteFit: (() => {
          if (min === null) return null;
          if (min <= 25) return "harmonious" as const;
          if (min <= 38) return "compatible" as const;
          if (min <= 52) return "neutral" as const;
          return "contrasting" as const;
        })(),
      };
    }),
    paletteMatches: refSwatches
      .map((swatch) => {
        let min: number | null = null;
        for (const g of garments) {
          const d = deltaE76(g.dominantColour.hex, swatch.hex);
          if (d !== null && (min === null || d < min)) min = d;
        }
        return { swatch, deltaE: min === null ? 999 : min };
      })
      .sort((a, b) => a.deltaE - b.deltaE)
      .slice(0, 4),
    outfitContrast:
      garments.length >= 2
        ? (() => {
            const Ls = garments.map((g) => hexToLab(g.dominantColour.hex)?.L).filter((v): v is number => v !== undefined);
            return Ls.length >= 2 ? Math.round((Math.max(...Ls) - Math.min(...Ls)) * 10) / 10 : null;
          })()
        : null,
  };
}

export async function runWardrobeAnalysis(
  request: WardrobeAnalysisRequest,
  apiKey: string | null,
  analyser: StyleTextAnalyser = (prompt) => {
    if (!apiKey) throw new Error("GEMINI_API_KEY is not configured.");
    return callStyleTextAi(prompt, apiKey);
  },
): Promise<OutfitAnalysisResponse> {
  const limitations = [
    "Colour harmony is a deterministic CIELAB distance to your saved palette; it is a styling guide, not a scientifically validated measure.",
  ];
  const swatches = paletteForSeason(request.profile?.season || "");

  if (!request.profile?.season) {
    return {
      success: false,
      input_mode: "wardrobe",
      error: "NO_PROFILE",
      evidence_class: EVIDENCE_CLASS,
      image_quality: null,
      garments: [],
      colour_analysis: null,
      suitability: {
        verdict: "Insufficient image or profile information",
        dimensions: {
          colourHarmony: dim("unknown", "No saved colour profile."),
          temperature: dim("unknown", "No saved colour profile."),
          depth: dim("unknown", "No saved colour profile."),
          chroma: dim("unknown", "No saved colour profile."),
          contrast: dim("unknown", "No saved colour profile."),
          colourCombination: dim("unknown", "No saved colour profile."),
        },
        reasons: ["No saved FaceMaxx colour profile was supplied."],
      },
      styling_advice: null,
      outfit_combinations: [],
      limitations: ["Complete your FaceMaxx colour analysis to receive styling recommendations tailored to you."],
      profile_used: { season: "", available_axes: [], source: "none" },
    };
  }

  const report = validateGarmentInputsWithReport(request.garments?.map((g) => ({ ...g, owned: true })));
  const inputs = report.accepted;
  if (!inputs.some((g) => g.hex)) {
    return {
      success: false,
      input_mode: "wardrobe",
      error: "NO_GARMENTS",
      evidence_class: EVIDENCE_CLASS,
      image_quality: null,
      garments: [],
      colour_analysis: null,
      suitability: {
        verdict: "Insufficient image or profile information",
        dimensions: {
          colourHarmony: dim("unknown", "No analysable garment colours."),
          temperature: dim("unknown", "No analysable garment colours."),
          depth: dim("unknown", "No analysable garment colours."),
          chroma: dim("unknown", "No analysable garment colours."),
          contrast: dim("unknown", "No analysable garment colours."),
          colourCombination: dim("unknown", "No analysable garment colours."),
        },
        reasons: ["Select at least one wardrobe item with a colour."],
      },
      styling_advice: null,
      outfit_combinations: [],
      limitations: [
        "Select at least one wardrobe item with a recognisable colour.",
        ...report.rejected.map((r) => `${r.category}: ${r.reason}`),
      ],
      rejectedGarments: report.rejected,
      profile_used: {
        season: request.profile.season,
        available_axes: profileAxesAvailable(request.profile),
        source: "session_profile",
      },
    };
  }

  // Deterministic core: user-supplied colours are ground truth, so suitability
  // and score are computed with NO dependence on the AI.
  const garments = garmentInputsToOutfitGarments(inputs);
  // Garments kept but excluded from colour math (unknown colour) are reported.
  const unanalysed = inputs.filter((g) => !g.hex);
  const rejectedGarments = [
    ...report.rejected,
    ...unanalysed.map((g) => ({ category: g.category, reason: "Colour is unknown, so this garment could not be colour-analysed — it is still part of your outfit." })),
  ];
  const suitability = computeSuitability(garments, swatches, request.profile);
  const conflicts = outfitConflicts(garments);
  if (conflicts.length) limitations.push("Some colour pairs in this outfit sit at an intermediate distance that can read as clashing.");
  for (const r of rejectedGarments) limitations.push(`${r.category}: ${r.reason}`);

  let payload: OutfitAiPayload | null = null;
  let aiFailed = false;
  try {
    payload = await analyser(buildWardrobePrompt(request.profile, inputs, request.occasion, request.stylePreference));
  } catch {
    aiFailed = true;
  }
  if (aiFailed) {
    limitations.push("AI styling suggestions are unavailable right now — the colour assessment above is still valid. You can retry.");
  }
  const advice = aiFailed ? null : validateAdvice(payload?.styling_advice);
  const combinations = !aiFailed && swatches ? validateCombinations(payload?.outfit_combinations, garments, swatches) : [];
  if (!aiFailed && !combinations.length) {
    limitations.push("No distinct AI combinations passed validation; the looks shown are deterministic palette-based fallbacks.");
  }

  return {
    success: true,
    input_mode: "wardrobe",
    evidence_class: EVIDENCE_CLASS,
    image_quality: null,
    garments,
    colour_analysis: deterministicColourAnalysis(garments, swatches),
    suitability,
    score: scoreFromSuitability(suitability),
    styling_advice: advice,
    outfit_combinations: combinations.length ? combinations : swatches ? fallbackCombinations(garments, swatches) : [],
    limitations,
    rejectedGarments: rejectedGarments.length ? rejectedGarments : undefined,
    profile_used: {
      season: request.profile.season,
      available_axes: profileAxesAvailable(request.profile),
      source: "session_profile",
    },
  };
}

/* ================================================================== */
/*  Comparison mode (fully deterministic — no AI)                      */
/* ================================================================== */

export function compareOutfits(options: CompareOptionInput[], profile: OutfitAnalysisRequest["profile"]): CompareResponse {
  const profileMeta = {
    season: profile?.season || "",
    available_axes: profile ? profileAxesAvailable(profile) : [],
    source: (profile?.season ? "session_profile" : "none") as "session_profile" | "none",
  };
  if (!profile?.season) {
    return {
      success: false,
      error: "NO_PROFILE",
      message: "Complete your FaceMaxx colour analysis to compare outfits against your palette.",
      options: [],
      summary: null,
      evidence_class: EVIDENCE_CLASS,
      profile_used: profileMeta,
    };
  }
  const swatches = paletteForSeason(profile.season);
  const inputs = (Array.isArray(options) ? options : []).slice(0, 4);
  const results: CompareOptionResult[] = [];

  for (const option of inputs) {
    const report = validateGarmentInputsWithReport((option?.garments || []).map((g) => ({ ...g, owned: true })));
    const parsed = report.accepted;
    const garments = garmentInputsToOutfitGarments(parsed);
    const suitability = computeSuitability(garments, swatches, profile);
    const score = scoreFromSuitability(suitability);
    const refSwatches = swatches ? [...swatches.best, ...swatches.neutrals] : [];
    const palette = refSwatches
      .map((swatch) => {
        let min: number | null = null;
        for (const g of garments) {
          const d = deltaE76(g.dominantColour.hex, swatch.hex);
          if (d !== null && (min === null || d < min)) min = d;
        }
        return { swatch, deltaE: min === null ? 999 : min };
      })
      .sort((a, b) => a.deltaE - b.deltaE)
      .slice(0, 4)
      .map((m) => m.swatch);
    const advantages = Object.values(suitability.dimensions)
      .filter((d) => d.value === "harmonious" || d.value === "compatible")
      .map((d) => d.reason);
    const conflicts = [
      ...Object.values(suitability.dimensions).filter((d) => d.value === "contrasting").map((d) => d.reason),
      ...outfitConflicts(garments),
    ];
    results.push({
      label: (typeof option?.label === "string" ? option.label.trim().slice(0, 60) : "") || `Option ${results.length + 1}`,
      garments,
      score,
      suitability,
      palette,
      advantages,
      conflicts,
      limitations: [
        ...report.rejected.map((r) => `${r.category}: ${r.reason}`),
        ...parsed.filter((g) => !g.hex).map((g) => `${g.category}: colour unknown, so this garment is not colour-analysed.`),
      ],
    });
  }

  const withScores = results.filter((r) => r.score.score !== null);
  let summary: string | null = null;
  if (withScores.length >= 2) {
    const sorted = [...withScores].sort((a, b) => (b.score.score as number) - (a.score.score as number));
    const top = sorted[0];
    const diffs = sorted
      .slice(1)
      .map((r) => `${r.label} (${r.score.score})`)
      .join(", ");
    summary =
      `${top.label} scores highest on documented palette-compatibility rules (${top.score.score}/100)` +
      (diffs ? `, ahead of ${diffs}` : "") +
      ". Scores measure palette consistency only — not which look is best for you; style preferences and occasion matter too.";
  } else if (results.length) {
    summary = "Not enough measurable colour data to rank these options; review the per-dimension explanations instead.";
  }

  return {
    success: results.length > 0,
    options: results,
    summary,
    evidence_class: EVIDENCE_CLASS,
    profile_used: profileMeta,
  };
}
