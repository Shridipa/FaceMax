/**
 * Outfit Analysis & Styling — shared types (frontend + backend).
 *
 * The suitability verdict comes from deterministic CIELAB colour math
 * against the user's saved FaceMaxx palette (api/outfit-core.ts). The AI
 * layer identifies garments, explains the verdict and generates styling
 * combinations; it never invents the colour verdict.
 */

export interface OutfitGarment {
  /** stable identifier carried from the user's input where available */
  id?: string;
  category: string;
  dominantColour: { name: string; hex: string };
  secondaryColour?: { name: string; hex: string } | null;
  pattern: string;
  patternConfidence: number;
  confidence: number;
}

/** A garment supplied by the user that failed validation — reported, never silently dropped. */
export interface RejectedGarment {
  category: string;
  reason: string;
}

export interface OutfitImageQuality {
  brightness: number;
  contrast: number;
  saturation: number;
  sharpness_proxy: number;
  flags: string[];
}

export interface OutfitColourAnalysis {
  garmentColours: { hex: string; name: string; deltaEToNearestPaletteColour: number | null; paletteFit?: "harmonious" | "compatible" | "neutral" | "contrasting" | null }[];
  paletteMatches: { swatch: { name: string; hex: string }; deltaE: number }[];
  outfitContrast: number | null;
}

export interface SuitabilityDimension {
  value: "harmonious" | "compatible" | "neutral" | "contrasting" | "unknown";
  reason: string;
}

export interface OutfitSuitability {
  verdict: "Strong palette harmony" | "Good palette harmony" | "Mixed palette compatibility" | "Limited palette compatibility" | "Insufficient image or profile information";
  dimensions: {
    colourHarmony: SuitabilityDimension;
    temperature: SuitabilityDimension;
    depth: SuitabilityDimension;
    chroma: SuitabilityDimension;
    contrast: SuitabilityDimension;
    colourCombination: SuitabilityDimension;
  };
  reasons: string[];
}

export interface OutfitStylingAdvice {
  currentOutfit: string[];
  footwear: string[];
  accessories: string[];
  layering: string[];
  makeupAndJewellery?: string[];
}

export interface OutfitCombination {
  name: string;
  description: string;
  anchorGarment: string;
  pieces: { item: string; colourName: string; hex: string; owned?: boolean }[];
  footwear: string;
  accessories: string[];
  rationale: string;
  palette: { name: string; hex: string }[];
  aesthetic?: string;
}

/**
 * Deterministic 0–100 weighted compatibility score (engineering parameters,
 * NOT scientifically validated). null when it cannot be computed honestly.
 */
export interface OutfitScore {
  score: number | null;
  breakdown: {
    label: string;
    weight: number;
    percent: number | null;
    value: SuitabilityDimension["value"];
  }[];
  note: string;
}

/** A garment supplied by text description or the wardrobe (never invented by AI). */
export interface GarmentInput {
  id: string;
  category: string;
  colourName: string;
  /** "" means the colour is unknown — never replaced by a default */
  hex: string;
  pattern?: string | null;
  fit?: string | null;
  owned: boolean;
}

export interface DescribeResponse {
  success: boolean;
  error?: string;
  message?: string;
  garments: GarmentInput[];
  assumptions: string[];
  limitations: string[];
  rejectedGarments?: RejectedGarment[];
  profile_used: { season: string; available_axes: string[]; source: "session_profile" | "none" };
}

export interface WardrobeAnalysisRequest {
  garments: {
    id?: string;
    category: string;
    colourName: string;
    hex: string;
    pattern?: string | null;
    fit?: string | null;
  }[];
  profile: OutfitAnalysisRequest["profile"];
  occasion?: string;
  stylePreference?: string;
}

export interface CompareOptionInput {
  label: string;
  garments: {
    id?: string;
    category: string;
    colourName: string;
    hex: string;
    pattern?: string | null;
    fit?: string | null;
  }[];
}

export interface CompareOptionResult {
  label: string;
  /** the complete analysed outfit for this option (all accepted garments) */
  garments: OutfitGarment[];
  score: OutfitScore;
  suitability: OutfitSuitability;
  palette: { name: string; hex: string }[];
  advantages: string[];
  conflicts: string[];
  limitations: string[];
}

export interface CompareResponse {
  success: boolean;
  error?: string;
  message?: string;
  options: CompareOptionResult[];
  summary: string | null;
  evidence_class: OutfitAnalysisResponse["evidence_class"];
  profile_used: { season: string; available_axes: string[]; source: "session_profile" | "none" };
}

export interface OutfitAnalysisResponse {
  success: boolean;
  error?: string;
  input_mode?: "image" | "wardrobe";
  evidence_class: "DETERMINISTIC PALETTE MATCH + AI STYLING (NOT SCIENTIFICALLY VALIDATED)";
  image_quality: OutfitImageQuality | null;
  garments: OutfitGarment[];
  colour_analysis: OutfitColourAnalysis | null;
  suitability: OutfitSuitability;
  /** deterministic weighted score (not AI-generated) */
  score?: OutfitScore;
  styling_advice: OutfitStylingAdvice | null;
  outfit_combinations: OutfitCombination[];
  limitations: string[];
  /** garments supplied by the user but not analysable — reported explicitly, never silently dropped */
  rejectedGarments?: RejectedGarment[];
  profile_used: {
    season: string;
    available_axes: string[];
    source: "session_profile" | "none";
  };
}

export interface OutfitAnalysisRequest {
  /** base64 (data URL allowed) outfit photograph */
  image: string;
  /** the colour profile saved from the completed FaceMaxx analysis */
  profile: {
    season: string;
    undertone?: string;
    skin_depth?: string;
    chroma?: string;
    contrast?: string;
  };
  /** optional user-selected context */
  occasion?: string;
  stylePreference?: string;
}
