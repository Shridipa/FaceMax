/**
 * Frontend-derived styling recommendations.
 *
 * These are deterministic mappings from the backend-returned
 * color profile (season, undertone, depth, chroma) to practical
 * styling guidance. They are NOT produced by the ML model.
 *
 * Clearly separated from the analysis pipeline.
 */

export interface ClothingRecommendation {
  category: string;
  items: { color: string; note: string }[];
}

export interface MakeupRecommendation {
  category: string;
  shades: { name: string; hex?: string }[];
  note: string;
}

export interface JewelryGuidance {
  metal: string;
  verdict: "best" | "good" | "use_carefully";
  note: string;
}

export interface HairGuidance {
  direction: string;
  verdict: "best" | "good" | "use_carefully";
  note: string;
}

export interface OutfitCombination {
  label: string;
  items: { part: string; color: string; hex: string }[];
}

export interface FullStyleGuide {
  clothing: ClothingRecommendation[];
  makeup: MakeupRecommendation[];
  jewelry: JewelryGuidance[];
  hair: HairGuidance[];
  outfit: OutfitCombination[];
}

/* ------------------------------------------------------------------ */
/*  Season-based palette data (same as api.ts but exported for styling) */
/* ------------------------------------------------------------------ */
import { SEASON_DESCRIPTIONS } from "../types/api";
import type { ColourAnalysisResponse } from "../types/api";

/* ------------------------------------------------------------------ */
/*  Clothing recommendations per season family                         */
/* ------------------------------------------------------------------ */
const CLOTHING_BY_SEASON: Record<string, ClothingRecommendation[]> = {
  Spring: [
    { category: "Everyday", items: [
      { color: "Warm peach tops", note: "Soft warm tones near the face bring out a natural glow." },
      { color: "Golden camel blouses", note: "Pairs well with warm neutrals for a cohesive everyday look." },
      { color: "Coral or warm pink dresses", note: "Adds vibrancy without overwhelming your coloring." },
    ]},
    { category: "Formal", items: [
      { color: "Warm gold or champagne", note: "Elegant for evening events, harmonizes with warm undertones." },
      { color: "Teal or warm aqua gowns", note: "A refreshing yet sophisticated choice." },
    ]},
    { category: "Outerwear", items: [
      { color: "Camel or warm tan coats", note: "A classic neutral that flatters spring coloring." },
      { color: "Olive or warm sage jackets", note: "Adds depth without clashing." },
    ]},
    { category: "Denim", items: [
      { color: "Medium-wash blue", note: "The warmth in spring coloring pairs well with classic blue denim." },
      { color: "Warm-toned white jeans", note: "Better than stark white for spring types." },
    ]},
    { category: "Accessories", items: [
      { color: "Gold or brass jewelry", note: "Complements the warm undertone naturally." },
      { color: "Warm brown leather bags", note: "A refined neutral accessory." },
    ]},
  ],
  Summer: [
    { category: "Everyday", items: [
      { color: "Soft lavender or lilac tops", note: "Cool, muted tones that blend with your soft coloring." },
      { color: "Dusty rose blouses", note: "A flattering cool-toned pink." },
      { color: "Soft slate or blue-gray", note: "A sophisticated everyday neutral." },
    ]},
    { category: "Formal", items: [
      { color: "Soft navy or periwinkle", note: "Elegant and cool-toned for formal occasions." },
      { color: "Mauve or dusty plum", note: "Adds richness while staying within your cool palette." },
    ]},
    { category: "Outerwear", items: [
      { color: "Gray or dove-gray coats", note: "A cool-toned classic." },
      { color: "Soft teal or slate blazers", note: "Professional and harmonious." },
    ]},
    { category: "Denim", items: [
      { color: "Light to medium wash", note: "Cool-toned denim works well for summer types." },
      { color: "Gray or faded blue jeans", note: "Avoid overly warm or golden washes." },
    ]},
    { category: "Accessories", items: [
      { color: "Silver or platinum jewelry", note: "Cool metals complement your undertone." },
      { color: "Cool-toned taupe or gray bags", note: "Refined and cohesive." },
    ]},
  ],
  Autumn: [
    { category: "Everyday", items: [
      { color: "Olive and forest green tops", note: "Earth tones that naturally harmonize with autumn coloring." },
      { color: "Warm terracotta or rust blouses", note: "Rich, warm tones that bring out depth." },
      { color: "Chocolate brown knits", note: "A quintessential autumn neutral." },
    ]},
    { category: "Formal", items: [
      { color: "Deep burgundy or wine", note: "Rich and sophisticated for evening wear." },
      { color: "Warm plum or aubergine", note: "Adds drama while staying in your palette." },
    ]},
    { category: "Outerwear", items: [
      { color: "Camel or warm tan coats", note: "A timeless autumn staple." },
      { color: "Olive or khaki jackets", note: "Practical and color-harmonious." },
    ]},
    { category: "Denim", items: [
      { color: "Warm medium wash", note: "Avoid overly cool or bleached denim." },
      { color: "Chocolate or warm brown corduroy", note: "A rich alternative to blue denim." },
    ]},
    { category: "Accessories", items: [
      { color: "Gold or bronze jewelry", note: "Warm metals that complement your undertone." },
      { color: "Warm brown or cognac leather", note: "A natural accessory choice." },
    ]},
  ],
  Winter: [
    { category: "Everyday", items: [
      { color: "True red or crimson tops", note: "Bold, clear colors that match your high-contrast coloring." },
      { color: "Navy or black basics", note: "Classic dark neutrals for winter types." },
      { color: "Emerald or jewel-toned blouses", note: "Vivid colors that complement your depth." },
    ]},
    { category: "Formal", items: [
      { color: "Black or deep navy gowns", note: "Timeless elegance for winter coloring." },
      { color: "Fuchsia or electric blue", note: "Statement colors that harmonize." },
    ]},
    { category: "Outerwear", items: [
      { color: "Black or charcoal coats", note: "Sharp, high-contrast neutrals." },
      { color: "Deep navy blazers", note: "Professional and flattering." },
    ]},
    { category: "Denim", items: [
      { color: "Dark indigo or black denim", note: "High-contrast dark washes suit winter types." },
      { color: "Avoid very light or golden washes", note: "These can wash out winter coloring." },
    ]},
    { category: "Accessories", items: [
      { color: "Silver or platinum jewelry", note: "Cool metals that complement your undertone." },
      { color: "Black or deep navy leather", note: "Sleek and cohesive." },
    ]},
  ],
};

/* ------------------------------------------------------------------ */
/*  Makeup recommendations per season family                           */
/* ------------------------------------------------------------------ */
const MAKEUP_BY_SEASON: Record<string, MakeupRecommendation[]> = {
  Spring: [
    { category: "Lips", shades: [
      { name: "Warm coral" }, { name: "Peach" }, { name: "Warm berry" }, { name: "Rose gold" }
    ], note: "Warm, fresh tones that enhance your natural warmth." },
    { category: "Blush", shades: [
      { name: "Peach" }, { name: "Warm rose" }, { name: "Soft apricot" }
    ], note: "Subtle warmth adds a natural flush." },
    { category: "Eyes", shades: [
      { name: "Warm bronze" }, { name: "Peach gold" }, { name: "Soft olive" }, { name: "Warm taupe" }
    ], note: "Golden and warm earth tones complement your coloring." },
    { category: "Nails", shades: [
      { name: "Coral" }, { name: "Warm pink" }, { name: "Peach nude" }
    ], note: "Fresh, warm tones." },
  ],
  Summer: [
    { category: "Lips", shades: [
      { name: "Dusty rose" }, { name: "Mauve" }, { name: "Soft berry" }, { name: "Rose" }
    ], note: "Cool, muted tones that harmonize with your soft coloring." },
    { category: "Blush", shades: [
      { name: "Soft pink" }, { name: "Mauve" }, { name: "Rose" }
    ], note: "Cool-toned pinks add a natural glow." },
    { category: "Eyes", shades: [
      { name: "Soft slate" }, { name: "Lavender" }, { name: "Cool taupe" }, { name: "Blue-gray" }
    ], note: "Muted cool tones that blend with your coloring." },
    { category: "Nails", shades: [
      { name: "Mauve" }, { name: "Dusty pink" }, { name: "Soft lilac" }
    ], note: "Cool, refined tones." },
  ],
  Autumn: [
    { category: "Lips", shades: [
      { name: "Brick red" }, { name: "Warm terracotta" }, { name: "Deep berry" }, { name: "Warm wine" }
    ], note: "Rich, warm tones that complement your depth." },
    { category: "Blush", shades: [
      { name: "Terracotta" }, { name: "Warm rose" }, { name: "Cinnamon" }
    ], note: "Earthy warmth adds dimension." },
    { category: "Eyes", shades: [
      { name: "Espresso" }, { name: "Bronze" }, { name: "Olive" }, { name: "Deep chocolate" }
    ], note: "Rich earth tones that harmonize with your coloring." },
    { category: "Nails", shades: [
      { name: "Rust" }, { name: "Warm brown" }, { name: "Olive" }
    ], note: "Deep, warm tones." },
  ],
  Winter: [
    { category: "Lips", shades: [
      { name: "True red" }, { name: "Deep berry" }, { name: "Fuchsia" }, { name: "Wine" }
    ], note: "Bold, clear colors that match your high contrast." },
    { category: "Blush", shades: [
      { name: "Cool rose" }, { name: "Berry" }, { name: "Plum" }
    ], note: "Cool-toned definition." },
    { category: "Eyes", shades: [
      { name: "Charcoal" }, { name: "Deep navy" }, { name: "Black" }, { name: "Cool plum" }
    ], note: "High-contrast, dramatic tones." },
    { category: "Nails", shades: [
      { name: "True red" }, { name: "Deep berry" }, { name: "Black" }
    ], note: "Bold, clear tones." },
  ],
};

/* ------------------------------------------------------------------ */
/*  Jewelry guidance per undertone                                     */
/* ------------------------------------------------------------------ */
const JEWELRY_BY_UNDERTONE: Record<string, JewelryGuidance[]> = {
  Warm: [
    { metal: "Gold", verdict: "best", note: "The warm yellow tone harmonizes with your natural warmth." },
    { metal: "Rose Gold", verdict: "good", note: "A warm alternative that complements most warm undertones." },
    { metal: "Silver", verdict: "use_carefully", note: "Can work in small amounts, but gold is more harmonious." },
  ],
  "Neutral-Warm": [
    { metal: "Gold", verdict: "best", note: "Warm gold flatters your slightly warm undertone." },
    { metal: "Rose Gold", verdict: "best", note: "The balance of warm and cool in rose gold suits you." },
    { metal: "Silver", verdict: "good", note: "Works as an accent alongside warmer metals." },
  ],
  Neutral: [
    { metal: "Gold", verdict: "good", note: "Works well, especially in smaller pieces." },
    { metal: "Silver", verdict: "good", note: "Equally flattering for neutral undertones." },
    { metal: "Rose Gold", verdict: "best", note: "The blend of warm and cool is ideal for neutral coloring." },
  ],
  "Neutral-Cool": [
    { metal: "Silver", verdict: "best", note: "Cool-toned metals harmonize with your undertone." },
    { metal: "Rose Gold", verdict: "best", note: "The slight warmth balances your cool undertone beautifully." },
    { metal: "Gold", verdict: "good", note: "Works in smaller pieces or as accents." },
  ],
  Cool: [
    { metal: "Silver", verdict: "best", note: "Cool metallic tones complement your undertone naturally." },
    { metal: "Platinum", verdict: "best", note: "A luxurious cool-toned option." },
    { metal: "Gold", verdict: "use_carefully", note: "Warm gold can clash; opt for white gold instead." },
  ],
  Olive: [
    { metal: "Gold", verdict: "best", note: "Warm gold brings out the golden undertones in olive skin." },
    { metal: "Bronze", verdict: "best", note: "A rich warm metal that harmonizes beautifully." },
    { metal: "Silver", verdict: "use_carefully", note: "Can appear stark against olive tones." },
  ],
};

/* ------------------------------------------------------------------ */
/*  Hair color guidance per season                                     */
/* ------------------------------------------------------------------ */
const HAIR_BY_SEASON: Record<string, HairGuidance[]> = {
  Spring: [
    { direction: "Warm golden highlights", verdict: "best", note: "Adds dimension while complementing your warm coloring." },
    { direction: "Honey or caramel tones", verdict: "best", note: "Natural-looking warmth that harmonizes." },
    { direction: "Warm copper or auburn", verdict: "good", note: "A richer option that stays within your warm palette." },
    { direction: "Icy platinum or cool ash", verdict: "use_carefully", note: "Cool tones can clash with your warm coloring." },
  ],
  Summer: [
    { direction: "Cool ash brown", verdict: "best", note: "A natural-looking cool tone that harmonizes." },
    { direction: "Soft lavender or mauve highlights", verdict: "good", note: "A subtle cool-toned accent." },
    { direction: "Cool-toned blonde", verdict: "best", note: "Ash or pearl blonde complements your cool undertone." },
    { direction: "Warm golden or copper", verdict: "use_carefully", note: "Warm tones can clash with your cool coloring." },
  ],
  Autumn: [
    { direction: "Rich chocolate brown", verdict: "best", note: "A quintessential autumn hair color." },
    { direction: "Warm auburn or copper", verdict: "best", note: "Deep, warm tones that harmonize beautifully." },
    { direction: "Warm brunette", verdict: "best", note: "Natural-looking warmth." },
    { direction: "Icy platinum or cool blonde", verdict: "use_carefully", note: "Cool tones clash with warm autumn coloring." },
  ],
  Winter: [
    { direction: "Deep espresso or jet black", verdict: "best", note: "High-contrast dark tones suit winter coloring." },
    { direction: "Cool-toned dark brown", verdict: "best", note: "Rich and harmonious." },
    { direction: "Bold blue-black", verdict: "good", note: "A dramatic option that works with winter depth." },
    { direction: "Warm golden or honey blonde", verdict: "use_carefully", note: "Warm tones can look unnatural against cool coloring." },
  ],
};

/* ------------------------------------------------------------------ */
/*  Outfit combinations per season                                     */
/* ------------------------------------------------------------------ */
const OUTFITS_BY_SEASON: Record<string, OutfitCombination[]> = {
  Spring: [
    { label: "Everyday casual", items: [
      { part: "Top", color: "Warm peach blouse", hex: "#FFDAB9" },
      { part: "Bottom", color: "Camel chinos", hex: "#C19A6B" },
      { part: "Shoes", color: "Warm tan loafers", hex: "#A0846B" },
      { part: "Accessory", color: "Gold pendant", hex: "#FFD700" },
    ]},
    { label: "Office ready", items: [
      { part: "Top", color: "Warm coral blazer", hex: "#F08080" },
      { part: "Bottom", color: "Cream trousers", hex: "#FFFDD0" },
      { part: "Bag", color: "Warm brown leather", hex: "#8B6914" },
    ]},
  ],
  Summer: [
    { label: "Everyday casual", items: [
      { part: "Top", color: "Soft lavender tee", hex: "#E6E6FA" },
      { part: "Bottom", color: "Gray denim", hex: "#9E9E9E" },
      { part: "Shoes", color: "Cool taupe sneakers", hex: "#8B7D6B" },
      { part: "Accessory", color: "Silver bracelet", hex: "#C0C0C0" },
    ]},
    { label: "Office ready", items: [
      { part: "Top", color: "Dusty rose blouse", hex: "#DCAE96" },
      { part: "Bottom", color: "Soft navy skirt", hex: "#4E6172" },
      { part: "Bag", color: "Cool gray tote", hex: "#A9A9A9" },
    ]},
  ],
  Autumn: [
    { label: "Everyday casual", items: [
      { part: "Top", color: "Olive knit", hex: "#556B2F" },
      { part: "Bottom", color: "Chocolate corduroy", hex: "#3A2418" },
      { part: "Shoes", color: "Cognac boots", hex: "#9A4E1C" },
      { part: "Accessory", color: "Gold hoops", hex: "#FFD700" },
    ]},
    { label: "Office ready", items: [
      { part: "Top", color: "Burgundy blazer", hex: "#6E1F2A" },
      { part: "Bottom", color: "Espresso trousers", hex: "#3A2418" },
      { part: "Bag", color: "Warm brown satchel", hex: "#7B5B3A" },
    ]},
  ],
  Winter: [
    { label: "Everyday casual", items: [
      { part: "Top", color: "True red sweater", hex: "#CC0000" },
      { part: "Bottom", color: "Black jeans", hex: "#000000" },
      { part: "Shoes", color: "Black ankle boots", hex: "#1A1A1A" },
      { part: "Accessory", color: "Silver watch", hex: "#C0C0C0" },
    ]},
    { label: "Office ready", items: [
      { part: "Top", color: "Navy blazer", hex: "#000080" },
      { part: "Bottom", color: "Charcoal trousers", hex: "#36454F" },
      { part: "Bag", color: "Black leather briefcase", hex: "#1A1A1A" },
    ]},
  ],
};

/* ------------------------------------------------------------------ */
/*  Family lookup from season name                                     */
/* ------------------------------------------------------------------ */
function getSeasonFamily(season: string): string {
  if (season.includes("Spring")) return "Spring";
  if (season.includes("Summer")) return "Summer";
  if (season.includes("Autumn")) return "Autumn";
  if (season.includes("Winter")) return "Winter";
  return "Autumn"; // fallback
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */
export function getStyleGuide(data: ColourAnalysisResponse): FullStyleGuide {
  const family = getSeasonFamily(data.season.primary);
  const undertone = data.axes.undertone.prediction;

  return {
    clothing: CLOTHING_BY_SEASON[family] || CLOTHING_BY_SEASON.Autumn,
    makeup: MAKEUP_BY_SEASON[family] || MAKEUP_BY_SEASON.Autumn,
    jewelry: JEWELRY_BY_UNDERTONE[undertone] || JEWELRY_BY_UNDERTONE.Neutral,
    hair: HAIR_BY_SEASON[family] || HAIR_BY_SEASON.Autumn,
    outfit: OUTFITS_BY_SEASON[family] || OUTFITS_BY_SEASON.Autumn,
  };
}
