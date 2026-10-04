/**
 * Outfit Studio — local deterministic clothing-description parser.
 *
 * The PRIMARY and only extraction mechanism for Describe Clothes:
 * a natural-language wardrobe description ("a cobalt blue oversized shirt,
 * teal wide-leg trousers and beige sneakers") is parsed into canonical
 * GarmentInput objects using static dictionaries only.
 *
 *   normalise -> segment into garment phrases -> match colours & patterns
 *   -> match garment categories (never inside a colour/pattern span)
 *   -> associate each colour/attribute with its nearest category word
 *   -> build garments (unknown colour stays unknown) -> dedupe -> cap
 *
 * Deliberate guarantees:
 *   - NO network, NO AI service, NO API key, NO filesystem, NO randomness.
 *   - Multi-garment: every identified garment phrase becomes its own entry.
 *   - Honest: unknown colours stay "Unknown" with an assumption; ambiguous
 *     colour-only phrases become an editable "Garment" placeholder, never an
 *     invented category.
 *   - Dictionary-first: supporting a new colour/garment/attribute means
 *     adding ONE table entry — no parser logic changes.
 */

import type { GarmentInput } from "./types/outfit.js";

/** Hard cap shared with the outfit engine's input validation. */
const MAX_PARSED_GARMENTS = 12;

/* ------------------------------------------------------------------ */
/*  Dictionaries                                                        */
/* ------------------------------------------------------------------ */

/** [phrase, display name, hex] — alias rows share canonical names/hexes. */
const COLOURS: [string, string, string][] = [
  // compound fashion colours first for readability (matching is longest-first)
  ["cobalt blue", "Cobalt Blue", "#0047AB"],
  ["cobalt", "Cobalt Blue", "#0047AB"],
  ["royal blue", "Royal Blue", "#4169E1"],
  ["powder blue", "Powder Blue", "#B0E0E6"],
  ["sky blue", "Sky Blue", "#87CEEB"],
  ["light blue", "Sky Blue", "#87CEEB"],
  ["dark blue", "Navy", "#1A1A3E"],
  ["navy blue", "Navy", "#1A1A3E"],
  ["dark navy", "Dark Navy", "#1A1A3E"],
  ["navy", "Navy", "#1A1A3E"],
  ["baby blue", "Baby Blue", "#A6C8E5"],
  ["off white", "Off White", "#F8F6F0"],
  ["forest green", "Forest Green", "#234B35"],
  ["dark green", "Forest Green", "#234B35"],
  ["olive green", "Olive Green", "#6B8E23"],
  ["light green", "Light Green", "#7FBF7F"],
  ["emerald", "Emerald", "#50C878"],
  ["sage", "Sage", "#9CAF88"],
  ["mint", "Mint", "#3EB489"],
  ["aqua", "Aqua", "#00FFFF"],
  ["hot pink", "Hot Pink", "#FF69B4"],
  ["pastel pink", "Pastel Pink", "#FFD1DC"],
  ["dusty pink", "Dusty Pink", "#DCAE96"],
  ["baby pink", "Baby Pink", "#F4C2C2"],
  ["rose gold", "Rose Gold", "#E0BFB8"],
  ["rose", "Rose", "#D66D6D"],
  ["blush", "Blush", "#F4C2C2"],
  ["dark red", "Dark Red", "#8B0000"],
  ["wine", "Wine", "#722F37"],
  ["burgundy", "Burgundy", "#6E1F2A"],
  ["maroon", "Maroon", "#6E1F2A"],
  ["terracotta", "Terracotta", "#E2725B"],
  ["rust", "Rust", "#B7410E"],
  ["coral", "Coral", "#FF7F50"],
  ["peach", "Peach", "#FFDAB9"],
  ["mustard", "Mustard", "#D4A017"],
  ["marigold", "Marigold", "#EAA221"],
  ["golden", "Gold", "#FFD700"],
  ["charcoal", "Charcoal", "#36454F"],
  ["denim", "Denim Blue", "#5B7C99"],
  ["lavender", "Lavender", "#B57EDC"],
  ["turquoise", "Turquoise", "#40E0D0"],
  ["fuchsia", "Magenta", "#D648D6"],
  ["magenta", "Magenta", "#D648D6"],
  ["silver", "Silver", "#C0C0C0"],
  ["camel", "Camel", "#C19A6B"],
  ["cream", "Cream", "#FFFDD0"],
  ["ivory", "Ivory", "#FFFFF0"],
  ["beige", "Beige", "#F5F5DC"],
  ["khaki", "Khaki", "#C3B091"],
  ["tan", "Tan", "#D2B48C"],
  ["brown", "Brown", "#8B6914"],
  ["espresso", "Espresso", "#3A2418"],
  ["taupe", "Taupe", "#B38B6D"],
  ["champagne", "Champagne", "#F7E7CE"],
  ["olive", "Olive Green", "#6B8E23"],
  ["mauve", "Mauve", "#8B668B"],
  ["plum", "Plum", "#8E4585"],
  ["lilac", "Lilac", "#C8A2C8"],
  ["indigo", "Indigo", "#4B0082"],
  ["violet", "Violet", "#7F00FF"],
  ["black", "Black", "#000000"],
  ["white", "White", "#FFFFFF"],
  ["grey", "Grey", "#808080"],
  ["gray", "Grey", "#808080"],
  ["green", "Green", "#2E8B57"],
  ["teal", "Teal", "#008080"],
  ["blue", "Blue", "#0047AB"],
  ["yellow", "Yellow", "#FFD700"],
  ["gold", "Gold", "#FFD700"],
  ["orange", "Orange", "#FFA500"],
  ["pink", "Light Pink", "#FFB6C1"],
  ["red", "Red", "#C0392B"],
  ["purple", "Purple", "#6A0DAD"],
  // extended fashion/fabric colours (reds & pinks)
  ["scarlet", "Scarlet", "#C21E26"],
  ["crimson", "Crimson", "#DC143C"],
  ["cardinal", "Cardinal", "#9B1B30"],
  ["ruby", "Ruby", "#9B111E"],
  ["cherry red", "Cherry Red", "#D2042D"],
  ["cherry", "Cherry", "#D2042D"],
  ["berry", "Berry", "#6E2639"],
  ["oxblood", "Oxblood", "#4A0E0E"],
  ["claret", "Claret", "#6E1F2A"],
  ["aubergine", "Aubergine", "#47294E"],
  ["mulberry", "Mulberry", "#7B2D43"],
  ["salmon", "Salmon", "#FA8072"],
  ["dusty rose", "Dusty Rose", "#C9A9A6"],
  // extended fashion/fabric colours (warm neutrals & earth tones)
  ["tangerine", "Tangerine", "#F28500"],
  ["apricot", "Apricot", "#FBCEB1"],
  ["canary yellow", "Canary Yellow", "#FFEF00"],
  ["lemon", "Lemon", "#FFF44F"],
  ["amber", "Amber", "#FFBF00"],
  ["saffron", "Saffron", "#FF9933"],
  ["ochre", "Ochre", "#CC7722"],
  ["ocher", "Ochre", "#CC7722"],
  ["chocolate", "Chocolate", "#5B3A29"],
  ["mocha", "Mocha", "#967969"],
  ["caramel", "Caramel", "#C68E17"],
  ["toffee", "Toffee", "#9E6B4B"],
  ["butterscotch", "Butterscotch", "#E1A56A"],
  ["honey", "Honey", "#E8B04B"],
  ["cinnamon", "Cinnamon", "#A0522D"],
  ["chestnut", "Chestnut", "#954535"],
  ["walnut", "Walnut", "#5C4033"],
  ["bronze", "Bronze", "#CD7F32"],
  ["copper", "Copper", "#B87333"],
  ["brass", "Brass", "#B5A642"],
  ["sand", "Sand", "#C2B280"],
  ["stone", "Stone", "#A89F91"],
  ["mushroom", "Mushroom", "#BDA493"],
  ["eggshell", "Eggshell", "#F0EAD6"],
  // extended fashion/fabric colours (cool neutrals)
  ["platinum", "Platinum", "#E5E4E2"],
  ["pewter", "Pewter", "#8E9291"],
  ["gunmetal", "Gunmetal", "#2A3439"],
  ["graphite", "Graphite", "#41424A"],
  ["slate", "Slate", "#708090"],
  ["ash grey", "Ash Grey", "#B2BEB5"],
  ["onyx", "Onyx", "#35383D"],
  ["jet black", "Jet Black", "#0A0A0A"],
  // extended fashion/fabric colours (blues & greens)
  ["midnight blue", "Midnight Blue", "#191970"],
  ["sapphire", "Sapphire", "#0F52BA"],
  ["azure", "Azure", "#007FFF"],
  ["cerulean", "Cerulean", "#007BA7"],
  ["cornflower blue", "Cornflower Blue", "#6495ED"],
  ["periwinkle", "Periwinkle", "#CCCCFF"],
  ["steel blue", "Steel Blue", "#4682B4"],
  ["steel grey", "Steel Grey", "#536878"],
  ["dusty blue", "Dusty Blue", "#8CA8C4"],
  ["peacock blue", "Peacock Blue", "#0E7C9C"],
  ["ice blue", "Ice Blue", "#D6F0FA"],
  ["seafoam", "Seafoam", "#93E9BE"],
  ["sea green", "Sea Green", "#2E8B57"],
  ["jade", "Jade", "#00A86B"],
  ["lime", "Lime", "#BFFF00"],
  ["chartreuse", "Chartreuse", "#7FFF00"],
  ["moss", "Moss", "#8A9A5B"],
  ["bottle green", "Bottle Green", "#006A4E"],
  ["hunter green", "Hunter Green", "#355E3B"],
  ["pistachio", "Pistachio", "#93C572"],
  ["sage green", "Sage", "#9CAF88"],
];

/** [phrase, canonical category] — one entry per recognised garment word. */
const GARMENTS: [string, string][] = [
  ["crop top", "Crop Top"],
  ["tank top", "Tank Top"],
  ["tee shirt", "T-Shirt"],
  ["polo shirt", "Polo"],
  ["t shirt", "T-Shirt"],
  ["tshirt", "T-Shirt"],
  ["tee", "T-Shirt"],
  ["shirt", "Shirt"],
  ["blouse", "Blouse"],
  ["top", "Top"],
  ["camisole", "Camisole"],
  ["polo", "Polo"],
  ["sweatshirt", "Sweatshirt"],
  ["sweater", "Sweater"],
  ["jumper", "Sweater"],
  ["hoodie", "Hoodie"],
  ["cardigan", "Cardigan"],
  ["tunic", "Tunic"],
  ["kurti", "Kurti"],
  ["kurta", "Kurta"],
  ["blazer", "Blazer"],
  ["jacket", "Jacket"],
  ["coat", "Coat"],
  ["parka", "Coat"],
  ["waistcoat", "Waistcoat"],
  ["vest", "Vest"],
  ["shrug", "Shrug"],
  ["jeans", "Jeans"],
  ["trousers", "Trousers"],
  ["pants", "Trousers"],
  ["chinos", "Trousers"],
  ["bottoms", "Bottoms"],
  ["bottom", "Bottoms"],
  ["skirt", "Skirt"],
  ["shorts", "Shorts"],
  ["leggings", "Leggings"],
  ["joggers", "Joggers"],
  ["palazzo", "Palazzo"],
  ["dress", "Dress"],
  ["gown", "Dress"],
  ["jumpsuit", "Jumpsuit"],
  ["romper", "Romper"],
  ["saree", "Saree"],
  ["sari", "Saree"],
  ["lehenga", "Lehenga"],
  ["salwar", "Salwar"],
  ["suit", "Suit"],
  ["sneakers", "Sneakers"],
  ["trainers", "Sneakers"],
  ["shoes", "Shoes"],
  ["boots", "Boots"],
  ["heels", "Heels"],
  ["sandals", "Sandals"],
  ["flats", "Flats"],
  ["loafers", "Loafers"],
  ["moccasins", "Moccasins"],
  ["scarf", "Scarf"],
  ["belt", "Belt"],
  ["handbag", "Bag"],
  ["backpack", "Bag"],
  ["tote", "Bag"],
  ["bag", "Bag"],
  ["watch", "Watch"],
  ["necklace", "Necklace"],
  ["earrings", "Earrings"],
  ["sunglasses", "Sunglasses"],
  ["hat", "Hat"],
  ["cap", "Cap"],
  ["beanie", "Hat"],
  ["tie", "Tie"],
  // extended Indian wardrobe
  ["dupatta", "Dupatta"],
  ["salwar kameez", "Salwar Kameez"],
  ["kameez", "Kameez"],
  ["churidar", "Churidar"],
  ["gharara", "Gharara"],
  ["sharara", "Sharara"],
  ["sherwani", "Sherwani"],
  ["choli", "Choli"],
  ["shawl", "Shawl"],
  ["stole", "Scarf"],
  ["muffler", "Scarf"],
  ["scarves", "Scarf"],
  // extended global wardrobe
  ["anorak", "Anorak"],
  ["poncho", "Poncho"],
  ["cape", "Cape"],
  ["kimono", "Kimono"],
  ["kaftan", "Kaftan"],
  ["caftan", "Kaftan"],
  ["dungarees", "Dungarees"],
  ["overalls", "Overalls"],
  ["swimsuit", "Swimsuit"],
  ["bikini", "Bikini"],
  ["pyjamas", "Pyjamas"],
  ["pajamas", "Pyjamas"],
  ["nighties", "Nightdress"],
  ["nighty", "Nightdress"],
  ["nightdress", "Nightdress"],
  ["sweatpants", "Joggers"],
  ["track pants", "Joggers"],
  ["trackpants", "Joggers"],
  ["cargos", "Trousers"],
  ["palazzos", "Palazzo"],
  ["palazzo pants", "Palazzo"],
  ["tights", "Leggings"],
  ["overcoat", "Coat"],
  ["raincoat", "Coat"],
  ["windbreaker", "Jacket"],
  ["puffer jacket", "Jacket"],
  ["puffer", "Jacket"],
  ["bomber", "Jacket"],
  ["stilettos", "Heels"],
  ["oxfords", "Shoes"],
  ["espadrilles", "Shoes"],
  ["slippers", "Slippers"],
  ["footwear", "Shoes"],
  // singular forms for plural-only entries
  ["sneaker", "Sneakers"],
  ["trainer", "Sneakers"],
  ["shoe", "Shoes"],
  ["boot", "Boots"],
  ["heel", "Heels"],
  ["sandal", "Sandals"],
  ["loafer", "Loafers"],
  ["moccasin", "Moccasins"],
  ["trouser", "Trousers"],
  ["earring", "Earrings"],
  // more accessories
  ["gloves", "Gloves"],
  ["mittens", "Gloves"],
  ["socks", "Socks"],
  ["bracelet", "Bracelet"],
  ["bangle", "Bangle"],
  ["anklet", "Anklet"],
  ["necktie", "Tie"],
  ["bow tie", "Tie"],
  ["bowtie", "Tie"],
  ["ring", "Ring"],
];

/** [phrase, canonical pattern] (values match the UI pattern vocabulary). */
const PATTERNS: [string, string][] = [
  ["tie dye", "Printed"],
  ["polka dots", "Printed"],
  ["polka dot", "Printed"],
  ["polka", "Printed"],
  ["striped", "Striped"],
  ["stripes", "Striped"],
  ["stripe", "Striped"],
  ["checked", "Checked"],
  ["checks", "Checked"],
  ["check", "Checked"],
  ["plaid", "Checked"],
  ["floral", "Floral"],
  ["florals", "Floral"],
  ["geometric", "Geometric"],
  ["printed", "Printed"],
  ["graphic", "Printed"],
  ["print", "Printed"],
  ["embroidered", "Embroidered"],
  ["embroidery", "Embroidered"],
  ["solid", "Solid"],
  ["plain", "Solid"],
];

/** [match phrase, canonical fit label] — labels use hyphenated display form. */
const FITS: [string, string][] = [
  ["oversized", "oversized"],
  ["slim fit", "slim-fit"],
  ["slim", "slim"],
  ["skinny", "skinny"],
  ["straight leg", "straight-leg"],
  ["straight", "straight"],
  ["wide leg", "wide-leg"],
  ["relaxed", "relaxed"],
  ["tailored", "tailored"],
  ["fitted", "fitted"],
  ["baggy", "baggy"],
  ["loose", "loose"],
  ["cropped", "cropped"],
  ["regular", "regular"],
  ["flowy", "flowy"],
  ["mini", "mini"],
  ["midi", "midi"],
  ["maxi", "maxi"],
  ["ankle length", "ankle-length"],
  ["full length", "full-length"],
];

/** Chunks that talk about colours in general, not a specific garment. */
const META_RE =
  /\b(colours?|colors?|shades?|tones?|palette|wardrobe|closet|clothes|clothing|outfits?)\b/;

/** Leading filler words that carry no garment meaning. */
const FILLER_RE =
  /^(?:and|with|plus|over|also|then|i\s+(?:have|own|got|bought|brought|wear|am\s+wearing)|i'm\s+wearing|my|our|a|an|the|some|one|pair\s+of)\s+/;

/* ------------------------------------------------------------------ */
/*  Matching primitives                                                 */
/* ------------------------------------------------------------------ */

interface Span {
  start: number;
  end: number;
}

interface CompiledMatch<T> {
  phrase: string;
  value: T;
  re: RegExp;
}

interface Match<T> {
  span: Span;
  value: T;
  phrase: string;
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Compile a dictionary to word-boundary matchers, longest phrase first.
 * Spaces inside a phrase also accept hyphens ("wide leg" matches "wide-leg").
 */
function compile<T>(dict: [string, T][], opts?: { plural?: boolean }): CompiledMatch<T>[] {
  return dict
    .map((entry) => {
      const [phrase, value] = entry as [string, T];
      const words = phrase.split(" ").map(escapeRe);
      // Garment phrases pluralise on their last word, so one entry covers
      // "top"/"tops", "t-shirt"/"t-shirts", "watch"/"watches" and
      // "dress"/"dresses" without bloating the table.
      if (opts?.plural) {
        const last = words.pop() as string;
        words.push(`${last}(?:s|es)?`);
      }
      const body = words.join("[\\s-]+");
      return { phrase, value, re: new RegExp(`\\b${body}\\b`, "g") };
    })
    .sort((a, b) => b.phrase.length - a.phrase.length);
}

/**
 * All non-overlapping matches, longest phrase winning each overlap,
 * returned in position order (deterministic).
 */
function matchEntries<T>(text: string, entries: CompiledMatch<T>[]): Match<T>[] {
  const out: Match<T>[] = [];
  for (const { phrase, value, re } of entries) {
    re.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      const span = { start: m.index, end: m.index + m[0].length };
      if (m[0].length === 0) {
        re.lastIndex++;
        continue;
      }
      if (!out.some((o) => span.start < o.span.end && o.span.start < span.end)) {
        out.push({ span, value, phrase });
      }
    }
  }
  return out.sort((a, b) => a.span.start - b.span.start);
}

const COLOUR_ENTRIES = compile(COLOURS.map(([phrase, name, hex]) => [phrase, { name, hex }] as [string, { name: string; hex: string }]));
const GARMENT_ENTRIES = compile(GARMENTS, { plural: true });
const PATTERN_ENTRIES = compile(PATTERNS);
const FIT_ENTRIES = compile(FITS);

/* ------------------------------------------------------------------ */
/*  Stages                                                              */
/* ------------------------------------------------------------------ */

/** Lowercase + unify quotes/dashes without losing semantic content. */
function normaliseInput(input: string): string {
  return input
    .toLowerCase()
    .replace(/[\u2018\u2019\u02BC]/g, "'")
    .replace(/[\u2013\u2014]/g, "-")
    .replace(/\boffwhite\b/g, "off white")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Segment into garment phrases: commas, semicolons, periods, ampersands and
 * the connectors "and" / "with" / "plus" / "over" all separate garments
 * ("navy blazer over a white shirt with charcoal trousers" -> 3 phrases).
 */
function segment(text: string): string[] {
  return text
    .split(/\s*(?:[.,;&]|\b(?:and|with|plus|over)\b)\s*/)
    .map((c) => c.trim())
    .filter(Boolean);
}

function stripFillers(chunk: string): string {
  let s = chunk.trim();
  for (let i = 0; i < 5; i++) {
    const next = s.replace(FILLER_RE, "").trim();
    if (next === s) break;
    s = next;
  }
  return s;
}

/** Longest colour span fully inside [from, to), ties broken by position. */
function colourIn(
  colours: Match<{ name: string; hex: string }>[],
  from: number,
  to: number,
): Match<{ name: string; hex: string }> | null {
  const inScope = colours.filter((c) => c.span.start >= from && c.span.end <= to);
  if (!inScope.length) return null;
  inScope.sort(
    (a, b) =>
      b.span.end - b.span.start - (a.span.end - a.span.start) ||
      a.span.start - b.span.start,
  );
  return inScope[0];
}

/* ------------------------------------------------------------------ */
/*  Parser                                                              */
/* ------------------------------------------------------------------ */

export function parseDescriptionLocally(description: string): {
  garments: GarmentInput[];
  assumptions: string[];
} {
  const assumptions: string[] = [];
  const candidates: GarmentInput[] = [];
  const text = normaliseInput(description);
  if (!text) return { garments: [], assumptions };
  let nextId = 0;

  for (const rawChunk of segment(text)) {
    const chunk = stripFillers(rawChunk);
    if (chunk.length < 2) continue;

    const colours = matchEntries(chunk, COLOUR_ENTRIES).filter(
      // Hyphen compounds the dictionary does not model ("blue-green",
      // "emerald-green") must stay UNKNOWN — never silently matched to
      // "blue" or "green". Whole-phrase dictionary matches ("off-white",
      // "rose-gold") are unaffected: they cover the hyphen internally.
      (c) =>
        chunk[c.span.end] !== "-" &&
        !(c.span.start > 0 && chunk[c.span.start - 1] === "-"),
    );
    const patterns = matchEntries(chunk, PATTERN_ENTRIES);
    const fits = matchEntries(chunk, FIT_ENTRIES);
    const blockers = [...colours, ...patterns];
    const categories = matchEntries(chunk, GARMENT_ENTRIES).filter(
      (c) =>
        !blockers.some((b) => c.span.start < b.span.end && b.span.start < c.span.end),
    );

    if (!categories.length) {
      // Modifier phrase ("with gold embroidery") -> decorate the previous
      // garment; never fall through to a phantom new garment for these.
      const prev = candidates[candidates.length - 1];
      const pattern = patterns[0]?.value ?? null;
      const fit = fits[0]?.value ?? null;
      const modified =
        (prev && pattern && (!prev.pattern || prev.pattern === "Solid")
          ? ((prev.pattern = pattern), true)
          : false) ||
        (prev && fit && !prev.fit ? ((prev.fit = fit), true) : false);
      if (modified || ((pattern || fit) && prev)) continue;

      // Colour-only phrase ("and teal") -> honest editable placeholder.
      const colour = colours[0];
      if (colour && !META_RE.test(chunk)) {
        assumptions.push(
          `could not tell the garment type for the ${colour.value.name.toLowerCase()} item — set it below`,
        );
        candidates.push({
          id: `desc-${++nextId}`,
          category: "Garment",
          colourName: colour.value.name,
          hex: colour.value.hex,
          pattern: null,
          fit: null,
          owned: true,
        });
      }
      continue;
    }

    // One garment per category word; each colour/fit/pattern is attributed to
    // the nearest category word via its scope (from the previous category's
    // end to this category's start), so attributes never leak across garments.
    for (let i = 0; i < categories.length; i++) {
      const cat = categories[i];
      const scopeStart = i > 0 ? categories[i - 1].span.end : 0;
      const scopeEnd = cat.span.start;

      let colour = colourIn(colours, scopeStart, scopeEnd);
      if (!colour && i === 0) colour = colourIn(colours, 0, chunk.length);
      if (!colour && categories.length === 1 && colours.length === 1) {
        colour = colours[0];
      }

      const inScope = <T>(matches: Match<T>[]) =>
        matches.find((m) => m.span.start >= scopeStart && m.span.end <= scopeEnd)?.value ?? null;
      const pattern = inScope(patterns);
      const fit = inScope(fits);

      if (colour) {
        candidates.push({
          id: `desc-${++nextId}`,
          category: cat.value,
          colourName: colour.value.name,
          hex: colour.value.hex,
          pattern,
          fit,
          owned: true,
        });
      } else {
        assumptions.push(
          `could not tell the colour of the ${cat.value.toLowerCase()} — add it below`,
        );
        candidates.push({
          id: `desc-${++nextId}`,
          category: cat.value,
          colourName: "Unknown",
          hex: "",
          pattern,
          fit,
          owned: true,
        });
      }
    }
  }

  // Dedupe identical category+colour pairs (multi-rule double matches) while
  // keeping distinct garments — then cap, preserving deterministic order.
  const seen = new Set<string>();
  const garments: GarmentInput[] = [];
  for (const g of candidates) {
    const key = `${g.category.toLowerCase()}|${g.hex.toUpperCase()}`;
    if (g.hex && seen.has(key)) continue;
    seen.add(key);
    garments.push(g);
    if (garments.length >= MAX_PARSED_GARMENTS) break;
  }
  return { garments, assumptions };
}
