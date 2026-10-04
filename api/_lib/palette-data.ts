/**
 * Seasonal colour palette definitions for FaceMaxx Outfit Studio.
 * Self-contained in the `api/` directory so serverless functions do not
 * import runtime modules across the boundary from `src/`.
 */

export interface ColourSwatch {
  name: string;
  hex: string;
}

export interface SeasonPalette {
  best: ColourSwatch[];
  neutrals: ColourSwatch[];
  caution: ColourSwatch[];
}

export const SEASON_DESCRIPTIONS: Record<
  string,
  { family: string; description: string; palette: SeasonPalette }
> = {
  "Light Spring": {
    family: "Spring",
    description: "Light, warm and fresh colors complement your natural coloring.",
    palette: {
      best: [
        { name: "Peach", hex: "#FFDAB9" },
        { name: "Warm Coral", hex: "#F08080" },
        { name: "Light Gold", hex: "#FFD700" },
        { name: "Aquamarine", hex: "#7FFFD4" },
        { name: "Ivory", hex: "#FFFFF0" },
        { name: "Warm Taupe", hex: "#C4A882" },
      ],
      neutrals: [
        { name: "Cream", hex: "#FFFDD0" },
        { name: "Warm Beige", hex: "#D4B896" },
        { name: "Soft Brown", hex: "#A0846B" },
      ],
      caution: [
        { name: "Black", hex: "#000000" },
        { name: "Dark Navy", hex: "#1A1A3E" },
        { name: "Mauve", hex: "#8B668B" },
      ],
    },
  },
  "Warm Spring": {
    family: "Spring",
    description: "Warm, golden and vivid colors naturally harmonize with your coloring.",
    palette: {
      best: [
        { name: "Tomato Red", hex: "#FF6347" },
        { name: "Marigold", hex: "#EAA221" },
        { name: "Warm Green", hex: "#6B8E23" },
        { name: "Peach Gold", hex: "#FFC78D" },
        { name: "Warm Pink", hex: "#E8838B" },
        { name: "Warm Camel", hex: "#C19A6B" },
      ],
      neutrals: [
        { name: "Camel", hex: "#C19A6B" },
        { name: "Warm Brown", hex: "#8B6914" },
        { name: "Ivory", hex: "#FFFFF0" },
      ],
      caution: [
        { name: "Black", hex: "#000000" },
        { name: "Cool Pink", hex: "#FFB6C1" },
        { name: "Silver", hex: "#C0C0C0" },
      ],
    },
  },
  "Bright Spring": {
    family: "Spring",
    description: "Clear, bright and warm colors bring out the best in your natural coloring.",
    palette: {
      best: [
        { name: "Bright Coral", hex: "#FF4040" },
        { name: "Kelly Green", hex: "#4CBB17" },
        { name: "Cobalt Blue", hex: "#0047AB" },
        { name: "Cannon Yellow", hex: "#FFEF00" },
        { name: "Bright Turquoise", hex: "#08E8DE" },
        { name: "Warm Magenta", hex: "#D648D6" },
      ],
      neutrals: [
        { name: "Warm Navy", hex: "#2C3E50" },
        { name: "Tan", hex: "#D2B48C" },
        { name: "Bright White", hex: "#FFFFFF" },
      ],
      caution: [
        { name: "Muted Gray", hex: "#808080" },
        { name: "Dusty Rose", hex: "#DCAE96" },
        { name: "Brown", hex: "#6B4226" },
      ],
    },
  },
  "Light Summer": {
    family: "Summer",
    description: "Soft, light and cool colors create a gentle harmony with your natural tones.",
    palette: {
      best: [
        { name: "Powder Blue", hex: "#B0E0E6" },
        { name: "Soft Pink", hex: "#FFB7C5" },
        { name: "Lavender", hex: "#E6E6FA" },
        { name: "Soft Sage", hex: "#9DC183" },
        { name: "Rose Beige", hex: "#E8C4B8" },
        { name: "Light Mauve", hex: "#D8BFD8" },
      ],
      neutrals: [
        { name: "Soft White", hex: "#F8F8F8" },
        { name: "Dove Gray", hex: "#B0B0B0" },
        { name: "Taupe", hex: "#8B7D6B" },
      ],
      caution: [
        { name: "Black", hex: "#000000" },
        { name: "Orange", hex: "#FF8C00" },
        { name: "Bright Yellow", hex: "#FFFF00" },
      ],
    },
  },
  "Cool Summer": {
    family: "Summer",
    description: "Cool, muted and blue-based colors complement your natural cool undertone.",
    palette: {
      best: [
        { name: "Rose", hex: "#FF007F" },
        { name: "Periwinkle", hex: "#CCCCFF" },
        { name: "Soft Teal", hex: "#5F9EA0" },
        { name: "Blue Gray", hex: "#6699CC" },
        { name: "Mauve", hex: "#E0B0FF" },
        { name: "Soft Navy", hex: "#4E6172" },
      ],
      neutrals: [
        { name: "Cool Gray", hex: "#A9A9A9" },
        { name: "Steel", hex: "#71797E" },
        { name: "Off White", hex: "#F5F5F5" },
      ],
      caution: [
        { name: "Orange", hex: "#FF8C00" },
        { name: "Warm Brown", hex: "#8B6914" },
        { name: "Golden Yellow", hex: "#FFD700" },
      ],
    },
  },
  "Soft Summer": {
    family: "Summer",
    description: "Muted, soft and cool-toned colors blend beautifully with your natural coloring.",
    palette: {
      best: [
        { name: "Dusty Rose", hex: "#DCAE96" },
        { name: "Sage Green", hex: "#9CAF88" },
        { name: "Soft Slate", hex: "#708090" },
        { name: "Muted Plum", hex: "#8B668B" },
        { name: "Stone", hex: "#928E85" },
        { name: "Soft Teal", hex: "#6699A6" },
      ],
      neutrals: [
        { name: "Warm Gray", hex: "#9E9E9E" },
        { name: "Taupe", hex: "#8B7D6B" },
        { name: "Soft Cream", hex: "#F5F0E8" },
      ],
      caution: [
        { name: "Bright Orange", hex: "#FF4500" },
        { name: "Hot Pink", hex: "#FF69B4" },
        { name: "Black", hex: "#000000" },
      ],
    },
  },
  "Soft Autumn": {
    family: "Autumn",
    description: "Muted, warm and earthy tones create a natural harmony with your coloring.",
    palette: {
      best: [
        { name: "Terracotta", hex: "#CC5533" },
        { name: "Olive", hex: "#808000" },
        { name: "Warm Taupe", hex: "#B5937A" },
        { name: "Dusty Teal", hex: "#5F8A8B" },
        { name: "Warm Plum", hex: "#8B5E83" },
        { name: "Muted Gold", hex: "#C4A35A" },
      ],
      neutrals: [
        { name: "Khaki", hex: "#C3B091" },
        { name: "Warm Brown", hex: "#7B5B3A" },
        { name: "Cream", hex: "#FFFDD0" },
      ],
      caution: [
        { name: "Black", hex: "#000000" },
        { name: "Hot Pink", hex: "#FF69B4" },
        { name: "Bright Blue", hex: "#0000FF" },
      ],
    },
  },
  "Warm Autumn": {
    family: "Autumn",
    description: "Rich, warm and golden earth tones bring out the warmth in your natural coloring.",
    palette: {
      best: [
        { name: "Rust", hex: "#B7410E" },
        { name: "Mustard", hex: "#FFDB58" },
        { name: "Olive Green", hex: "#556B2F" },
        { name: "Warm Brown", hex: "#8B4513" },
        { name: "Pumpkin", hex: "#FF7518" },
        { name: "Warm Bronze", hex: "#CD7F32" },
      ],
      neutrals: [
        { name: "Camel", hex: "#C19A6B" },
        { name: "Espresso", hex: "#3C1414" },
        { name: "Warm Beige", hex: "#D4B896" },
      ],
      caution: [
        { name: "Black", hex: "#000000" },
        { name: "Icy Blue", hex: "#A5F2F3" },
        { name: "Pink", hex: "#FFC0CB" },
      ],
    },
  },
  "Deep Autumn": {
    family: "Autumn",
    description: "Deep, rich and warm colors complement your naturally deep and warm coloring.",
    palette: {
      best: [
        { name: "Burgundy", hex: "#6E1F2A" },
        { name: "Forest", hex: "#234B35" },
        { name: "Espresso", hex: "#3A2418" },
        { name: "Warm Plum", hex: "#64384F" },
        { name: "Deep Teal", hex: "#174F52" },
        { name: "Spice", hex: "#7A3B2E" },
      ],
      neutrals: [
        { name: "Espresso", hex: "#3A2418" },
        { name: "Dark Olive", hex: "#3B4A2B" },
        { name: "Cream", hex: "#F2E5CF" },
      ],
      caution: [
        { name: "Pastel Pink", hex: "#FFD1DC" },
        { name: "Ice Blue", hex: "#D6ECEF" },
        { name: "Bright Yellow", hex: "#FFFF00" },
      ],
    },
  },
  "Deep Winter": {
    family: "Winter",
    description: "Deep, cool and clear colors create a striking contrast with your natural coloring.",
    palette: {
      best: [
        { name: "True Red", hex: "#CC0000" },
        { name: "Emerald", hex: "#046307" },
        { name: "Navy", hex: "#000080" },
        { name: "Deep Plum", hex: "#4E2A5A" },
        { name: "Charcoal", hex: "#36454F" },
        { name: "Fuchsia", hex: "#FF00FF" },
      ],
      neutrals: [
        { name: "Black", hex: "#000000" },
        { name: "Dark Navy", hex: "#0C0C3A" },
        { name: "Bright White", hex: "#FFFFFF" },
      ],
      caution: [
        { name: "Peach", hex: "#FFDAB9" },
        { name: "Light Gold", hex: "#FFD700" },
        { name: "Camel", hex: "#C19A6B" },
      ],
    },
  },
  "Cool Winter": {
    family: "Winter",
    description: "Cool, clear and blue-based colors create a crisp harmony with your coloring.",
    palette: {
      best: [
        { name: "True Red", hex: "#CC0000" },
        { name: "Royal Blue", hex: "#4169E1" },
        { name: "Magenta", hex: "#FF0090" },
        { name: "Pine Green", hex: "#01796F" },
        { name: "Icy Pink", hex: "#FFE0E8" },
        { name: "Black Cherry", hex: "#4A0404" },
      ],
      neutrals: [
        { name: "Black", hex: "#000000" },
        { name: "Cool Gray", hex: "#808080" },
        { name: "White", hex: "#FFFFFF" },
      ],
      caution: [
        { name: "Orange", hex: "#FF8C00" },
        { name: "Warm Brown", hex: "#8B6914" },
        { name: "Peach", hex: "#FFDAB9" },
      ],
    },
  },
  "Bright Winter": {
    family: "Winter",
    description: "Clear, bright and high-contrast colors make your natural coloring come alive.",
    palette: {
      best: [
        { name: "True Red", hex: "#CC0000" },
        { name: "Bright Teal", hex: "#00CED1" },
        { name: "Electric Blue", hex: "#7DF9FF" },
        { name: "Hot Pink", hex: "#FF69B4" },
        { name: "Emerald", hex: "#046307" },
        { name: "Bright White", hex: "#FFFFFF" },
      ],
      neutrals: [
        { name: "Black", hex: "#000000" },
        { name: "Bright White", hex: "#FFFFFF" },
        { name: "Charcoal", hex: "#36454F" },
      ],
      caution: [
        { name: "Camel", hex: "#C19A6B" },
        { name: "Muted Gray", hex: "#808080" },
        { name: "Peach", hex: "#FFDAB9" },
      ],
    },
  },
};
