import type { VercelRequest, VercelResponse } from "@vercel/node";

export default function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  return res.status(200).json({
    categories: [
      {
        id: "face_shape",
        name: "Face Shape Styling Rules",
        tips: [
          "Oval Shape: Balanced proportions. Suits almost all hairstyles.",
          "Square Shape: Strong jawline. Soften with textured quiffs, classic side parts.",
          "Round Shape: Width equals length. Add height with a pompadour or high volume cut.",
          "Heart Shape: Wide forehead, pointed chin. Choose hairstyles with volume at the sides.",
          "Diamond Shape: Strong cheekbones. Side swept bangs or textured crops look best.",
        ],
      },
      {
        id: "photography_basics",
        name: "Professional Portrait Composition",
        tips: [
          "Lighting: Position your light source 45 degrees to the side.",
          "Lens Height: Keep the camera lens at or slightly above eye level.",
          "Background: A clean, neutral background highlights your silhouette.",
          "Expression: Relax the jaw for a clean submental angle.",
        ],
      },
    ],
  });
}
