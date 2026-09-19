import type { ColourAnalysisResponse } from "../types/api";

const BASE_URL = "";

export async function analyzeColours(base64Image: string): Promise<ColourAnalysisResponse> {
  const response = await fetch(`${BASE_URL}/api/color-analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: base64Image }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || "Analysis failed. Please try again.");
  }

  const data: ColourAnalysisResponse & { fallback?: boolean } = await response.json();

  // Handle stub/fallback responses (e.g. Vercel deployment without Python backend)
  if (data.fallback || !data.ok) {
    throw new Error(data.error || "Colour analysis is not available in this deployment. Please run locally with `npm run dev`.");
  }

  return data;
}

/**
 * Downscale an image to a JPEG data URL whose longest edge is `maxEdge` px.
 * Keeps uploads well under serverless request limits (~4.5 MB) and keeps the
 * colour analysis fast; 1024px is plenty for face detection + LAB sampling.
 */
export function downscaleImage(
  dataUrl: string,
  maxEdge = 1024,
  quality = 0.9,
): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, maxEdge / Math.max(img.width, img.height));
      if (scale >= 1) {
        resolve(dataUrl); // already small enough — send untouched
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        resolve(dataUrl);
        return;
      }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/jpeg", quality));
    };
    img.onerror = () => resolve(dataUrl); // on failure, fall back to original
    img.src = dataUrl;
  });
}

export async function readFileAsBase64(file: File): Promise<string> {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Failed to read image file."));
    reader.readAsDataURL(file);
  });
  return downscaleImage(dataUrl);
}

export function isValidImageType(file: File): boolean {
  return ["image/jpeg", "image/jpg", "image/png", "image/webp"].includes(file.type);
}
