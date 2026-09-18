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

  const data: ColourAnalysisResponse = await response.json();

  if (!data.ok) {
    throw new Error(data.error || "Analysis could not complete.");
  }

  return data;
}

export function readFileAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Failed to read image file."));
    reader.readAsDataURL(file);
  });
}

export function isValidImageType(file: File): boolean {
  return ["image/jpeg", "image/jpg", "image/png", "image/webp"].includes(file.type);
}
