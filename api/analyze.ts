import type { VercelRequest, VercelResponse } from "@vercel/node";

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { image } = req.body;

  if (!image) {
    return res.status(400).json({ error: "Image data (base64) is required." });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({
      error: "Gemini API key not configured. Please set GEMINI_API_KEY in your environment.",
    });
  }

  try {
    // Dynamic import to avoid bundling issues
    const { GoogleGenAI } = await import("@google/genai");

    const ai = new GoogleGenAI({
      apiKey,
      httpOptions: { headers: { "User-Agent": "aistudio-build" } },
    });

    // Isolate base64 data
    let base64Data = image;
    let mimeType = "image/jpeg";
    if (image.includes(";base64,")) {
      const parts = image.split(";base64,");
      const match = parts[0].match(/data:(.*?)$/);
      if (match) mimeType = match[1];
      base64Data = parts[1];
    }

    const imagePart = { inlineData: { mimeType, data: base64Data } };

    const systemInstruction = `You are the FaceMax AI analysis expert. Analyze this selfie and return a JSON report with scores, face analysis, photo quality, style recommendations, hairstyles, eyewear, skincare, and landmarks. Return ONLY valid JSON.`;

    const promptText = `Analyze this selfie and return a comprehensive JSON report with: scores (overallScore, photographyScore, groomingScore, styleScore), faceAnalysis (symmetryScore, symmetryDetails, jawlineType, eyeSpacingRatio, poseAssessment, expressionFeedback), photoQuality (lightingScore, lightingType, sharpnessScore), styleRecommendations, hairstyles, eyewear, skincare, and landmarks (leftEye, rightEye, noseBridge, noseTip, mouth, jawline, leftEyebrow, rightEyebrow, faceContour). All coordinates should be 0-100 relative to image dimensions. Return ONLY valid JSON.`;

    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: [imagePart, { text: promptText }],
      config: { systemInstruction, responseMimeType: "application/json" },
    });

    const textOutput = response.text || "{}";

    let parsedReport;
    try {
      let cleanText = textOutput.trim();
      if (cleanText.startsWith("```json")) cleanText = cleanText.substring(7);
      if (cleanText.endsWith("```")) cleanText = cleanText.substring(0, cleanText.length - 3);
      parsedReport = JSON.parse(cleanText.trim());
    } catch {
      return res.status(500).json({
        error: "Failed to parse analysis results. Please try a clearer photo.",
      });
    }

    // Add session ID
    const sessionId = "session_" + Math.random().toString(36).substring(2, 11);
    parsedReport.id = sessionId;
    parsedReport.timestamp = new Date().toISOString();

    return res.status(200).json(parsedReport);
  } catch (error: any) {
    console.error("Gemini analysis error:", error);
    return res.status(500).json({
      error: "Analysis failed. Please try again.",
      details: error?.message || String(error),
    });
  }
}
