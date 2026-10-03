import express from "express";
import path from "path";
import fs from "fs";
import os from "os";
import { execFile } from "child_process";
import dotenv from "dotenv";
import { GoogleGenAI, Type } from "@google/genai";
import { runOutfitAnalysis, runDescribeAnalysis, runWardrobeAnalysis, compareOutfits } from "./api/outfit-core";
import { createServer as createViteServer } from "vite";

// Load environment variables
dotenv.config();

const app = express();
const PORT = 3000;

// Set up body parsers with limits for handling large base64 uploads
app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ limit: "20mb", extended: true }));

// Initialize Gemini SDK lazily to prevent crashing if the key is missing on startup
let aiClient: GoogleGenAI | null = null;

function getAiClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable is required");
    }
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// Global sessions cache (in-memory, secure, self-cleaning)
const sessionsCache = new Map<string, { timestamp: number; data: any }>();

// Self-cleaning sessions interval (every hour, delete items older than 3 hours)
setInterval(() => {
  const now = Date.now();
  for (const [key, val] of sessionsCache.entries()) {
    if (now - val.timestamp > 3 * 60 * 60 * 1000) {
      sessionsCache.delete(key);
    }
  }
}, 60 * 60 * 1000);

// API Endpoints

/**
 * GET /api/health
 */
app.get("/api/health", (req, res) => {
  res.json({ status: "healthy", timestamp: new Date().toISOString() });
});

/**
 * GET /api/recommendations
 * General grooming and skincare intelligence database
 */
app.get("/api/recommendations", (req, res) => {
  res.json({
    categories: [
      {
        id: "face_shape",
        name: "Face Shape Styling Rules",
        tips: [
          "Oval Shape: Balanced proportions. Suits almost all hairstyles. Avoid heavy fringes that shorten the face.",
          "Square Shape: Strong jawline. Soften with textured quiffs, classic side parts, or round frames.",
          "Round Shape: Width equals length. Add height with a pompadour or high volume cut, and wear angular/geometric glasses.",
          "Heart Shape: Wide forehead, pointed chin. Choose hairstyles with volume at the sides or chin-length crops. Oval or round eyewear.",
          "Diamond Shape: Strong cheekbones. Side swept bangs or textured crops look best. Classic clubmaster or aviator eyewear balances width."
        ]
      },
      {
        id: "photography_basics",
        name: "Professional Portrait Composition",
        tips: [
          "Lighting: Position your light source 45 degrees to the side and slightly above eye level for dramatic but natural shadows.",
          "Lens Height: Keep the camera lens at or slightly above eye level to prevent perspective distortion.",
          "Background: A clean, neutral, or softly blurred background highlights your silhouette and projects professional authority.",
          "Expression: Relax the jaw, press your tongue to the roof of your mouth (mewing posture) for a clean submental angle, and smize."
        ]
      }
    ]
  });
});

/**
 * POST /api/simulate
 * Simulates a presentation modification (e.g. style overlays, lighting adjustment)
 */
app.post("/api/simulate", (req, res) => {
  const { sessionId, hairstyleId, eyewearId, lightingPreset } = req.body;

  if (!sessionId) {
    return res.status(400).json({ error: "sessionId is required for style simulation" });
  }

  const session = sessionsCache.get(sessionId);
  if (!session) {
    return res.status(404).json({ error: "Session data not found. Please re-upload or analyze your photo." });
  }

  // Simulate an updated presentation report based on the selected change
  const currentReport = session.data;
  const updatedScores = { ...currentReport.scores };

  let feedbackMessage = "";

  if (hairstyleId) {
    updatedScores.groomingScore = Math.min(100, updatedScores.groomingScore + 4);
    feedbackMessage += `Simulating hairstyle: '${hairstyleId}'. Tailored shape alignment increases facial silhouette symmetry. `;
  }

  if (eyewearId) {
    updatedScores.styleScore = Math.min(100, updatedScores.styleScore + 3);
    feedbackMessage += `Simulating eyewear frame: '${eyewearId}'. Structured lens borders define eye spacing and cheekbone ratios. `;
  }

  if (lightingPreset) {
    updatedScores.photographyScore = Math.min(100, updatedScores.photographyScore + 5);
    feedbackMessage += `Simulating lighting environment: '${lightingPreset}'. Professional key illumination eliminates flat submental shadows and enhances expression contrast. `;
  }

  // Recalculate overall
  updatedScores.overallScore = Math.round(
    (updatedScores.photographyScore + updatedScores.groomingScore + updatedScores.styleScore) / 3
  );

  res.json({
    success: true,
    feedback: feedbackMessage,
    scores: updatedScores,
    simulationApplied: { hairstyleId, eyewearId, lightingPreset }
  });
});

/**
 * POST /api/analyze
 * Accepts base64 selfie, runs Gemini multimodal analysis, returns a full presentation report
 */
app.post("/api/analyze", async (req, res) => {
  try {
    const { image, optInStoreSession } = req.body;

    if (!image) {
      return res.status(400).json({ error: "Image data (base64) is required." });
    }

    // Isolate base64 data from potential data URL prefixes
    let base64Data = image;
    let mimeType = "image/jpeg";

    if (image.includes(";base64,")) {
      const parts = image.split(";base64,");
      const match = parts[0].match(/data:(.*?)$/);
      if (match) {
        mimeType = match[1];
      }
      base64Data = parts[1];
    }

    const ai = getAiClient();

    const imagePart = {
      inlineData: {
        mimeType,
        data: base64Data
      }
    };

    const systemInstruction = `You are the lead computer vision and style intelligence expert in the FaceMax AI Luxury Glow-Up Laboratory.
Analyze the user's selfie.
Your feedback MUST be entirely constructive, supportive, and focused on visual presentation, photography quality, styling choices, hairstyle geometry, and skincare routines.
DO NOT output any attractiveness rankings, harsh judgments, or negative labels. Frame everything in terms of optimization and executive style.

You must return a raw JSON response that matches the exact structural schema requested.`;

    const promptText = `Analyze this selfie and generate a comprehensive presentation optimization report.
Detect facial landmarks (scaled 0-100 relative to the image coordinate box where x:0 is left edge, x:100 is right, y:0 is top, y:100 is bottom).
Generate the following JSON structure exactly:

{
  "scores": {
    "overallScore": number (70-98, represent executive presentation level),
    "photographyScore": number (65-100, composition/lighting/exposure),
    "groomingScore": number (65-100, hair neatness, skin balance),
    "styleScore": number (65-100, eyewear, colors, clothing harmony)
  },
  "faceAnalysis": {
    "symmetryScore": number (70-98),
    "symmetryDetails": "detailed descriptive string about horizontal/vertical alignment & jaw balance",
    "symmetryHeatmapPoints": [
      {"x": number, "y": number, "label": "string label of point"}
    ],
    "jawlineHarmony": "constructive analysis of jaw definition and grooming border rules",
    "jawlineType": "string (e.g. Angular, Soft Square, Curved, Diamond, Heart)",
    "eyeSpacingRatio": "e.g. 1:1 ratio",
    "eyeSpacingStatus": "string ('ideal' | 'wide' | 'close')",
    "poseAssessment": "constructive coaching on head tilt, chin angle, or gaze direction",
    "expressionFeedback": "analysis of smile symmetry, lip compression, or forehead micro-tensions"
  },
  "photoQuality": {
    "lightingScore": number (60-100),
    "lightingType": "string (e.g. Side-lit, Studio Diffused, Backlit, Flat Flash, Overhead Harsh)",
    "lightingDetails": "coaching on how to improve key/fill ratio or eliminate harsh glares",
    "compositionType": "string (e.g. Rule of Thirds Center, Tight Crop, Lower Angle Portrait)",
    "sharpnessScore": number (60-100),
    "photoQualityAssessment": "feedback on camera lens clean, blur, focus plane, or noise"
  },
  "styleRecommendations": [
    {
      "category": "string ('grooming' | 'accessories' | 'eyewear' | 'hairstyle' | 'color_palette')",
      "title": "string title",
      "description": "string detailed advice",
      "priority": "string ('high' | 'medium' | 'low')"
    }
  ],
  "hairstyles": [
    {
      "id": "string unique",
      "name": "string (e.g. Pompadour, Side-Swept Crop, Ivy League, Buzz)",
      "description": "style description",
      "suitabilityReason": "why it matches this specific facial contour shape"
    }
  ],
  "eyewear": [
    {
      "id": "string unique",
      "name": "string frame style name",
      "type": "string ('square' | 'aviator' | 'round' | 'browline' | 'geometric')",
      "description": "glasses details",
      "suitabilityReason": "why it balances the vertical/horizontal visual weights"
    }
  ],
  "skincare": {
    "skinTypeEstimate": "string (Oily, Dry, Combination, Sensitive)",
    "skincareScore": number (70-98),
    "primaryFocus": "skincare goal, e.g., balancing skin oils, micro-texture smoothing, hydration glow",
    "routine": [
      {
        "step": "Step index, e.g., morning",
        "productType": "Cleanser/Serum/Hydrator/SPF",
        "frequency": "Daily AM/PM",
        "benefits": "target effect"
      }
    ],
    "wellnessHabits": ["string advice, e.g. hydration limits, lymphatic massage, screen filter filters"]
  },
  "landmarks": {
    "leftEye": [
      {"x": number, "y": number, "label": "e.g. Outer Corner"}
    ],
    "rightEye": [
      {"x": number, "y": number, "label": "e.g. Outer Corner"}
    ],
    "noseBridge": [
      {"x": number, "y": number, "label": "e.g. Bridge Root"}
    ],
    "noseTip": {"x": number, "y": number, "label": "Nose Tip"},
    "mouth": [
      {"x": number, "y": number, "label": "e.g. Upper Lip Center"}
    ],
    "jawline": [
      {"x": number, "y": number, "label": "e.g. Jaw Left Point"}
    ],
    "leftEyebrow": [
      {"x": number, "y": number, "label": "e.g. Left Brow Tail"}
    ],
    "rightEyebrow": [
      {"x": number, "y": number, "label": "e.g. Right Brow Tail"}
    ],
    "faceContour": [
      {"x": number, "y": number, "label": "e.g. Forehead Crest"}
    ]
  }
}

Important:
1. Ensure all x/y coordinates are actual numerical values scaled relative to the face inside the photo, mapping where key points lie (so we can draw a beautiful glowing face mesh overlay on top of the image!). Ensure they match the actual eye, nose, mouth locations in the selfie.
2. Return ONLY clean, valid JSON. Do not write markdown blocks outside the JSON, do not include backticks unless wrapped in markdown format of JSON. Make sure the JSON parser can parse it directly.`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: [imagePart, { text: promptText }],
      config: {
        systemInstruction,
        responseMimeType: "application/json"
      }
    });

    const textOutput = response.text || "{}";
    
    // Safely parse JSON with fallback
    let parsedReport;
    try {
      // Clean backticks if any
      let cleanText = textOutput.trim();
      if (cleanText.startsWith("```json")) {
        cleanText = cleanText.substring(7);
      }
      if (cleanText.endsWith("```")) {
        cleanText = cleanText.substring(0, cleanText.length - 3);
      }
      parsedReport = JSON.parse(cleanText.trim());
    } catch (parseErr) {
      console.error("Failed to parse Gemini output:", textOutput);
      return res.status(500).json({
        error: "Failed to compile structured analysis results. Please capture a clearer portrait.",
        details: parseErr instanceof Error ? parseErr.message : String(parseErr),
        raw: textOutput
      });
    }

    // Add unique session identification & timestamp
    const sessionId = "session_" + Math.random().toString(36).substring(2, 11);
    parsedReport.id = sessionId;
    parsedReport.timestamp = new Date().toISOString();
    
    // If user opts in, or by default for active session simulation state, save it
    // Store temporarily for the simulation endpoint to reference
    sessionsCache.set(sessionId, {
      timestamp: Date.now(),
      data: parsedReport
    });

    // If user did NOT opt in, we delete or restrict access, but for active UI simulation flow, 
    // keeping it in short-lived memory (which deletes in 3 hours) is ideal.

    res.json(parsedReport);
  } catch (error: any) {
    console.error("AI Analysis error:", error);
    res.status(500).json({
      error: "The laboratory scanner encountered an error during image processing.",
      details: error?.message || String(error)
    });
  }
});

/**
 * POST /api/outfit-analysis
 * Outfit Studio: garment/colour analysis + deterministic palette suitability + AI styling.
 * Core logic lives in api/outfit-core.ts (shared with the serverless adapter).
 */
app.post("/api/outfit-analysis", async (req, res) => {
  try {
    const body = (req.body && typeof req.body === "object" ? req.body : {}) as Record<string, unknown>;
    const image = typeof body.image === "string" ? body.image : "";
    const profileRaw = (body.profile && typeof body.profile === "object" ? body.profile : {}) as Record<string, unknown>;
    const profile = {
      season: typeof profileRaw.season === "string" ? profileRaw.season : "",
      undertone: typeof profileRaw.undertone === "string" ? profileRaw.undertone : undefined,
      skin_depth: typeof profileRaw.skin_depth === "string" ? profileRaw.skin_depth : undefined,
      chroma: typeof profileRaw.chroma === "string" ? profileRaw.chroma : undefined,
      contrast: typeof profileRaw.contrast === "string" ? profileRaw.contrast : undefined,
    };
    const occasion = typeof body.occasion === "string" ? body.occasion.slice(0, 80) : undefined;
    const stylePreference = typeof body.stylePreference === "string" ? body.stylePreference.slice(0, 80) : undefined;

    if (!image) {
      return res.status(400).json({ success: false, error: "Image data (base64) is required." });
    }
    if (!profile.season) {
      return res.status(400).json({ success: false, error: "NO_PROFILE", message: "Complete your FaceMaxx colour analysis to receive styling recommendations tailored to you." });
    }

    const apiKey = process.env.GEMINI_API_KEY || null;
    const result = await runOutfitAnalysis({ image, profile, occasion, stylePreference }, apiKey);
    return res.json(result);
  } catch (error: any) {
    console.error("Outfit analysis error:", error);
    return res.status(500).json({
      success: false,
      error: "Outfit analysis failed. Please try again.",
      details: error?.message || String(error),
    });
  }
});

/**
 * POST /api/outfit-analysis/describe
 * Mode 2 — interpret a natural-language wardrobe description into structured garments.
 */
app.post("/api/outfit-analysis/describe", async (req, res) => {
  try {
    const body = (req.body && typeof req.body === "object" ? req.body : {}) as Record<string, unknown>;
    const description = typeof body.description === "string" ? body.description : "";
    const profileRaw = (body.profile && typeof body.profile === "object" ? body.profile : {}) as Record<string, unknown>;
    const profile = {
      season: typeof profileRaw.season === "string" ? profileRaw.season : "",
      undertone: typeof profileRaw.undertone === "string" ? profileRaw.undertone : undefined,
      skin_depth: typeof profileRaw.skin_depth === "string" ? profileRaw.skin_depth : undefined,
      chroma: typeof profileRaw.chroma === "string" ? profileRaw.chroma : undefined,
      contrast: typeof profileRaw.contrast === "string" ? profileRaw.contrast : undefined,
    };

    if (!description.trim()) {
      return res.status(400).json({ success: false, error: "EMPTY_DESCRIPTION", message: "Describe at least one garment." });
    }
    if (description.length > 2000) {
      return res.status(400).json({ success: false, error: "DESCRIPTION_TOO_LONG", message: "Please keep the description under 2000 characters." });
    }
    if (!profile.season) {
      return res.status(400).json({ success: false, error: "NO_PROFILE", message: "Complete your FaceMaxx colour analysis to receive styling recommendations tailored to you." });
    }

    const result = await runDescribeAnalysis({ description, profile });
    return res.json(result);
  } catch (error: any) {
    console.error("Outfit describe error:", error);
    return res.status(500).json({
      success: false,
      error: "Could not interpret the description. Please try again.",
      details: error?.message || String(error),
    });
  }
});

/**
 * POST /api/outfit-analysis/wardrobe
 * Mode 3 — analyse a user-selected wardrobe outfit (deterministic core + AI styling).
 */
app.post("/api/outfit-analysis/wardrobe", async (req, res) => {
  try {
    const body = (req.body && typeof req.body === "object" ? req.body : {}) as Record<string, unknown>;
    const garments = Array.isArray(body.garments) ? body.garments : [];
    const profileRaw = (body.profile && typeof body.profile === "object" ? body.profile : {}) as Record<string, unknown>;
    const profile = {
      season: typeof profileRaw.season === "string" ? profileRaw.season : "",
      undertone: typeof profileRaw.undertone === "string" ? profileRaw.undertone : undefined,
      skin_depth: typeof profileRaw.skin_depth === "string" ? profileRaw.skin_depth : undefined,
      chroma: typeof profileRaw.chroma === "string" ? profileRaw.chroma : undefined,
      contrast: typeof profileRaw.contrast === "string" ? profileRaw.contrast : undefined,
    };
    const occasion = typeof body.occasion === "string" ? body.occasion.slice(0, 80) : undefined;
    const stylePreference = typeof body.stylePreference === "string" ? body.stylePreference.slice(0, 80) : undefined;

    if (!profile.season) {
      return res.status(400).json({ success: false, error: "NO_PROFILE", message: "Complete your FaceMaxx colour analysis to receive styling recommendations tailored to you." });
    }

    const apiKey = process.env.GEMINI_API_KEY || null;
    const result = await runWardrobeAnalysis({ garments, profile, occasion, stylePreference }, apiKey);
    return res.json(result);
  } catch (error: any) {
    console.error("Outfit wardrobe error:", error);
    return res.status(500).json({
      success: false,
      error: "Outfit analysis failed. Please try again.",
      details: error?.message || String(error),
    });
  }
});

/**
 * POST /api/outfit-analysis/compare
 * Deterministic comparison of 2-4 outfit options (no AI involved).
 */
app.post("/api/outfit-analysis/compare", (req, res) => {
  try {
    const body = (req.body && typeof req.body === "object" ? req.body : {}) as Record<string, unknown>;
    const options = Array.isArray(body.options) ? body.options : [];
    const profileRaw = (body.profile && typeof body.profile === "object" ? body.profile : {}) as Record<string, unknown>;
    const profile = {
      season: typeof profileRaw.season === "string" ? profileRaw.season : "",
      undertone: typeof profileRaw.undertone === "string" ? profileRaw.undertone : undefined,
      skin_depth: typeof profileRaw.skin_depth === "string" ? profileRaw.skin_depth : undefined,
      chroma: typeof profileRaw.chroma === "string" ? profileRaw.chroma : undefined,
      contrast: typeof profileRaw.contrast === "string" ? profileRaw.contrast : undefined,
    };

    if (!profile.season) {
      return res.status(400).json({ success: false, error: "NO_PROFILE", message: "Complete your FaceMaxx colour analysis to compare outfits against your palette." });
    }
    if (options.length < 2) {
      return res.status(400).json({ success: false, error: "NOT_ENOUGH_OPTIONS", message: "Add at least two outfit options to compare." });
    }

    return res.json(compareOutfits(options, profile));
  } catch (error: any) {
    console.error("Outfit compare error:", error);
    return res.status(500).json({
      success: false,
      error: "Comparison failed. Please try again.",
      details: error?.message || String(error),
    });
  }
});

/**
 * POST /api/color-analysis
 * Accepts base64 selfie, runs the Python CV colour pipeline, returns season + axes + features.
 */
app.post("/api/color-analysis", (req, res) => {
  const { image } = req.body;

  if (!image) {
    return res.status(400).json({ ok: false, error: "Image data (base64) is required." });
  }

  // Isolate base64 data from potential data URL prefixes
  let base64Data = image;
  let ext = ".jpg";
  if (image.includes(";base64,")) {
    const parts = image.split(";base64,");
    const match = parts[0].match(/data:(.*?);/);
    if (match) {
      const mime = match[1];
      if (mime.includes("png")) ext = ".png";
      else if (mime.includes("webp")) ext = ".webp";
    }
    base64Data = parts[1];
  }

  // Write to temp file
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "fmax-"));
  const tmpPath = path.join(tmpDir, `selfie${ext}`);

  try {
    fs.writeFileSync(tmpPath, Buffer.from(base64Data, "base64"));
  } catch (err) {
    return res.status(500).json({ ok: false, error: "Failed to decode image data." });
  }

  // Resolve the ML directory relative to this server file
  const mlDir = path.join(process.cwd(), "ml");

  execFile(
    "python",
    ["-m", "facemax_color", "predict", tmpPath, "--root", "data"],
    { cwd: mlDir, timeout: 30000, maxBuffer: 1024 * 1024 },
    (error, stdout, stderr) => {
      // Clean up temp file
      try {
        fs.unlinkSync(tmpPath);
        fs.rmdirSync(tmpDir);
      } catch (_) {}

      if (error) {
        console.error("Colour analysis error:", error.message, stderr);
        return res.status(500).json({
          ok: false,
          error: "Colour analysis pipeline failed.",
          details: stderr || error.message,
        });
      }

      try {
        const result = JSON.parse(stdout.trim());
        res.json(result);
      } catch (parseErr) {
        console.error("Failed to parse colour analysis output:", stdout);
        return res.status(500).json({
          ok: false,
          error: "Failed to parse colour analysis results.",
          raw: stdout,
        });
      }
    }
  );
});

// Setup Vite Dev Server / Static Asset delivery
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    // Development mode
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa"
    });
    app.use(vite.middlewares);
    console.log("Vite development middleware loaded.");
  } else {
    // Production mode
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
    console.log("Serving static production assets from: " + distPath);
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`FaceMax AI Laboratory server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start the Express full-stack server:", err);
});
