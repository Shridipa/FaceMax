/**
 * Outfit Studio — three styling modes on one dedicated page:
 *   1. Photograph  — upload/capture an outfit photo (AI vision + deterministic verdict)
 *   2. Describe    — natural-language wardrobe description (AI extraction, editable)
 *   3. My Wardrobe — saved garments (localStorage), select + analyse an outfit
 *
 * The suitability verdict and 0–100 score are ALWAYS deterministic (server-side
 * CIELAB against the saved FaceMaxx palette); the AI only identifies garments,
 * explains results and proposes combinations. A comparison tray lets the user
 * evaluate several options with transparent, explainable ratings.
 */
import React, { Fragment, useState, useRef, useCallback, useEffect, useMemo } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft, ArrowRight, Camera, Shirt, Sparkles, Upload, Type, Columns2, Trash2,
} from "lucide-react";
import type {
  OutfitAnalysisResponse, OutfitCombination, OutfitGarment, SuitabilityDimension,
  GarmentInput, CompareResponse,
} from "../types/outfit";
import {
  analyseOutfit, loadSavedProfile, analyseWardrobeOutfit, compareOutfitsApi,
  type WardrobeGarments,
} from "../services/outfit";
import { readFileAsBase64, isValidImageType } from "../services/api";
import { OutfitResults } from "../components/outfit/OutfitResults";
import { DescribeMode } from "../components/outfit/DescribeMode";
import { WardrobeMode } from "../components/outfit/WardrobeMode";
import { ComparisonView } from "../components/outfit/ComparisonView";

const LOADING_MESSAGES = [
  "Identifying garments",
  "Extracting outfit colours",
  "Comparing with your palette",
  "Styling your combinations",
];

const MAX_FILE_MB = 15;

const OCCASIONS = [
  "Work & office",
  "Evening out",
  "Weekend casual",
  "Travel",
  "Wedding guest",
  "Date night",
  "Interview",
];

type StudioMode = "photo" | "describe" | "wardrobe";

const MODES: { id: StudioMode; label: string; hint: string }[] = [
  { id: "photo", label: "Photograph", hint: "Upload or capture an outfit photo" },
  { id: "describe", label: "Describe Clothes", hint: "Tell me what you own in plain words" },
  { id: "wardrobe", label: "My Wardrobe", hint: "Save garments and mix your own outfits" },
];

type ComparisonOption = { label: string; garments: WardrobeGarments };

function garmentKey(garments: { category: string; hex: string }[]): string {
  return JSON.stringify(garments.map((g) => [g.category.toLowerCase(), g.hex.toUpperCase()]));
}

/** Map a response garment back onto the user's original input by id (then by category+hex). */
function responseGarmentToWardrobeGarment(
  g: OutfitGarment,
  inputs: GarmentInput[],
): WardrobeGarments[number] {
  const src =
    (g.id && inputs.find((i) => i.id === g.id)) ||
    inputs.find(
      (i) => i.hex && i.hex.toUpperCase() === g.dominantColour.hex.toUpperCase() && i.category === g.category,
    );
  return {
    id: src?.id || g.id,
    category: g.category,
    colourName: g.dominantColour.name,
    hex: g.dominantColour.hex,
    pattern: src?.pattern ?? null,
    fit: src?.fit ?? null,
  };
}

/* ------------------------------------------------------------------ */
/*  Outfit Studio page                                                 */
/* ------------------------------------------------------------------ */

export default function OutfitStudio() {
  const { profile, raw } = useMemo(() => loadSavedProfile(), []);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const objectUrlRef = useRef<string | null>(null);

  const [mode, setMode] = useState<StudioMode>("photo");

  // Shared analysis state (photo + wardrobe/describe results)
  const [analyzing, setAnalyzing] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState(LOADING_MESSAGES[0]);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<OutfitAnalysisResponse | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [lastInputs, setLastInputs] = useState<GarmentInput[]>([]);

  // Photo mode
  const [dragActive, setDragActive] = useState(false);

  // Camera state (pattern reused from Analyze.tsx)
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Comparison tray
  const [comparisonItems, setComparisonItems] = useState<ComparisonOption[]>([]);
  const [addedKeys, setAddedKeys] = useState<Set<string>>(new Set());
  const [comparison, setComparison] = useState<CompareResponse | null>(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);

  const hasProfile = Boolean(profile?.season);

  // Release object URLs / camera on unmount
  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 960 } },
      });
      streamRef.current = stream;
      setCameraActive(true);
    } catch {
      setCameraError("Could not access camera. Please check your browser permissions — upload works without it.");
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  }, []);

  useEffect(() => {
    if (cameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [cameraActive]);

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const base64 = canvas.toDataURL("image/jpeg", 0.92);
    setPreview(base64);
    stopCamera();
  }, [stopCamera]);

  const handleFile = useCallback(async (file: File) => {
    setError(null);
    if (!isValidImageType(file)) {
      setError("Please upload a JPG, PNG or WebP image.");
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setError(`Image is too large. Please use an image under ${MAX_FILE_MB} MB.`);
      return;
    }
    const base64 = await readFileAsBase64(file); // downscales like the existing flow
    setPreview(base64);
  }, []);

  const switchMode = useCallback((next: StudioMode) => {
    if (next === mode) return;
    setMode(next);
    setError(null);
    setResult(null);
    setPreview(null);
    setLastInputs([]);
  }, [mode]);

  const runWithLoading = useCallback(async (fn: () => Promise<void>) => {
    setAnalyzing(true);
    setError(null);
    setResult(null);
    let msgIndex = 0;
    const interval = setInterval(() => {
      msgIndex = (msgIndex + 1) % LOADING_MESSAGES.length;
      setLoadingMsg(LOADING_MESSAGES[msgIndex]);
    }, 1600);
    try {
      await fn();
    } finally {
      clearInterval(interval);
      setAnalyzing(false);
    }
  }, []);

  const startPhotoAnalysis = useCallback(() => {
    if (!preview || !profile) return;
    runWithLoading(async () => {
      try {
        const res = await analyseOutfit(preview, profile);
        setResult(res);
      } catch (err) {
        const e = err as Error & { code?: string };
        if (e.code === "NO_PROFILE") {
          setError("Complete your FaceMaxx colour analysis to receive styling recommendations tailored to you.");
        } else {
          setError(e.message || "Outfit analysis failed. Please try again.");
        }
      }
    });
  }, [preview, profile, runWithLoading]);

  const analyseGarmentInputs = useCallback(
    (garments: GarmentInput[], occasion: string, stylePreference: string) => {
      if (!profile || !garments.length) return;
      runWithLoading(async () => {
        try {
          // The full input — ids included — is sent so the response can be traced
          // back to the exact garments the user supplied.
          const res = await analyseWardrobeOutfit(
            garments.map((g) => ({
              id: g.id, category: g.category, colourName: g.colourName, hex: g.hex,
              pattern: g.pattern, fit: g.fit,
            })),
            profile,
            occasion || undefined,
            stylePreference || undefined,
          );
          setLastInputs(garments);
          setResult(res);
        } catch (err) {
          const e = err as Error & { code?: string };
          if (e.code === "NO_PROFILE") {
            setError("Complete your FaceMaxx colour analysis to receive styling recommendations tailored to you.");
          } else {
            setError(e.message || "Outfit analysis failed. Please try again.");
          }
        }
      });
    },
    [profile, runWithLoading],
  );

  const resetForAnotherOutfit = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    setPreview(null);
    setResult(null);
    setError(null);
    setCameraActive(false);
    setLastInputs([]);
  }, []);

  /* ---------------- comparison tray ---------------- */

  // The COMPLETE outfit (every garment in the analysed result, traced back to
  // the user's inputs) — this is what the comparison tray stores.
  const currentResultGarments: WardrobeGarments = useMemo(() => {
    if (result && result.garments.length) {
      return result.garments.map((g) => responseGarmentToWardrobeGarment(g, lastInputs));
    }
    if (lastInputs.length) {
      return lastInputs.map((g) => ({
        id: g.id,
        category: g.category, colourName: g.colourName, hex: g.hex, pattern: g.pattern, fit: g.fit,
      }));
    }
    return [];
  }, [result, lastInputs]);

  const currentKey = currentResultGarments.length ? garmentKey(currentResultGarments) : "";
  const addedCurrent = Boolean(currentKey) && addedKeys.has(currentKey);

  const addCurrentToComparison = useCallback(() => {
    if (!currentResultGarments.length) return;
    const key = garmentKey(currentResultGarments);
    if (addedKeys.has(key) || comparisonItems.length >= 4) return;
    const label =
      currentResultGarments
        .map((g) => `${g.colourName} ${g.category}`)
        .join(" + ")
        .slice(0, 60) || `Option ${comparisonItems.length + 1}`;
    // Full outfit (all garments + pattern/fit) is stored, not just the anchor.
    setComparisonItems((prev) => [...prev, { label, garments: currentResultGarments.map((g) => ({ ...g })) }]);
    setAddedKeys((prev) => new Set(prev).add(key));
  }, [currentResultGarments, addedKeys, comparisonItems.length]);

  const runComparison = useCallback(async () => {
    if (!profile) return;
    if (comparisonItems.length < 2) {
      setCompareError("Add at least two options to compare.");
      return;
    }
    setComparing(true);
    setCompareError(null);
    setComparison(null);
    try {
      setComparison(await compareOutfitsApi(comparisonItems, profile));
    } catch (err) {
      setCompareError((err as Error).message || "Comparison failed. Please try again.");
    } finally {
      setComparing(false);
    }
  }, [profile, comparisonItems]);

  const clearComparison = useCallback(() => {
    setComparisonItems([]);
    setAddedKeys(new Set());
    setComparison(null);
    setCompareError(null);
  }, []);

  /* ---------------- missing-profile empty state ---------------- */

  if (!hasProfile) {
    return (
      <main className="pt-16 min-h-screen flex items-center justify-center">
        <div className="text-center px-6">
          <Shirt className="w-8 h-8 text-[var(--color-text-tertiary)] mx-auto mb-5" />
          <h1 className="text-2xl md:text-3xl font-serif mb-3">Outfit Studio</h1>
          <p className="text-sm text-[var(--color-text-secondary)] mb-7 max-w-sm mx-auto">
            Complete your FaceMaxx colour analysis to receive styling recommendations tailored to your personal colour palette.
          </p>
          <Link
            to="/analyze"
            className="inline-flex items-center gap-2.5 px-7 py-3.5 bg-white text-[var(--color-text-primary)] text-sm font-medium rounded-full border border-[var(--color-border)] hover:bg-[var(--color-surface-muted)] transition-colors"
          >
            Start colour analysis <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </main>
    );
  }

  return (
    <main className="pt-16">
      <section className="max-w-4xl mx-auto px-6 pt-14 pb-16">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors mb-8"
        >
          <ArrowLeft className="w-4 h-4" /> Back to home
        </Link>

        <p className="text-[11px] font-medium tracking-[0.25em] uppercase text-[var(--color-text-tertiary)] mb-3">
          AI Personal Stylist &amp; Wardrobe Analysis
        </p>
        <h1 className="text-4xl md:text-5xl font-serif tracking-tight mb-4">Outfit Studio</h1>
        <p className="text-base text-[var(--color-text-secondary)] leading-relaxed max-w-xl mb-8">
          Your wardrobe, your colours, your style. Discover new ways to wear what you already own.
        </p>

        {/* Profile connection */}
        <div className="flex items-center gap-3 bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-2xl px-5 py-4 mb-8">
          <span className="w-2 h-2 rounded-full bg-[var(--color-success)]" aria-hidden="true" />
          <p className="text-sm text-[var(--color-text-secondary)]">
            Recommendations personalised with your saved analysis:{" "}
            <span className="font-medium text-[var(--color-text-primary)]">{profile!.season}</span>
            {raw?.axes?.undertone?.prediction ? ` · ${raw.axes.undertone.prediction} undertone` : ""}
            {raw?.axes?.skin_depth?.prediction ? ` · ${raw.axes.skin_depth.prediction} depth` : ""}
          </p>
          <Link
            to="/results"
            className="ml-auto text-xs font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] underline underline-offset-4"
          >
            View palette
          </Link>
        </div>

        {/* Mode selector */}
        <div className="inline-flex flex-wrap bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-full p-1 mb-10" role="tablist" aria-label="Styling mode">
          {MODES.map((m) => (
            <button
              key={m.id}
              role="tab"
              aria-selected={mode === m.id}
              title={m.hint}
              onClick={() => switchMode(m.id)}
              className={`inline-flex items-center gap-2 px-5 py-2.5 text-sm font-medium rounded-full transition-colors ${
                mode === m.id
                  ? "bg-[var(--color-accent)] text-white"
                  : "text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
              }`}
            >
              {m.id === "photo" && <Camera className="w-4 h-4" aria-hidden="true" />}
              {m.id === "describe" && <Type className="w-4 h-4" aria-hidden="true" />}
              {m.id === "wardrobe" && <Shirt className="w-4 h-4" aria-hidden="true" />}
              {m.label}
            </button>
          ))}
        </div>

        {/* ---------------- mode content ---------------- */}
        {mode === "photo" && (
          <div className="mb-12">
            {!result ? (
              <div>
                {cameraActive ? (
                  <div className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-2xl p-6">
                    <div className="rounded-xl overflow-hidden border border-[var(--color-border-light)] mb-4">
                      <video ref={videoRef} autoPlay playsInline muted className="w-full" />
                    </div>
                    <div className="flex gap-3">
                      <button
                        onClick={capturePhoto}
                        className="flex-1 inline-flex items-center justify-center gap-2 px-6 py-3 bg-[var(--color-accent)] text-white text-sm font-medium rounded-lg hover:bg-[var(--color-text-primary)] transition-colors"
                      >
                        <Camera className="w-4 h-4" /> Capture
                      </button>
                      <button
                        onClick={stopCamera}
                        className="px-6 py-3 border border-[var(--color-border)] text-sm font-medium rounded-lg hover:bg-[var(--color-surface-muted)] transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                    {cameraError && <p className="text-xs text-[var(--color-error)] mt-3">{cameraError}</p>}
                  </div>
                ) : !preview ? (
                  <div
                    className={`border-2 border-dashed rounded-2xl px-6 py-14 text-center transition-colors ${
                      dragActive
                        ? "border-[var(--color-accent)] bg-[var(--color-surface-muted)]"
                        : "border-[var(--color-border)] bg-[var(--color-surface)]"
                    }`}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragActive(true);
                    }}
                    onDragLeave={() => setDragActive(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDragActive(false);
                      const file = e.dataTransfer.files?.[0];
                      if (file) handleFile(file);
                    }}
                  >
                    <Upload className="w-6 h-6 text-[var(--color-text-tertiary)] mx-auto mb-4" />
                    <p className="text-sm font-medium mb-1">Upload an outfit photo</p>
                    <p className="text-xs text-[var(--color-text-secondary)] mb-6">
                      JPG, PNG or WebP · up to {MAX_FILE_MB} MB · garments clearly visible
                    </p>
                    <div className="flex flex-wrap gap-3 justify-center">
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="px-6 py-3 bg-white border border-[var(--color-border)] text-sm font-medium rounded-lg hover:bg-[var(--color-surface-muted)] transition-colors"
                      >
                        Choose file
                      </button>
                      <button
                        onClick={startCamera}
                        className="px-6 py-3 border border-[var(--color-border)] text-sm font-medium rounded-lg hover:bg-[var(--color-surface-muted)] transition-colors"
                      >
                        Use camera
                      </button>
                    </div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/jpeg,image/jpg,image/png,image/webp"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleFile(file);
                        e.target.value = "";
                      }}
                    />
                    {error && <p className="text-xs text-[var(--color-error)] mt-5">{error}</p>}
                    {cameraError && !cameraActive && (
                      <p className="text-xs text-[var(--color-caution)] mt-2">
                        Camera unavailable — upload still works.
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-2xl p-6">
                    <div className="flex items-start gap-5">
                      <img
                        src={preview}
                        alt="Your outfit preview"
                        className="w-32 h-32 object-cover rounded-xl border border-[var(--color-border-light)]"
                      />
                      <div className="flex-1">
                        <p className="text-sm font-medium mb-1">Ready to analyse</p>
                        <p className="text-xs text-[var(--color-text-secondary)] mb-5">
                          Garments and colours are identified from this photo; suitability is computed
                          against your saved palette.
                        </p>
                        <div className="flex flex-wrap gap-3">
                          <button
                            onClick={startPhotoAnalysis}
                            disabled={analyzing}
                            className="inline-flex items-center gap-2 px-7 py-3 bg-[var(--color-accent)] text-white text-sm font-medium rounded-lg hover:bg-[var(--color-text-primary)] transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                          >
                            {analyzing ? (
                              "Analysing…"
                            ) : (
                              <>
                                <Sparkles className="w-4 h-4" /> Analyse outfit
                              </>
                            )}
                          </button>
                          <button
                            onClick={resetForAnotherOutfit}
                            disabled={analyzing}
                            className="px-6 py-3 border border-[var(--color-border)] text-sm font-medium rounded-lg hover:bg-[var(--color-surface-muted)] transition-colors"
                          >
                            Replace image
                          </button>
                        </div>
                      </div>
                    </div>
                    {analyzing && (
                      <p className="text-xs text-[var(--color-text-secondary)] mt-5" role="status">
                        {loadingMsg}…
                      </p>
                    )}
                    {error && <p className="text-xs text-[var(--color-error)] mt-5">{error}</p>}
                  </div>
                )}
              </div>
            ) : (
              /* ---------------- photo results ---------------- */
              <div className="space-y-8">
                <OutfitResults
                  result={result}
                  photoPreview={preview}
                  onAddToComparison={addCurrentToComparison}
                  addedToComparison={addedCurrent}
                />
                <div className="flex flex-wrap gap-3 pt-2">
                  <button
                    onClick={resetForAnotherOutfit}
                    className="inline-flex items-center gap-2.5 px-7 py-3.5 bg-white text-[var(--color-text-primary)] text-sm font-medium rounded-full border border-[var(--color-border)] hover:bg-[var(--color-surface-muted)] transition-colors"
                  >
                    Analyse another outfit <ArrowRight className="w-4 h-4" />
                  </button>
                  <Link
                    to="/results"
                    className="inline-flex items-center gap-2.5 px-7 py-3.5 border border-[var(--color-border)] text-sm font-medium rounded-full hover:bg-[var(--color-surface)] transition-colors"
                  >
                    Back to my palette
                  </Link>
                </div>
              </div>
            )}
          </div>
        )}

        {mode === "describe" && (
          <div className="mb-12">
            {!result ? (
              <div>
                <DescribeMode profile={profile!} onAnalyse={analyseGarmentInputs} analyzing={analyzing} />
                {analyzing && (
                  <p className="text-xs text-[var(--color-text-secondary)] mt-5" role="status">
                    {loadingMsg}…
                  </p>
                )}
                {error && <p className="text-xs text-[var(--color-error)] mt-3">{error}</p>}
              </div>
            ) : (
              <div className="space-y-8">
                <OutfitResults
                  result={result}
                  onAddToComparison={addCurrentToComparison}
                  addedToComparison={addedCurrent}
                />
                <div className="flex flex-wrap gap-3 pt-2">
                  <button
                    onClick={() => setResult(null)}
                    className="inline-flex items-center gap-2.5 px-7 py-3.5 bg-white text-[var(--color-text-primary)] text-sm font-medium rounded-full border border-[var(--color-border)] hover:bg-[var(--color-surface-muted)] transition-colors"
                  >
                    Describe another outfit <ArrowRight className="w-4 h-4" />
                  </button>
                  <Link
                    to="/results"
                    className="inline-flex items-center gap-2.5 px-7 py-3.5 border border-[var(--color-border)] text-sm font-medium rounded-full hover:bg-[var(--color-surface)] transition-colors"
                  >
                    Back to my palette
                  </Link>
                </div>
              </div>
            )}
          </div>
        )}

        {mode === "wardrobe" && (
          <div className="mb-12">
            {!result ? (
              <div>
                <WardrobeMode onAnalyse={analyseGarmentInputs} analyzing={analyzing} />
                {analyzing && (
                  <p className="text-xs text-[var(--color-text-secondary)] mt-5" role="status">
                    {loadingMsg}…
                  </p>
                )}
                {error && <p className="text-xs text-[var(--color-error)] mt-3">{error}</p>}
              </div>
            ) : (
              <div className="space-y-8">
                <OutfitResults
                  result={result}
                  onAddToComparison={addCurrentToComparison}
                  addedToComparison={addedCurrent}
                />
                <div className="flex flex-wrap gap-3 pt-2">
                  <button
                    onClick={() => setResult(null)}
                    className="inline-flex items-center gap-2.5 px-7 py-3.5 bg-white text-[var(--color-text-primary)] text-sm font-medium rounded-full border border-[var(--color-border)] hover:bg-[var(--color-surface-muted)] transition-colors"
                  >
                    Style another selection <ArrowRight className="w-4 h-4" />
                  </button>
                  <Link
                    to="/results"
                    className="inline-flex items-center gap-2.5 px-7 py-3.5 border border-[var(--color-border)] text-sm font-medium rounded-full hover:bg-[var(--color-surface)] transition-colors"
                  >
                    Back to my palette
                  </Link>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ---------------- comparison tray + view ---------------- */}
        {(comparisonItems.length > 0 || comparison || comparing) && (
          <div className="border-t border-[var(--color-border-light)] pt-10">
            {comparisonItems.length > 0 && (
              <div className="mb-6">
                <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-3">
                  Comparison tray ({comparisonItems.length}/4)
                </p>
                <div className="flex flex-wrap gap-2">
                  {comparisonItems.map((item, i) => (
                    <span
                      key={i}
                      className="inline-flex items-center gap-2 bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-full pl-3 pr-2 py-1.5"
                    >
                      <span className="flex gap-1">
                        {item.garments.slice(0, 3).map((g, j) => (
                          <span
                            key={j}
                            className="w-4 h-4 rounded-full border border-[var(--color-border-light)]"
                            style={{ backgroundColor: g.hex }}
                            aria-hidden="true"
                          />
                        ))}
                      </span>
                      <span className="text-xs text-[var(--color-text-secondary)]">{item.label}</span>
                      <button
                        onClick={() => {
                          setComparisonItems((prev) => prev.filter((_, idx) => idx !== i));
                          setAddedKeys((prev) => {
                            const next = new Set(prev);
                            next.delete(garmentKey(item.garments));
                            return next;
                          });
                        }}
                        aria-label={`Remove ${item.label} from comparison`}
                        className="text-[var(--color-text-tertiary)] hover:text-[var(--color-error)] transition-colors"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </span>
                  ))}
                </div>
                <button
                  onClick={runComparison}
                  disabled={comparing || comparisonItems.length < 2}
                  className="mt-4 inline-flex items-center gap-2 px-6 py-3 bg-[var(--color-accent)] text-white text-sm font-medium rounded-lg hover:bg-[var(--color-text-primary)] transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  <Columns2 className="w-4 h-4" />
                  {comparing ? "Comparing…" : "Compare options"}
                </button>
                {comparisonItems.length < 2 && (
                  <p className="text-xs text-[var(--color-text-tertiary)] mt-2">
                    Analyse another outfit (in any mode) and use “Add to comparison” to build a second option.
                  </p>
                )}
              </div>
            )}

            {(comparison || comparing || compareError) && (
              <ComparisonView
                comparison={comparison}
                comparing={comparing}
                compareError={compareError}
                onClear={clearComparison}
              />
            )}
          </div>
        )}
      </section>
    </main>
  );
}
