import React, { useState, useRef, useCallback, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Upload, Check, ArrowLeft, Camera, X } from "lucide-react";
import { analyzeColours, readFileAsBase64, isValidImageType } from "../services/api";
import type { ColourAnalysisResponse } from "../types/api";

const GUIDELINES = [
  "Face clearly visible",
  "Natural-looking lighting",
  "Minimal filters",
  "No sunglasses",
  "Hair visible where possible",
];

const LOADING_MESSAGES = [
  "Understanding your coloring",
  "Finding your undertone",
  "Comparing contrast levels",
  "Building your palette",
];

export default function Analyze() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState(LOADING_MESSAGES[0]);
  const [error, setError] = useState<string | null>(null);

  // Camera state
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Start camera
  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
      });
      streamRef.current = stream;
      setCameraActive(true);
    } catch {
      setCameraError("Could not access camera. Please check your browser permissions.");
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  }, []);

  // Attach stream to video element
  useEffect(() => {
    if (cameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [cameraActive]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // Capture photo from camera
  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Mirror the image (selfie mode)
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const base64 = canvas.toDataURL("image/jpeg", 0.92);
    setPreview(base64);
    stopCamera();
  }, [stopCamera]);

  // Handle file upload
  const handleFile = useCallback(async (file: File) => {
    setError(null);
    if (!isValidImageType(file)) {
      setError("Please upload a JPG, PNG or WebP image.");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setError("Image is too large. Please use an image under 15 MB.");
      return;
    }
    const base64 = await readFileAsBase64(file);
    setPreview(base64);
  }, []);

  // Start analysis
  const startAnalysis = useCallback(async () => {
    if (!preview) return;
    setAnalyzing(true);
    setError(null);

    let msgIndex = 0;
    const interval = setInterval(() => {
      msgIndex = (msgIndex + 1) % LOADING_MESSAGES.length;
      setLoadingMsg(LOADING_MESSAGES[msgIndex]);
    }, 1600);

    try {
      const result: ColourAnalysisResponse = await analyzeColours(preview);
      clearInterval(interval);
      sessionStorage.setItem("fmax_analysis", JSON.stringify(result));
      sessionStorage.setItem("fmax_image", preview);
      navigate("/results");
    } catch (err: any) {
      clearInterval(interval);
      setError(err.message || "We couldn't analyze this photo. Try a clearer image with more even lighting.");
    } finally {
      setAnalyzing(false);
    }
  }, [preview, navigate]);

  const reset = () => {
    setPreview(null);
    setError(null);
    setAnalyzing(false);
    stopCamera();
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <main className="pt-16 min-h-screen">
      <div className="max-w-2xl mx-auto px-6 py-14 md:py-20">
        {/* Hidden canvas for camera capture */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Header */}
        <div className="mb-8">
          {preview && !analyzing && (
            <button
              onClick={reset}
              className="inline-flex items-center gap-1.5 text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors mb-5"
            >
              <ArrowLeft className="w-4 h-4" /> Change photo
            </button>
          )}
          <h1 className="text-3xl md:text-4xl font-serif mb-2">
            {analyzing ? "Reading your colors" : "Find your colors"}
          </h1>
          {!preview && !analyzing && !cameraActive && (
            <p className="text-sm text-[var(--color-text-secondary)]">
              Upload a clear photo or take one with your camera.
            </p>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl">
            <p className="text-sm font-medium text-red-800 mb-1">Analysis couldn&apos;t complete</p>
            <p className="text-sm text-red-600">{error}</p>
            <button onClick={reset} className="mt-3 text-sm font-medium text-red-700 underline hover:no-underline">
              Try another photo
            </button>
          </div>
        )}

        {/* Camera error */}
        {cameraError && (
          <div className="mb-6 p-4 bg-amber-50 border border-amber-200 rounded-xl text-sm text-amber-700">
            {cameraError}
          </div>
        )}

        {/* Camera active: show viewfinder */}
        {cameraActive && !preview && (
          <div className="space-y-4">
            <div className="relative rounded-2xl overflow-hidden border border-[var(--color-border-light)] bg-black aspect-[4/3]">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover scale-x-[-1]"
              />
              {/* Alignment guide */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="w-[65%] h-[70%] border border-dashed border-white/30 rounded-[100px]" />
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={stopCamera}
                className="px-5 py-3 text-sm font-medium border border-[var(--color-border)] rounded-full hover:bg-[var(--color-surface-muted)] transition-colors text-[var(--color-text-primary)]"
              >
                Cancel
              </button>
              <button
                onClick={capturePhoto}
                className="flex-1 px-5 py-3 text-sm font-medium bg-white text-[var(--color-text-primary)] rounded-full border border-[var(--color-border)] hover:bg-[var(--color-surface-muted)] transition-colors flex items-center justify-center gap-2"
              >
                <Camera className="w-4 h-4" /> Capture photo
              </button>
            </div>
          </div>
        )}

        {/* No image, no camera: show upload + camera + guidelines */}
        {!preview && !analyzing && !cameraActive && (
          <div className="space-y-6">
            {/* Upload zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              role="button"
              tabIndex={0}
              aria-label="Upload photo"
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click(); }}
              className={`border-2 border-dashed rounded-2xl p-10 md:p-14 text-center cursor-pointer transition-all ${
                dragActive
                  ? "border-[var(--color-accent)] bg-[var(--color-surface-muted)]"
                  : "border-[var(--color-border)] hover:border-[var(--color-text-tertiary)] bg-[var(--color-surface)]"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/jpg,image/png,image/webp"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
                className="hidden"
                aria-hidden="true"
              />
              <div className="w-12 h-12 rounded-full bg-[var(--color-surface-muted)] border border-[var(--color-border-light)] flex items-center justify-center mx-auto mb-4">
                <Upload className="w-5 h-5 text-[var(--color-text-secondary)]" />
              </div>
              <p className="text-base font-medium mb-1">Upload your photo</p>
              <p className="text-sm text-[var(--color-text-tertiary)] mb-2">or drag and drop here</p>
              <p className="text-xs text-[var(--color-text-tertiary)]">JPG &middot; PNG &middot; WEBP</p>
            </div>

            {/* Camera button */}
            <button
              onClick={startCamera}
              className="w-full flex items-center justify-center gap-2 px-5 py-3.5 text-sm font-medium bg-white text-[var(--color-text-primary)] rounded-full border border-[var(--color-border)] hover:bg-[var(--color-surface-muted)] transition-colors"
            >
              <Camera className="w-4 h-4" /> Take a photo with your camera
            </button>

            {/* Guidelines */}
            <div className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-xl p-5">
              <p className="text-[10px] font-medium tracking-[0.15em] uppercase text-[var(--color-text-tertiary)] mb-3">
                Photo guidance
              </p>
              <ul className="space-y-2">
                {GUIDELINES.map((g) => (
                  <li key={g} className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
                    <Check className="w-3.5 h-3.5 text-[var(--color-success)] shrink-0" />
                    {g}
                  </li>
                ))}
              </ul>
            </div>

            <p className="text-xs text-[var(--color-text-tertiary)] text-center">
              Your photo is used only to generate your analysis. It is not stored.
            </p>
          </div>
        )}

        {/* Image preview + analyze */}
        {preview && !analyzing && (
          <div className="space-y-5">
            <div className="relative rounded-2xl overflow-hidden border border-[var(--color-border-light)] bg-[var(--color-surface)]">
              <img src={preview} alt="Your uploaded photo" className="w-full max-h-[480px] object-contain" />
            </div>

            <div className="flex gap-3">
              <button
                onClick={reset}
                className="px-5 py-3 text-sm font-medium border border-[var(--color-border)] rounded-full hover:bg-[var(--color-surface-muted)] transition-colors text-[var(--color-text-primary)]"
              >
                Change photo
              </button>
              <button
                onClick={startAnalysis}
                className="flex-1 px-5 py-3 text-sm font-medium bg-white text-[var(--color-text-primary)] rounded-full border border-[var(--color-border)] hover:bg-[var(--color-surface-muted)] transition-colors"
              >
                Analyze my colors
              </button>
            </div>

            <p className="text-xs text-[var(--color-text-tertiary)] text-center">
              Your photo is used only to generate your analysis. It is not stored.
            </p>
          </div>
        )}

        {/* Analyzing state */}
        {analyzing && (
          <div className="space-y-6">
            {preview && (
              <div className="relative rounded-2xl overflow-hidden border border-[var(--color-border-light)] bg-[var(--color-surface)]">
                <img src={preview} alt="Analyzing your photo" className="w-full max-h-[380px] object-contain opacity-50" />
              </div>
            )}

            <div className="text-center py-10">
              <div className="w-8 h-8 border-2 border-[var(--color-border)] border-t-[var(--color-text-primary)] rounded-full animate-spin mx-auto mb-5" />
              <p className="text-sm font-medium mb-1">{loadingMsg}</p>
              <p className="text-xs text-[var(--color-text-tertiary)]">This usually takes a few seconds</p>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
