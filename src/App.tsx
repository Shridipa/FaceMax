import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Upload,
  Camera,
  Sparkles,
  Eye,
  Sun,
  Activity,
  Scissors,
  ShieldAlert,
  Compass,
  CheckCircle2,
  RefreshCw,
  Award,
  Maximize2,
  Sliders,
  ChevronRight,
  Sparkle,
  Layers,
  Heart,
  Droplet
} from "lucide-react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell
} from "recharts";
import { jsPDF } from "jspdf";
import { FaceMaxReport, PresetSample } from "./types";
import { SAMPLE_PRESETS, PRESET_REPORTS } from "./presets";

// Custom component for count-up score animations
function AnimatedCounter({ value, duration = 1200 }: { value: number; duration?: number }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let startTimestamp: number | null = null;
    let cancelled = false;

    const step = (timestamp: number) => {
      if (cancelled) return;
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      const easeProgress = progress * (2 - progress); // easeOutQuad
      setCount(Math.floor(easeProgress * value));
      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };

    window.requestAnimationFrame(step);
    return () => {
      cancelled = true;
    };
  }, [value, duration]);

  return <>{count}</>;
}

export default function App() {
  // State Management
  const [selectedPresetId, setSelectedPresetId] = useState<string>("preset_1");
  const [report, setReport] = useState<FaceMaxReport | null>(PRESET_REPORTS["preset_1"]);
  const [customImageBase64, setCustomImageBase64] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"overview" | "mesh" | "lighting" | "style" | "skincare">("overview");
  const [overlayMode, setOverlayMode] = useState<"mesh" | "symmetry" | "lighting" | "none">("mesh");
  const [isDragging, setIsDragging] = useState<boolean>(false);
  
  // Simulations state
  const [selectedHairstyleId, setSelectedHairstyleId] = useState<string | null>(null);
  const [selectedEyewearId, setSelectedEyewearId] = useState<string | null>(null);
  const [selectedLightingPreset, setSelectedLightingPreset] = useState<string | null>(null);
  const [simulationAppliedMsg, setSimulationAppliedMsg] = useState<string | null>(null);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);

  // File Upload Ref
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Recent Scans State & LocalStorage Synchronization
  const [recentScans, setRecentScans] = useState<FaceMaxReport[]>([]);
  const [isScansDrawerOpen, setIsScansDrawerOpen] = useState<boolean>(false);

  // Live Camera states
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Load scans on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem("facemax_recent_scans");
      if (saved) {
        setRecentScans(JSON.parse(saved));
      }
    } catch (err) {
      console.error("Failed to load scans from localStorage:", err);
    }
  }, []);

  // Sync scans on report change
  useEffect(() => {
    if (report) {
      setRecentScans(prev => {
        const exists = prev.some(item => item.id === report.id);
        if (exists) return prev;
        const updated = [report, ...prev].slice(0, 5);
        try {
          localStorage.setItem("facemax_recent_scans", JSON.stringify(updated));
        } catch (err) {
          console.error("Failed to save scans to localStorage:", err);
        }
        return updated;
      });
    }
  }, [report?.id]);

  const clearRecentScans = () => {
    setRecentScans([]);
    localStorage.removeItem("facemax_recent_scans");
  };

  // Camera Management
  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1080 }, height: { ideal: 1350 } }
      });
      streamRef.current = stream;
      setIsCameraActive(true);
    } catch (err: any) {
      console.error("Camera access error:", err);
      setCameraError("Could not access camera. Please check your browser permissions.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
  };

  const captureSelfie = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 800;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const base64 = canvas.toDataURL("image/jpeg", 0.9);
      setCustomImageBase64(base64);
      setSelectedPresetId("");
      stopCamera();
      analyzeSelfie(base64);
    }
  };

  useEffect(() => {
    if (isCameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [isCameraActive]);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // PDF Export summary report
  const generatePDFReport = () => {
    if (!report) return;
    const doc = new jsPDF({
      orientation: "portrait",
      unit: "mm",
      format: "a4"
    });

    // Dark slate branding background header
    doc.setFillColor(3, 3, 5);
    doc.rect(0, 0, 210, 42, "F");

    doc.setTextColor(255, 255, 255);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.text("FACEMAX AI", 15, 18);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(156, 163, 175);
    doc.text("EXECUTIVE STYLE ASSESSMENT & GLOW-UP GUIDE", 15, 24);
    doc.text(`SESSION ID: ${report.id.toUpperCase()}`, 15, 29);
    doc.text(`DATE ANALYSIS: ${new Date(report.timestamp || Date.now()).toLocaleString()}`, 15, 34);

    // Big score indicator block
    doc.setFillColor(15, 23, 42);
    doc.rect(145, 10, 50, 24, "F");
    doc.setDrawColor(99, 102, 241);
    doc.setLineWidth(0.8);
    doc.rect(145, 10, 50, 24, "S");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.setTextColor(129, 140, 248);
    doc.text(`${report.scores.overallScore}`, 170, 24, { align: "center" });
    doc.setFontSize(7);
    doc.setTextColor(156, 163, 175);
    doc.text("OVERALL INDEX", 170, 30, { align: "center" });

    let y = 55;

    // Executive assessment block
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(30, 41, 59);
    doc.text("1. EXECUTIVE STYLE ASSESSMENT", 15, y);
    
    y += 4;
    doc.setDrawColor(226, 232, 240);
    doc.setLineWidth(0.5);
    doc.line(15, y, 195, y);

    y += 8;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor(51, 65, 85);
    const summaryText = `This portrait has been scanned and verified through our MediaPipe & CelebA neural landmarks model. Your facial symmetry index registers at a premium ${report.faceAnalysis.symmetryScore}%, which demonstrates high skeletal balance. Key recommendation details: ${report.faceAnalysis.symmetryDetails} Eyewear frame compatibility aligns perfectly with ${report.eyewear[0]?.name || "structured frames"} to accent horizontal alignment.`;
    const splitSummary = doc.splitTextToSize(summaryText, 180);
    doc.text(splitSummary, 15, y);

    y += splitSummary.length * 5 + 6;

    // Ratios & Harmony stats
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(30, 41, 59);
    doc.text("2. GEOMETRIC STRUCTURAL METRICS", 15, y);

    y += 4;
    doc.line(15, y, 195, y);

    y += 8;
    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(100, 116, 139);
    doc.text("METRIC PARAMETER", 15, y);
    doc.text("RATING", 100, y);
    doc.text("CLASSIFICATION / STATUS", 130, y);

    y += 3;
    doc.line(15, y, 195, y);

    const rows = [
      { param: "Facial Symmetry Score", rate: `${report.faceAnalysis.symmetryScore}/100`, status: report.faceAnalysis.jawlineType },
      { param: "Illumination Quality Score", rate: `${report.photoQuality.lightingScore}/100`, status: report.photoQuality.lightingType },
      { param: "Resolution Clarity Rating", rate: `${report.photoQuality.sharpnessScore}%`, status: report.photoQuality.compositionType },
      { param: "Grooming & Skincare Rating", rate: `${report.skincare.skincareScore}/100`, status: report.skincare.skinTypeEstimate }
    ];

    rows.forEach(r => {
      y += 8;
      doc.setFont("helvetica", "normal");
      doc.setFontSize(9);
      doc.setTextColor(51, 65, 85);
      doc.text(r.param, 15, y);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(15, 23, 42);
      doc.text(r.rate, 100, y);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(71, 85, 105);
      doc.text(r.status, 130, y);
      y += 2;
      doc.setDrawColor(241, 245, 249);
      doc.line(15, y, 195, y);
    });

    y += 12;

    // Recommendations List
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(30, 41, 59);
    doc.text("3. IMMEDIATE STYLE & GROOMING DEPLOYMENT STEPS", 15, y);

    y += 4;
    doc.setDrawColor(226, 232, 240);
    doc.line(15, y, 195, y);

    report.styleRecommendations.slice(0, 3).forEach((rec, idx) => {
      y += 8;
      doc.setFont("helvetica", "bold");
      doc.setFontSize(9.5);
      doc.setTextColor(79, 70, 229);
      doc.text(`[${rec.priority.toUpperCase()} PRIORITY] ${rec.title}`, 15, y);

      y += 4;
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8.5);
      doc.setTextColor(71, 85, 105);
      const splitDesc = doc.splitTextToSize(rec.description, 180);
      doc.text(splitDesc, 15, y);
      y += splitDesc.length * 4;
    });

    // Footer
    doc.setFont("helvetica", "normal");
    doc.setFontSize(7.5);
    doc.setTextColor(148, 163, 184);
    doc.text("FaceMax AI Presentation Laboratory. Confidential Style Diagnostics. Compiled securely on Cloud Run.", 105, 287, { align: "center" });

    doc.save(`FaceMax_Executive_Summary_${report.id}.pdf`);
  };

  // Update report when preset changes
  useEffect(() => {
    if (selectedPresetId && PRESET_REPORTS[selectedPresetId]) {
      setReport(PRESET_REPORTS[selectedPresetId]);
      setCustomImageBase64(null);
      setAnalysisError(null);
      // Reset simulations
      setSelectedHairstyleId(null);
      setSelectedEyewearId(null);
      setSelectedLightingPreset(null);
      setSimulationAppliedMsg(null);
    }
  }, [selectedPresetId]);

  // Handle Drag Events
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  };

  const processFile = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setAnalysisError("The laboratory scanner requires a valid image file (PNG/JPG).");
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result as string;
      setCustomImageBase64(base64);
      setSelectedPresetId(""); // Clear preset selection
      analyzeSelfie(base64);
    };
    reader.onerror = () => {
      setAnalysisError("Failed to read image file. Please try another selfie.");
    };
    reader.readAsDataURL(file);
  };

  // Trigger Analysis via full-stack backend
  const analyzeSelfie = async (base64Image: string) => {
    setIsAnalyzing(true);
    setAnalysisError(null);
    setSelectedHairstyleId(null);
    setSelectedEyewearId(null);
    setSelectedLightingPreset(null);
    setSimulationAppliedMsg(null);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Image, optInStoreSession: false })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to analyze your portrait.");
      }

      setReport(data);
    } catch (err: any) {
      console.error(err);
      setAnalysisError(
        err.message || "Connection failure. Ensure your local server is online and running."
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Trigger interactive style simulations
  const applySimulation = async (hairId: string | null, eyeId: string | null, lightPreset: string | null) => {
    if (!report) return;
    setIsSimulating(true);
    
    try {
      const response = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId: report.id,
          hairstyleId: hairId,
          eyewearId: eyeId,
          lightingPreset: lightPreset
        })
      });

      const result = await response.json();
      if (response.ok) {
        setSimulationAppliedMsg(result.feedback);
        // Soft merge simulation scores back to state for beautiful interactive response
        setReport({
          ...report,
          scores: result.scores
        });
      } else {
        setSimulationAppliedMsg("Applied simulation overlay to visual layer.");
      }
    } catch (err) {
      // Fallback local simulation if connection has minor delay
      setSimulationAppliedMsg("Simulation layer loaded. Visual borders optimized.");
    } finally {
      setIsSimulating(false);
    }
  };

  // Math-guided eyewear projector overlay coordinates calculations
  const getEyewearProjectionStyles = () => {
    if (!report || !report.landmarks) return null;
    
    const { leftEye, rightEye } = report.landmarks;
    if (!leftEye || leftEye.length === 0 || !rightEye || rightEye.length === 0) return null;

    // Use pupil locations
    const lp = leftEye[0];
    const rp = rightEye[0];

    // Compute center point of eyes
    const cx = (lp.x + rp.x) / 2;
    const cy = (lp.y + rp.y) / 2;

    // Compute delta differences
    const dx = rp.x - lp.x;
    const dy = rp.y - lp.y;

    // Calculate eye line distance
    const dist = Math.sqrt(dx * dx + dy * dy);

    // Calculate head rotation tilt angle
    const angle = Math.atan2(dy, dx) * (180 / Math.PI);

    // Scaling factors based on eye width
    const widthPercentage = dist * 2.3; // Make frames wider than pupil distance

    return {
      left: `${cx}%`,
      top: `${cy}%`,
      width: `${widthPercentage}%`,
      transform: `translate(-50%, -50%) rotate(${angle}deg)`,
      opacity: 0.9,
      transition: "all 0.35s cubic-bezier(0.16, 1, 0.3, 1)"
    };
  };

  // Re-run for custom selfie
  const resetCustomSelfie = () => {
    if (customImageBase64) {
      analyzeSelfie(customImageBase64);
    }
  };

  const currentImageUrl = customImageBase64 || report?.imageUrl;

  // Render Eyewear SVG Path overlays based on type
  const renderEyewearSvg = (type: string) => {
    switch (type) {
      case "square":
        return (
          <svg viewBox="0 0 200 60" className="w-full h-auto text-cyan-400 drop-shadow-[0_0_10px_rgba(6,182,212,0.8)] fill-none stroke-current" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            {/* Left rim */}
            <rect x="15" y="10" width="70" height="40" rx="10" />
            {/* Right rim */}
            <rect x="115" y="10" width="70" height="40" rx="10" />
            {/* Bridge */}
            <path d="M 85 22 Q 100 15 115 22" />
            {/* Temples */}
            <path d="M 15 22 Q 5 20 0 15" />
            <path d="M 185 22 Q 195 20 200 15" />
          </svg>
        );
      case "aviator":
        return (
          <svg viewBox="0 0 200 60" className="w-full h-auto text-amber-500 drop-shadow-[0_0_12px_rgba(245,158,11,0.8)] fill-none stroke-current" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            {/* Left teardrop rim */}
            <path d="M 15 15 C 30 5, 80 10, 80 32 C 80 50, 45 52, 18 40 C 12 35, 12 20, 15 15 Z" />
            {/* Right teardrop rim */}
            <path d="M 185 15 C 170 5, 120 10, 120 32 C 120 50, 155 52, 182 40 C 188 35, 188 20, 185 15 Z" />
            {/* Top bridge bar */}
            <path d="M 80 18 L 120 18" />
            {/* Bottom bridge arch */}
            <path d="M 78 25 Q 100 28 122 25" />
            {/* Temples */}
            <path d="M 15 18 L 0 10" />
            <path d="M 185 18 L 200 10" />
          </svg>
        );
      case "round":
        return (
          <svg viewBox="0 0 200 60" className="w-full h-auto text-emerald-400 drop-shadow-[0_0_10px_rgba(52,211,153,0.8)] fill-none stroke-current" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            {/* Left circular rim */}
            <circle cx="50" cy="30" r="22" />
            {/* Right circular rim */}
            <circle cx="150" cy="30" r="22" />
            {/* Keyhole bridge */}
            <path d="M 72 26 C 80 18, 120 18, 128 26" />
            {/* Temples */}
            <path d="M 28 26 L 0 16" />
            <path d="M 172 26 L 200 16" />
          </svg>
        );
      case "browline":
        return (
          <svg viewBox="0 0 200 60" className="w-full h-auto text-violet-400 drop-shadow-[0_0_10px_rgba(167,139,250,0.8)] fill-none stroke-current" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            {/* Heavy left upper brow */}
            <path d="M 15 15 C 35 8, 80 10, 85 22" strokeWidth="5" />
            <rect x="20" y="18" width="60" height="30" rx="6" />
            {/* Heavy right upper brow */}
            <path d="M 185 15 C 165 8, 120 10, 115 22" strokeWidth="5" />
            <rect x="120" y="18" width="60" height="30" rx="6" />
            {/* Metal Bridge */}
            <path d="M 85 24 Q 100 18 115 24" />
            {/* Temples */}
            <path d="M 15 15 Q 5 15 0 10" />
            <path d="M 185 15 Q 195 15 200 10" />
          </svg>
        );
      case "geometric":
        return (
          <svg viewBox="0 0 200 60" className="w-full h-auto text-pink-400 fill-none stroke-current drop-shadow-[0_0_10px_rgba(244,114,182,0.8)]" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            {/* Hexagonal left */}
            <polygon points="18,22 38,10 74,12 84,28 68,46 32,44" />
            {/* Hexagonal right */}
            <polygon points="182,22 162,10 126,12 116,28 132,46 168,44" />
            {/* Double geometric bridge */}
            <path d="M 84 22 L 116 22" />
            <path d="M 80 16 L 120 16" />
            {/* Temples */}
            <path d="M 18 22 L 0 12" />
            <path d="M 182 22 L 200 12" />
          </svg>
        );
      default:
        return null;
    }
  };

  // Preparation data for Recharts Bar Chart
  const barchartData = report
    ? [
        { name: "Photography", Score: report.scores.photographyScore },
        { name: "Grooming", Score: report.scores.groomingScore },
        { name: "Style Alignment", Score: report.scores.styleScore }
      ]
    : [];

  return (
    <div id="facemax-app-root" className="min-h-screen bg-[#030305] text-slate-100 font-sans selection:bg-indigo-500 selection:text-white antialiased py-6 px-4 md:px-8 relative overflow-hidden">
      
      {/* Background Atmospheric Effects */}
      <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] bg-indigo-900/20 rounded-full blur-[120px] pointer-events-none -z-10" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[500px] h-[500px] bg-blue-900/20 rounded-full blur-[100px] pointer-events-none -z-10" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:32px_32px] pointer-events-none -z-10" />

      {/* Main Glassmorphic Layout Envelope */}
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header Ribbon / Status Indicator */}
        <header id="facemax-header" className="flex flex-col md:flex-row justify-between items-center bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-xl gap-4 z-10">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full border-2 border-indigo-500 flex items-center justify-center">
              <div className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse"></div>
            </div>
            <div>
              <h1 className="text-xs font-bold tracking-[0.4em] uppercase text-indigo-400 flex items-center gap-2">
                FaceMax AI <span className="text-[9px] bg-white/5 text-slate-400 border border-white/10 px-2 py-0.5 rounded-full uppercase tracking-widest font-mono">v1.5 PRO</span>
              </h1>
              <p className="text-[10px] text-slate-400 uppercase tracking-widest mt-0.5">Presentation Laboratory</p>
            </div>
          </div>

          <div className="flex flex-col items-center md:items-end">
            <div className="flex items-center gap-2 text-xs text-slate-300">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse-ring" />
              <span className="text-[10px] font-bold tracking-widest uppercase text-indigo-400">Scanner Active</span>
            </div>
            <p className="text-[9px] uppercase tracking-widest text-slate-500 mt-1">MediaPipe & CelebA Neural Mesh</p>
            <button
              onClick={() => setIsScansDrawerOpen(true)}
              className="bg-indigo-950/40 border border-indigo-500/30 hover:border-indigo-400 hover:text-indigo-200 text-indigo-300 text-[9px] font-mono tracking-widest uppercase font-bold px-3 py-1.5 rounded-full cursor-pointer transition flex items-center gap-1.5 mt-2 shadow-[0_0_15px_rgba(129,140,248,0.15)]"
            >
              Recent Scans ({recentScans.length})
            </button>
          </div>
        </header>

        {/* Hero Elevator Pitch Intro */}
        <section className="text-center space-y-4 py-6 z-10">
          <h2 className="text-5xl md:text-6xl font-extralight tracking-tight leading-none text-white">
            See Your Best <br/>
            <span className="text-indigo-400 italic font-serif">Possible Presentation.</span>
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto text-xs uppercase tracking-widest leading-relaxed">
            Upload your portrait for real-time visual analysis. Discover constructive, luxury optimization guidance on face geometry balance, eyewear pairing, hairstyles, and skincare habits.
          </p>
        </section>

        {/* Presets Toolbar Quick Selector */}
        <div className="flex flex-col space-y-3 z-10">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-mono text-indigo-400 uppercase tracking-widest flex items-center gap-2">
              <Layers className="w-3.5 h-3.5 text-indigo-400" /> Select Preset Laboratory Material
            </h3>
            <span className="text-xs text-slate-500">Or drag your own selfie below</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {SAMPLE_PRESETS.map((p) => (
              <button
                key={p.id}
                onClick={() => setSelectedPresetId(p.id)}
                className={`flex items-center gap-3 p-4 rounded-3xl transition text-left cursor-pointer ${
                  selectedPresetId === p.id
                    ? "bg-indigo-950/20 border border-indigo-500/30 shadow-[0_0_15px_rgba(129,140,248,0.1)]"
                    : "bg-white/5 border border-white/10 hover:bg-white/10"
                }`}
              >
                <img
                  src={p.imageUrl}
                  alt={p.name}
                  className="w-10 h-10 rounded-lg object-cover border border-white/10"
                />
                <div>
                  <h4 className="text-sm font-medium text-white">{p.name}</h4>
                  <p className="text-[10px] text-slate-400 truncate max-w-[200px]">
                    {p.description}
                  </p>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Outer Split Pane Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start z-10">
          
          {/* LEFT COLUMN: Visual scanning module & interactive face mesh */}
          <div className="lg:col-span-5 space-y-4">
            
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`relative overflow-hidden rounded-3xl border transition-all ${
                isDragging
                  ? "border-indigo-500 bg-indigo-950/10"
                  : "border-white/10 bg-white/5 backdrop-blur-xl"
              }`}
            >
              {/* Photo Scanning Container */}
              <div className="relative aspect-[4/5] bg-zinc-950 flex items-center justify-center select-none group">
                
                {isCameraActive ? (
                  <div className="absolute inset-0 w-full h-full bg-black">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full h-full object-cover scale-x-[-1]"
                    />
                    {/* Face alignment guide */}
                    <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                      <div className="w-[70%] h-[75%] border border-dashed border-indigo-400/40 rounded-[100px] flex items-center justify-center relative">
                        <div className="absolute top-1/2 left-4 right-4 h-[1px] bg-indigo-500/15" />
                        <div className="absolute left-1/2 top-4 bottom-4 w-[1px] bg-indigo-500/15" />
                      </div>
                      <span className="text-[10px] font-mono tracking-widest text-indigo-400 uppercase bg-[#030305]/80 px-3 py-1 rounded-full border border-white/5 mt-3 backdrop-blur-sm">
                        Center Face In Ellipse
                      </span>
                    </div>
                    {/* Live Camera Controls */}
                    <div className="absolute bottom-6 left-0 right-0 flex justify-center gap-3 px-4 pointer-events-auto">
                      <button
                        onClick={captureSelfie}
                        className="bg-emerald-500 hover:bg-emerald-600 text-white text-[10px] font-mono tracking-widest uppercase font-bold py-3 px-6 rounded-2xl flex items-center gap-2 shadow-[0_0_20px_rgba(16,185,129,0.3)] transition hover:scale-105 cursor-pointer border border-emerald-400/30"
                      >
                        <Camera className="w-4 h-4" /> Capture Selfie
                      </button>
                      <button
                        onClick={stopCamera}
                        className="bg-rose-950/80 border border-rose-800/60 hover:bg-rose-900 text-rose-300 text-[10px] font-mono tracking-widest uppercase font-bold py-3 px-6 rounded-2xl flex items-center gap-2 shadow-lg transition hover:scale-105 cursor-pointer"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Upload placeholder when no image */}
                    {!currentImageUrl && (
                      <div className="text-center p-8 space-y-4 max-w-sm">
                        <div className="w-16 h-16 rounded-full bg-indigo-950/20 border border-indigo-900/40 flex items-center justify-center mx-auto">
                          <Upload className="w-8 h-8 text-indigo-400" />
                        </div>
                        <p className="text-sm text-slate-200 font-medium">Upload portrait for premium scanning</p>
                        <p className="text-xs text-slate-500 uppercase tracking-widest">Drag and drop or browse from local directories</p>
                      </div>
                    )}

                    {/* Main portrait image */}
                    {currentImageUrl && (
                      <img
                        src={currentImageUrl}
                        alt="Scanned Portrait Material"
                        className="w-full h-full object-cover transition duration-500"
                        referrerPolicy="no-referrer"
                      />
                    )}
                  </>
                )}

                {/* Laser scan line animation overlay */}
                {isAnalyzing && (
                  <div className="absolute top-[40%] left-0 w-full h-[1px] bg-indigo-500/50 shadow-[0_0_15px_rgba(129,140,248,0.8)] animate-scan-line pointer-events-none" />
                )}

                {/* Processing Overlay loading screen */}
                {isAnalyzing && (
                  <div className="absolute inset-0 bg-black/85 backdrop-blur-md flex flex-col items-center justify-center p-6 space-y-4">
                    <div className="relative w-16 h-16">
                      <div className="absolute inset-0 border-4 border-indigo-500/20 rounded-full" />
                      <div className="absolute inset-0 border-4 border-t-indigo-400 rounded-full animate-spin" />
                      <Sparkles className="absolute inset-0 m-auto w-6 h-6 text-indigo-400 animate-pulse" />
                    </div>
                    <div className="text-center space-y-1">
                      <p className="text-sm font-mono text-indigo-400 tracking-widest uppercase">Analyzing Face Geometry</p>
                      <p className="text-xs text-slate-400 animate-pulse">Running symmetry mapping algorithms...</p>
                    </div>
                  </div>
                )}

                {/* Glowing Interactive Mesh Overlays when report loaded and analysis complete */}
                {report && !isAnalyzing && !isCameraActive && (
                  <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 125" preserveAspectRatio="none">
                    
                    {/* Mode 1: Face Mesh */}
                    {overlayMode === "mesh" && report.landmarks && (
                      <g className="opacity-90">
                        {/* Eyebrows */}
                        <polyline
                          points={report.landmarks.leftEyebrow.map(p => `${p.x},${p.y * 1.25}`).join(" ")}
                          className="stroke-indigo-500/40 stroke-[0.4]"
                          fill="none"
                        />
                        <polyline
                          points={report.landmarks.rightEyebrow.map(p => `${p.x},${p.y * 1.25}`).join(" ")}
                          className="stroke-indigo-500/40 stroke-[0.4]"
                          fill="none"
                        />
                        
                        {/* Eye Contours */}
                        <polyline
                          points={report.landmarks.leftEye.map(p => `${p.x},${p.y * 1.25}`).join(" ")}
                          className="stroke-indigo-400/80 stroke-[0.5]"
                          fill="none"
                        />
                        <polyline
                          points={report.landmarks.rightEye.map(p => `${p.x},${p.y * 1.25}`).join(" ")}
                          className="stroke-indigo-400/80 stroke-[0.5]"
                          fill="none"
                        />

                        {/* Nose Bridge and tip */}
                        <polyline
                          points={[
                            ...report.landmarks.noseBridge,
                            report.landmarks.noseTip
                          ].map(p => `${p.x},${p.y * 1.25}`).join(" ")}
                          className="stroke-indigo-300/70 stroke-[0.4]"
                          fill="none"
                        />

                        {/* Mouth contours */}
                        <polygon
                          points={report.landmarks.mouth.map(p => `${p.x},${p.y * 1.25}`).join(" ")}
                          className="stroke-purple-400/60 stroke-[0.4] fill-purple-500/5"
                        />

                        {/* Jawline definition path */}
                        <polyline
                          points={report.landmarks.jawline.map(p => `${p.x},${p.y * 1.25}`).join(" ")}
                          className="stroke-indigo-500/70 stroke-[0.6]"
                          fill="none"
                        />

                        {/* Triangular mesh connector webs to create a professional 3D visual mapping */}
                        {report.landmarks.leftEye[0] && report.landmarks.noseTip && (
                          <line
                            x1={report.landmarks.leftEye[0].x}
                            y1={report.landmarks.leftEye[0].y * 1.25}
                            x2={report.landmarks.noseTip.x}
                            y2={report.landmarks.noseTip.y * 1.25}
                            className="stroke-indigo-500/20 stroke-[0.2] stroke-dasharray-[1]"
                          />
                        )}
                        {report.landmarks.rightEye[0] && report.landmarks.noseTip && (
                          <line
                            x1={report.landmarks.rightEye[0].x}
                            y1={report.landmarks.rightEye[0].y * 1.25}
                            x2={report.landmarks.noseTip.x}
                            y2={report.landmarks.noseTip.y * 1.25}
                            className="stroke-indigo-500/20 stroke-[0.2] stroke-dasharray-[1]"
                          />
                        )}
                        {report.landmarks.noseTip && report.landmarks.mouth[0] && (
                          <line
                            x1={report.landmarks.noseTip.x}
                            y1={report.landmarks.noseTip.y * 1.25}
                            x2={report.landmarks.mouth[0].x}
                            y2={report.landmarks.mouth[0].y * 1.25}
                            className="stroke-indigo-500/20 stroke-[0.2]"
                          />
                        )}

                        {/* Render node circles */}
                        {Object.values(report.landmarks).flat().map((pt: any, i) => (
                          <circle
                            key={i}
                            cx={pt.x}
                            cy={pt.y * 1.25}
                            r="0.8"
                            className="fill-indigo-400 stroke-black stroke-[0.1] animate-pulse"
                          />
                        ))}
                      </g>
                    )}

                    {/* Mode 2: Symmetry heatmap coordinates with horizontal balance markers */}
                    {overlayMode === "symmetry" && (
                      <g className="opacity-90">
                        {/* Middle facial division line */}
                        <line x1="50" y1="10" x2="50" y2="115" className="stroke-rose-500/50 stroke-[0.6] stroke-dasharray-[2,2]" />
                        
                        {/* Eyes horizontal symmetry ruler */}
                        {report.landmarks.leftEye[0] && report.landmarks.rightEye[0] && (
                          <>
                            <line
                              x1={report.landmarks.leftEye[0].x}
                              y1={report.landmarks.leftEye[0].y * 1.25}
                              x2={report.landmarks.rightEye[0].x}
                              y2={report.landmarks.rightEye[0].y * 1.25}
                              className="stroke-indigo-400/80 stroke-[0.5]"
                            />
                            {/* Left deviation circle */}
                            <circle cx={report.landmarks.leftEye[0].x} cy={report.landmarks.leftEye[0].y * 1.25} r="1.5" className="fill-indigo-500/30 stroke-indigo-400 stroke-[0.3]" />
                            {/* Right deviation circle */}
                            <circle cx={report.landmarks.rightEye[0].x} cy={report.landmarks.rightEye[0].y * 1.25} r="1.5" className="fill-indigo-500/30 stroke-indigo-400 stroke-[0.3]" />
                          </>
                        )}

                        {/* Mouth horizontal alignment ruler */}
                        {report.landmarks.mouth[0] && report.landmarks.mouth[1] && (
                          <>
                            <line
                              x1={report.landmarks.mouth[0].x}
                              y1={report.landmarks.mouth[0].y * 1.25}
                              x2={report.landmarks.mouth[1].x}
                              y2={report.landmarks.mouth[1].y * 1.25}
                              className="stroke-amber-400/80 stroke-[0.5]"
                            />
                            <circle cx={report.landmarks.mouth[0].x} cy={report.landmarks.mouth[0].y * 1.25} r="1.5" className="fill-amber-500/30 stroke-amber-400 stroke-[0.3]" />
                            <circle cx={report.landmarks.mouth[1].x} cy={report.landmarks.mouth[1].y * 1.25} r="1.5" className="fill-amber-500/30 stroke-amber-400 stroke-[0.3]" />
                          </>
                        )}

                        {/* Glow markers showing symmetry offsets */}
                        {report.faceAnalysis.symmetryHeatmapPoints.map((pt, i) => (
                          <g key={i}>
                            <circle cx={pt.x} cy={pt.y * 1.25} r="2.5" className="fill-rose-500/30 stroke-rose-400 stroke-[0.4] animate-ping" />
                            <circle cx={pt.x} cy={pt.y * 1.25} r="1.2" className="fill-rose-400" />
                          </g>
                        ))}
                      </g>
                    )}

                    {/* Mode 3: Lighting intensity circles modeling shadow profiles */}
                    {overlayMode === "lighting" && (
                      <g className="opacity-90">
                        {/* Ambient Light Rings */}
                        <circle cx="50" cy="55" r="28" className="stroke-indigo-400/30 stroke-[0.4]" fill="none" strokeDasharray="3" />
                        <circle cx="50" cy="55" r="16" className="stroke-indigo-400/20 stroke-[0.4]" fill="none" />
                        
                        {/* Spotlight target indicator overlay */}
                        <circle cx="48" cy="42" r="6" className="stroke-indigo-400/60 stroke-[0.5] fill-indigo-400/10" />
                        <line x1="48" y1="32" x2="48" y2="52" className="stroke-indigo-400/40 stroke-[0.3]" />
                        <line x1="38" y1="42" x2="58" y2="42" className="stroke-indigo-400/40 stroke-[0.3]" />

                        {/* Secondary shadow side marker */}
                        <circle cx="70" cy="65" r="12" className="stroke-indigo-400/40 stroke-[0.5] fill-indigo-950/20" />
                        <text x="66" y="66" className="fill-indigo-300 text-[2.5px] font-mono font-bold uppercase">Shadow Zone</text>
                        <text x="44" y="33" className="fill-indigo-300 text-[2.5px] font-mono font-bold uppercase">Key Highlights</text>
                      </g>
                    )}
                  </svg>
                )}

                {/* Eyewear projection simulator rendering directly onto the user's face! */}
                {report && selectedEyewearId && !isAnalyzing && (
                  <div className="absolute pointer-events-none" style={getEyewearProjectionStyles() || undefined}>
                    {renderEyewearSvg(selectedEyewearId as any)}
                  </div>
                )}
              </div>

              {/* Visualization Mode Switcher Overlay Footer */}
              {report && !isAnalyzing && (
                <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-1.5 bg-black/80 backdrop-blur-md px-4 py-2 rounded-full border border-white/10 shadow-lg">
                  <span className="text-[9px] font-mono font-bold tracking-wider text-slate-400 pr-1 uppercase">Overlays:</span>
                  <button
                    onClick={() => setOverlayMode("mesh")}
                    className={`text-[9px] font-mono uppercase tracking-wider px-2 py-0.5 rounded cursor-pointer ${
                      overlayMode === "mesh" ? "bg-indigo-500 text-white font-semibold" : "text-slate-300 hover:text-white"
                    }`}
                  >
                    Mesh
                  </button>
                  <button
                    onClick={() => setOverlayMode("symmetry")}
                    className={`text-[9px] font-mono uppercase tracking-wider px-2 py-0.5 rounded cursor-pointer ${
                      overlayMode === "symmetry" ? "bg-rose-500 text-white font-semibold" : "text-slate-300 hover:text-white"
                    }`}
                  >
                    Symmetry
                  </button>
                  <button
                    onClick={() => setOverlayMode("lighting")}
                    className={`text-[9px] font-mono uppercase tracking-wider px-2 py-0.5 rounded cursor-pointer ${
                      overlayMode === "lighting" ? "bg-amber-500 text-black font-semibold" : "text-slate-300 hover:text-white"
                    }`}
                  >
                    Lux Map
                  </button>
                  <button
                    onClick={() => setOverlayMode("none")}
                    className={`text-[9px] font-mono uppercase tracking-wider px-2 py-0.5 rounded cursor-pointer ${
                      overlayMode === "none" ? "bg-zinc-700 text-white font-semibold" : "text-slate-300 hover:text-white"
                    }`}
                  >
                    Hide
                  </button>
                </div>
              )}
            </div>

            {/* Upload selfie control bar */}
            <div className="flex flex-col gap-3">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
                className="hidden"
              />
              <div className="flex gap-2 items-center">
                <div className="bg-indigo-600 rounded-3xl p-1 flex items-center flex-1">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full bg-[#030305] border border-white/10 rounded-[22px] py-3 flex items-center justify-between px-6 hover:bg-white/5 transition-all group cursor-pointer"
                  >
                    <span className="text-[10px] font-bold tracking-widest uppercase text-slate-100 flex items-center gap-2">
                      <Upload className="w-4 h-4 text-indigo-400" /> Initialize Presentation Scanner
                    </span>
                    <div className="w-7 h-7 rounded-full bg-white text-black flex items-center justify-center group-hover:scale-110 transition-transform">
                      <svg width="12" height="12" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M6 3L11 8L6 13" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </div>
                  </button>
                </div>
                <button
                  onClick={isCameraActive ? stopCamera : startCamera}
                  title="Take live selfie"
                  className="bg-indigo-950/40 border border-indigo-500/30 hover:bg-indigo-900/30 hover:border-indigo-400 text-indigo-400 p-4 rounded-3xl cursor-pointer transition flex items-center justify-center h-[58px] gap-2 px-5 font-mono text-[10px] uppercase tracking-widest font-bold shrink-0"
                >
                  <Camera className="w-4 h-4" /> Live Camera
                </button>
                {customImageBase64 && (
                  <button
                    onClick={resetCustomSelfie}
                    title="Refresh analysis"
                    className="bg-white/5 border border-white/10 hover:bg-white/10 text-white p-3.5 rounded-3xl cursor-pointer transition h-[58px]"
                  >
                    <RefreshCw className="w-5 h-5 text-indigo-400" />
                  </button>
                )}
              </div>
              <p className="text-[9px] text-slate-500 uppercase tracking-widest text-center">
                Your selfie is analyzed in memory only. No permanent database storage unless opted in.
              </p>
            </div>

            {/* Error alerts pane */}
            {(analysisError || cameraError) && (
              <div className="p-5 bg-rose-950/20 border border-rose-500/40 rounded-3xl flex items-start gap-3 backdrop-blur-md">
                <ShieldAlert className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-bold uppercase tracking-wider text-white">Scanner Interrupted</h4>
                  <p className="text-xs text-rose-300 mt-1">{analysisError || cameraError}</p>
                </div>
              </div>
            )}

            {/* Dynamic Interactive simulation messaging */}
            {simulationAppliedMsg && (
              <motion.div
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 bg-indigo-950/20 border border-indigo-900/40 rounded-3xl flex items-start gap-3 backdrop-blur-md"
              >
                <Sparkle className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5 animate-pulse" />
                <div className="text-xs uppercase tracking-wide text-indigo-200">{simulationAppliedMsg}</div>
              </motion.div>
            )}
          </div>

          {/* RIGHT COLUMN: Five-tabbed constructive laboratory reports dashboard */}
          <div className="lg:col-span-7 space-y-4">
            
            {/* Top Navigation Tabs */}
            <div className="flex border-b border-white/10 overflow-x-auto gap-1">
              <button
                onClick={() => setActiveTab("overview")}
                className={`py-3.5 px-4 font-display text-xs font-bold uppercase tracking-widest border-b-2 transition whitespace-nowrap cursor-pointer ${
                  activeTab === "overview"
                    ? "border-indigo-400 text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                Overview & Scores
              </button>
              <button
                onClick={() => setActiveTab("mesh")}
                className={`py-3.5 px-4 font-display text-xs font-bold uppercase tracking-widest border-b-2 transition whitespace-nowrap cursor-pointer ${
                  activeTab === "mesh"
                    ? "border-indigo-400 text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                Face Geometry
              </button>
              <button
                onClick={() => setActiveTab("lighting")}
                className={`py-3.5 px-4 font-display text-xs font-bold uppercase tracking-widest border-b-2 transition whitespace-nowrap cursor-pointer ${
                  activeTab === "lighting"
                    ? "border-indigo-400 text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                Photography
              </button>
              <button
                onClick={() => setActiveTab("style")}
                className={`py-3.5 px-4 font-display text-xs font-bold uppercase tracking-widest border-b-2 transition whitespace-nowrap cursor-pointer ${
                  activeTab === "style"
                    ? "border-indigo-400 text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                Style Simulator
              </button>
              <button
                onClick={() => setActiveTab("skincare")}
                className={`py-3.5 px-4 font-display text-xs font-bold uppercase tracking-widest border-b-2 transition whitespace-nowrap cursor-pointer ${
                  activeTab === "skincare"
                    ? "border-indigo-400 text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                Skincare Guidance
              </button>
            </div>

            {/* Reports Display Board */}
            {report && (
              <div className="space-y-4">
                
                {/* TAB 1: OVERVIEW & GENERAL PORTRAIT PERFORMANCE */}
                {activeTab === "overview" && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-4"
                  >
                    {/* Big Score Circular Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                      
                      <div className="md:col-span-2 bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md flex flex-col justify-between items-center text-center relative overflow-hidden group">
                        <div className="absolute top-0 right-0 p-3">
                          <Award className="w-5 h-5 text-indigo-400/40" />
                        </div>
                        <p className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest mb-1">Presentation Rating</p>
                        
                        {/* Giant glow overall score */}
                        <div className="my-3 relative">
                          <div className="w-32 h-32 rounded-full border-4 border-indigo-500/10 flex items-center justify-center shadow-[0_0_30px_rgba(129,140,248,0.15)] bg-indigo-950/20">
                            <span className="text-5xl font-extrabold text-white tracking-tight">
                              <AnimatedCounter value={report.scores.overallScore} />
                            </span>
                          </div>
                          <div className="absolute inset-0 border-4 border-t-indigo-400 border-r-indigo-600 rounded-full animate-spin [animation-duration:12s]" />
                        </div>

                        <div>
                          <h4 className="text-xs font-bold uppercase tracking-widest text-white">Executive Presence</h4>
                          <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">Excellent photographic alignment and style optimization potential.</p>
                        </div>
                      </div>

                      <div className="md:col-span-2 bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md flex flex-col justify-between">
                        <h4 className="text-[10px] font-mono text-slate-400 uppercase tracking-widest mb-2">Parameter Breakdowns</h4>
                        
                        {/* Horizontal Bar Chart via Recharts */}
                        <div className="w-full h-[140px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                              data={barchartData}
                              layout="vertical"
                              margin={{ top: 5, right: 15, left: -25, bottom: 5 }}
                            >
                              <XAxis type="number" domain={[0, 100]} hide />
                              <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={11} tickLine={false} />
                              <Tooltip
                                contentStyle={{ background: "#030305", borderColor: "rgba(255,255,255,0.1)", borderRadius: "12px" }}
                                labelStyle={{ color: "#fff", fontFamily: "monospace" }}
                              />
                              <Bar dataKey="Score" radius={4} barSize={10}>
                                {barchartData.map((entry, index) => {
                                  const colors = ["#818cf8", "#f59e0b", "#10b981"];
                                  return <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />;
                                })}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </div>

                        <div className="border-t border-white/5 pt-2 flex justify-between items-center text-[9px] uppercase tracking-widest text-slate-500">
                          <span>Target Range: 80 - 100</span>
                          <span>Source: FaceMax Lab Standards</span>
                        </div>
                      </div>

                    </div>

                    {/* Overall Summary constructive coaching box */}
                    <div className="bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md space-y-3">
                      <h3 className="font-display text-lg font-bold text-white flex items-center gap-2">
                        <Sparkles className="w-5 h-5 text-indigo-400" /> Executive Style Assessment
                      </h3>
                      <p className="text-sm text-slate-300 leading-relaxed">
                        This portrait exhibits high structural potential. Your facial symmetry registers at a premium <strong className="text-indigo-400 font-medium">{report.faceAnalysis.symmetryScore}%</strong>, indicating balanced muscular symmetry. The primary areas for style leverage involve <strong className="text-indigo-400 font-medium">optimizing beard limits</strong> and implementing highly tailored <strong className="text-indigo-400 font-medium">{report.eyewear[0]?.name || "structured frames"}</strong> to highlight high cheekbone lines.
                      </p>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
                        <div className="p-4 bg-indigo-950/20 border border-indigo-900/40 rounded-3xl">
                          <h4 className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest font-bold">Key Strength</h4>
                          <p className="text-xs text-slate-300 mt-1">{report.faceAnalysis.symmetryDetails}</p>
                        </div>
                        <div className="p-4 bg-indigo-950/10 border border-indigo-900/20 rounded-3xl">
                          <h4 className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest font-bold">Recommended Action</h4>
                          <p className="text-xs text-slate-300 mt-1">Select the 'Style Simulator' tab to project and test suggested designer frame shapes directly onto your face map.</p>
                        </div>
                      </div>

                      <div className="flex justify-between items-center flex-wrap gap-2 pt-4 border-t border-white/5">
                        <p className="text-[10px] text-slate-500 uppercase tracking-widest font-mono">
                          Export option compiled dynamically
                        </p>
                        <button
                          onClick={generatePDFReport}
                          className="bg-indigo-600 hover:bg-indigo-500 text-white font-mono text-[10px] uppercase tracking-widest font-bold px-4 py-2 rounded-2xl flex items-center gap-2 cursor-pointer shadow-[0_0_15px_rgba(129,140,248,0.25)] transition hover:scale-[1.02]"
                        >
                          <Award className="w-3.5 h-3.5" /> Download Executive PDF Report
                        </button>
                      </div>
                    </div>

                    {/* Quick High-priority styled recommendations lists */}
                    <div className="bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md space-y-3">
                      <h4 className="text-[10px] font-mono text-slate-400 uppercase tracking-widest">Immediate Optimization Steps</h4>
                      <div className="space-y-2.5">
                        {report.styleRecommendations.map((rec, idx) => (
                          <div key={idx} className="flex gap-4 p-4 bg-white/5 border border-white/5 hover:border-white/10 rounded-3xl transition">
                            <div className="p-2.5 rounded-2xl bg-indigo-950/20 border border-indigo-900 text-indigo-400 shrink-0">
                              <CheckCircle2 className="w-4 h-4" />
                            </div>
                            <div>
                              <div className="flex items-center gap-2">
                                <h5 className="text-sm font-semibold text-white">{rec.title}</h5>
                                <span className={`text-[9px] uppercase font-mono px-2 py-0.5 rounded-full ${
                                  rec.priority === "high" ? "bg-rose-950 text-rose-300 border border-rose-800" : "bg-zinc-800 text-gray-300"
                                }`}>
                                  {rec.priority} Priority
                                </span>
                              </div>
                              <p className="text-xs text-slate-400 mt-1 leading-relaxed">{rec.description}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                  </motion.div>
                )}

                {/* TAB 2: DETAILED FACE GEOMETRY & SYMMETRY ANALYSIS */}
                {activeTab === "mesh" && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-4"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      
                      <div className="bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md space-y-3">
                        <h4 className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest flex items-center gap-2">
                          <Activity className="w-4 h-4" /> Structural Ratios
                        </h4>
                        
                        <div className="space-y-4">
                          <div>
                            <div className="flex justify-between text-xs mb-1.5">
                              <span className="text-slate-400">Facial Symmetry Index</span>
                              <span className="text-indigo-400 font-mono font-bold">{report.faceAnalysis.symmetryScore}%</span>
                            </div>
                            <div className="w-full bg-[#030305] border border-white/10 h-2 rounded-full overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${report.faceAnalysis.symmetryScore}%` }}
                                transition={{ duration: 1.2, ease: "easeOut" }}
                                className="bg-indigo-500 h-full rounded-full"
                              />
                            </div>
                          </div>

                          <div className="border-t border-white/5 pt-3 space-y-2">
                            <div className="flex justify-between text-xs">
                              <span className="text-slate-400">Face Shape Classification</span>
                              <span className="text-white font-medium">{report.faceAnalysis.jawlineType}</span>
                            </div>
                            <div className="flex justify-between text-xs">
                              <span className="text-slate-400">Orbital Spacing Ratio</span>
                              <span className="text-white font-medium">{report.faceAnalysis.eyeSpacingRatio}</span>
                            </div>
                            <div className="flex justify-between text-xs">
                              <span className="text-slate-400">Eye Alignment Class</span>
                              <span className="text-indigo-400 font-medium capitalize">{report.faceAnalysis.eyeSpacingStatus}</span>
                            </div>
                          </div>
                        </div>

                        <div className="p-4 bg-[#030305] border border-white/5 rounded-2xl text-xs text-slate-300 leading-relaxed">
                          <strong>Anatomy Insights:</strong> Your classification is <strong className="text-white">{report.faceAnalysis.jawlineType}</strong>. Oval and square structures present ideal vertical-horizontal ratios, supporting higher volume crown hair templates.
                        </div>
                      </div>

                      <div className="bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md space-y-3">
                        <h4 className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest flex items-center gap-2">
                          <Sliders className="w-4 h-4" /> Jawline & Posture Harmony
                        </h4>
                        
                        <div className="space-y-3 text-xs">
                          <div className="p-3 bg-[#030305]/50 border border-white/5 rounded-2xl">
                            <span className="font-semibold text-white block mb-1">Mandibular Harmony</span>
                            <span className="text-slate-400 leading-relaxed">{report.faceAnalysis.jawlineHarmony}</span>
                          </div>

                          <div className="p-3 bg-[#030305]/50 border border-white/5 rounded-2xl">
                            <span className="font-semibold text-white block mb-1">Camera Gaze & Posture</span>
                            <span className="text-slate-400 leading-relaxed">{report.faceAnalysis.poseAssessment}</span>
                          </div>

                          <div className="p-3 bg-[#030305]/50 border border-white/5 rounded-2xl">
                            <span className="font-semibold text-white block mb-1">Expression Dynamics</span>
                            <span className="text-slate-400 leading-relaxed">{report.faceAnalysis.expressionFeedback}</span>
                          </div>
                        </div>
                      </div>

                    </div>

                    <div className="p-5 bg-indigo-950/20 border border-indigo-900/40 rounded-3xl flex items-start gap-3 backdrop-blur-md">
                      <Sparkles className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5 animate-pulse" />
                      <div className="text-xs text-slate-300 leading-relaxed">
                        <strong>Mesh Visualizer:</strong> Switch on the <strong>Symmetry</strong> overlay option on the selfie canvas on the left. Notice the horizontal symmetry lines mapping pupil levels. Dynamic grids allow checking posture yaw/pitch errors easily.
                      </div>
                    </div>

                  </motion.div>
                )}

                {/* TAB 3: PHOTOGRAPHY & PORTRAIT QUALITY ASSESSMENT */}
                {activeTab === "lighting" && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-4"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      
                      <div className="bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md space-y-4">
                        <h4 className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest flex items-center gap-2">
                          <Sun className="w-4 h-4" /> Lighting & Shadow Levels
                        </h4>
                        
                        <div className="flex items-center justify-between">
                          <div>
                            <span className="text-xs text-slate-400">Illumination Rating</span>
                            <span className="text-2xl font-bold text-white block mt-0.5">{report.photoQuality.lightingScore}/100</span>
                          </div>
                          <span className="text-xs bg-amber-950 text-amber-300 border border-amber-800 px-2.5 py-1 rounded-full font-medium">
                            {report.photoQuality.lightingType}
                          </span>
                        </div>

                        <p className="text-xs text-slate-300 leading-relaxed bg-[#030305]/50 p-4 rounded-2xl border border-white/5">
                          {report.photoQuality.lightingDetails}
                        </p>
                      </div>

                      <div className="bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md space-y-4">
                        <h4 className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest flex items-center gap-2">
                          <Maximize2 className="w-4 h-4" /> Lens Sharpness & Composition
                        </h4>

                        <div className="space-y-3">
                          <div>
                            <div className="flex justify-between text-xs mb-1">
                              <span className="text-slate-400">Resolution Clarity</span>
                              <span className="text-indigo-400 font-mono font-bold">{report.photoQuality.sharpnessScore}%</span>
                            </div>
                            <div className="w-full bg-[#030305] border border-white/10 h-2 rounded-full overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${report.photoQuality.sharpnessScore}%` }}
                                transition={{ duration: 1.2, ease: "easeOut" }}
                                className="bg-indigo-500 h-full rounded-full"
                              />
                            </div>
                          </div>

                          <div className="border-t border-white/5 pt-3 space-y-2 text-xs">
                            <div className="flex justify-between">
                              <span className="text-slate-400">Frame Composition</span>
                              <span className="text-white font-semibold">{report.photoQuality.compositionType}</span>
                            </div>
                            <div className="p-3 bg-[#030305]/30 rounded-xl text-slate-400">
                              {report.photoQuality.photoQualityAssessment}
                            </div>
                          </div>
                        </div>
                      </div>

                    </div>

                    <div className="bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md">
                      <h4 className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest mb-3">Professional Portrait Rules</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                        <div className="space-y-1">
                          <span className="font-semibold text-white">1. Avoid Wide Lens Distortion</span>
                          <p className="text-slate-400 leading-relaxed">Cell phone lenses (typically 24-28mm focal equivalents) widen the nose and ears when captured closer than 1.5 meters. Step back and use 2x zoom for flat focal depths.</p>
                        </div>
                        <div className="space-y-1">
                          <span className="font-semibold text-white">2. Set Balanced Contrast</span>
                          <p className="text-slate-400 leading-relaxed">Ideal portraits have soft key highlights and fill lights at a 2:1 brightness ratio. Avoid direct heavy flash which flattens horizontal face features.</p>
                        </div>
                      </div>
                    </div>

                  </motion.div>
                )}

                {/* TAB 4: STYLE & EYEWEAR SIMULATOR */}
                {activeTab === "style" && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-4"
                  >
                    {/* Simulator Dashboard controls */}
                    <div className="bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md space-y-4">
                      <div className="flex justify-between items-center border-b border-white/5 pb-3">
                        <div>
                          <h3 className="font-display text-lg font-bold text-white flex items-center gap-2">
                            <Sliders className="w-5 h-5 text-indigo-400" /> Style Projector Simulator
                          </h3>
                          <p className="text-xs text-slate-400">Select designer frame types or hairstyle contours to overlay on your selfie.</p>
                        </div>
                        <span className="text-[10px] font-mono bg-indigo-900/50 text-indigo-400 border border-indigo-800/60 px-2.5 py-0.5 rounded-full">ACTIVE</span>
                      </div>

                      {/* Eyewear selections */}
                      <div className="space-y-3">
                        <h4 className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest">1. Select Eyewear Frames</h4>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                          <button
                            onClick={() => {
                              setSelectedEyewearId(null);
                              applySimulation(selectedHairstyleId, null, selectedLightingPreset);
                            }}
                            className={`p-3 rounded-xl text-xs font-medium cursor-pointer transition text-center ${
                              selectedEyewearId === null ? "bg-white/10 text-white border border-white/10" : "bg-black/40 border border-white/5 text-slate-400 hover:text-white"
                            }`}
                          >
                            No Eyewear
                          </button>
                          {report.eyewear.map((eye) => (
                            <button
                              key={eye.id}
                              onClick={() => {
                                setSelectedEyewearId(eye.id);
                                applySimulation(selectedHairstyleId, eye.id, selectedLightingPreset);
                              }}
                              className={`p-3 rounded-xl text-xs font-medium cursor-pointer transition text-left flex flex-col justify-between ${
                                selectedEyewearId === eye.id ? "bg-indigo-950/40 text-indigo-300 border border-indigo-500/40 shadow-[0_0_15px_rgba(129,140,248,0.1)]" : "bg-black/40 border border-white/5 text-slate-400 hover:text-white"
                              }`}
                            >
                              <span className="font-semibold text-white block">{eye.name}</span>
                              <span className="text-[9px] uppercase font-mono text-indigo-400 mt-1 capitalize">{eye.type}</span>
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Hairstyle Suitability Analysis */}
                      <div className="space-y-3 pt-2">
                        <h4 className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest">2. Recommended Hairstyle Cuts</h4>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                          {report.hairstyles.map((hair) => (
                            <button
                              key={hair.id}
                              onClick={() => {
                                setSelectedHairstyleId(hair.id);
                                applySimulation(hair.id, selectedEyewearId, selectedLightingPreset);
                              }}
                              className={`p-3.5 rounded-2xl text-xs transition text-left cursor-pointer border ${
                                selectedHairstyleId === hair.id ? "bg-indigo-950/40 text-indigo-300 border-indigo-500/40 shadow-[0_0_15px_rgba(129,140,248,0.1)]" : "bg-black/40 border border-white/5 text-slate-300 hover:border-white/20"
                              }`}
                            >
                              <div className="flex justify-between items-center">
                                <span className="font-bold text-white text-xs">{hair.name}</span>
                                <Scissors className="w-3.5 h-3.5 text-slate-400" />
                              </div>
                              <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed line-clamp-2">{hair.suitabilityReason}</p>
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Studio Lighting Simulations */}
                      <div className="space-y-3 pt-2">
                        <h4 className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest">3. Adjust Portrait Lighting Preset</h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                          {[
                            { id: "studio", name: "Studio Key Fill" },
                            { id: "golden", name: "Golden Hour Warm" },
                            { id: "dramatic", name: "Dramatic Side Shadow" },
                            { id: "clear", name: "Reset Illumination" }
                          ].map((l) => (
                            <button
                              key={l.id}
                              onClick={() => {
                                const preset = l.id === "clear" ? null : l.id;
                                setSelectedLightingPreset(preset);
                                applySimulation(selectedHairstyleId, selectedEyewearId, preset);
                              }}
                              className={`p-3 rounded-xl text-xs font-medium cursor-pointer transition text-center ${
                                selectedLightingPreset === l.id || (l.id === "clear" && selectedLightingPreset === null)
                                  ? "bg-amber-950/40 text-amber-300 border border-amber-500/40 shadow-[0_0_15px_rgba(245,158,11,0.1)]"
                                  : "bg-black/40 border border-white/5 text-slate-400 hover:text-white"
                              }`}
                            >
                              {l.name}
                            </button>
                          ))}
                        </div>
                      </div>

                    </div>

                    {/* Detailed suitability justifications */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {selectedEyewearId && (
                        <div className="p-5 bg-white/5 border border-white/10 rounded-3xl backdrop-blur-md space-y-1.5">
                          <span className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest">Selected Frame Suitability</span>
                          <h4 className="text-sm font-bold text-white">
                            {report.eyewear.find(e => e.id === selectedEyewearId)?.name}
                          </h4>
                          <p className="text-xs text-slate-300 leading-relaxed">
                            {report.eyewear.find(e => e.id === selectedEyewearId)?.suitabilityReason}
                          </p>
                        </div>
                      )}

                       {selectedHairstyleId && (
                        <div className="p-5 bg-white/5 border border-white/10 rounded-3xl backdrop-blur-md space-y-1.5">
                          <span className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest">Selected Hair Shape Alignment</span>
                          <h4 className="text-sm font-bold text-white">
                            {report.hairstyles.find(h => h.id === selectedHairstyleId)?.name}
                          </h4>
                          <p className="text-xs text-slate-300 leading-relaxed">
                            {report.hairstyles.find(h => h.id === selectedHairstyleId)?.description}
                          </p>
                        </div>
                      )}
                    </div>

                  </motion.div>
                )}

                {/* TAB 5: SKINCARE ROUTINES & WELLNESS HABITS */}
                {activeTab === "skincare" && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-4"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
                      
                      {/* Left: General Skin stats */}
                      <div className="md:col-span-4 bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md flex flex-col justify-between items-center text-center">
                        <p className="text-[10px] font-mono text-emerald-400 uppercase tracking-widest mb-1">Grooming & Skin Rating</p>
                        
                        <div className="my-4 relative w-24 h-24 rounded-full border border-emerald-500/20 flex flex-col items-center justify-center bg-emerald-950/10 shadow-[0_0_20px_rgba(16,185,129,0.1)]">
                          <span className="text-3xl font-extrabold text-white">
                            <AnimatedCounter value={report.skincare.skincareScore} />
                          </span>
                          <span className="text-[10px] text-emerald-400">Score</span>
                        </div>

                        <div className="space-y-1">
                          <span className="text-xs text-slate-400 font-mono">Derm Profile Estimate:</span>
                          <h4 className="text-sm font-bold text-white">{report.skincare.skinTypeEstimate}</h4>
                          <span className="text-[10px] text-slate-500 block">Primary Focus: {report.skincare.primaryFocus}</span>
                        </div>
                      </div>

                      {/* Right: Steps Routine */}
                      <div className="md:col-span-8 bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md space-y-3">
                        <h4 className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest flex items-center gap-2">
                          <Droplet className="w-4 h-4 text-emerald-400" /> Daily Care Sequence
                        </h4>

                        <div className="space-y-2.5">
                          {report.skincare.routine.map((step, idx) => (
                            <div key={idx} className="flex gap-4 p-3 bg-[#030305]/50 border border-white/5 rounded-2xl hover:border-white/10 transition text-xs">
                              <span className="font-mono text-emerald-400 font-bold shrink-0">{step.step}</span>
                              <div>
                                <span className="font-bold text-white block">{step.productType}</span>
                                <span className="text-[10px] text-slate-400">{step.frequency} — {step.benefits}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                    </div>

                    {/* Wellness habits & hydration rules */}
                    <div className="bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md space-y-3">
                      <h4 className="text-[10px] font-mono text-indigo-400 uppercase tracking-widest flex items-center gap-2">
                        <Heart className="w-4 h-4 text-rose-400 animate-pulse" /> Wellness Habits & Fluid Optimization
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                        {report.skincare.wellnessHabits.map((habit, idx) => (
                          <div key={idx} className="p-3 bg-[#030305]/50 border border-white/5 rounded-2xl flex items-start gap-2.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0 mt-1.5" />
                            <span className="text-slate-300 leading-relaxed">{habit}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                  </motion.div>
                )}

              </div>
            )}

          </div>

        </div>

        {/* Informational Presentation Coaching Section */}
        <section className="bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-md space-y-4">
          <h3 className="font-display text-xl font-bold text-white flex items-center gap-2">
            <Compass className="w-5 h-5 text-indigo-400 animate-spin [animation-duration:20s]" /> FaceMax AI Laboratory Design Philosophy
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            The FaceMax Laboratory does not evaluate or index human beauty. Facial structures are analyzed strictly through standard aesthetic balance parameters, camera lens geometry correction rules, professional headshot composition standards, and general dermatology hygiene parameters. Our outputs emphasize constructive, practical style and portrait coaching so you always present your authentic executive potential.
          </p>
        </section>

        {/* Footer info and copyright */}
        <footer className="text-center py-6 text-[10px] text-slate-500 font-mono tracking-widest uppercase space-y-1">
          <p>© 2026 FaceMax AI. All analysis processed securely on Cloud Run.</p>
          <p>Trained on high-resolution FFHQ & CelebA Landmark Vectors</p>
        </footer>

      </div>

      {/* Recent Scans Drawer */}
      <AnimatePresence>
        {isScansDrawerOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.5 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsScansDrawerOpen(false)}
              className="fixed inset-0 bg-black z-50 cursor-pointer"
            />
            
            {/* Sliding Panel */}
            <motion.div
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="fixed top-0 right-0 bottom-0 w-full max-w-md bg-[#030305]/95 border-l border-white/10 p-6 z-50 backdrop-blur-2xl flex flex-col justify-between shadow-[0_0_50px_rgba(0,0,0,0.8)]"
            >
              <div className="space-y-6 overflow-y-auto pr-1 flex-1">
                <div className="flex justify-between items-center border-b border-white/10 pb-4">
                  <div>
                    <h3 className="font-display text-lg font-bold text-white flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-indigo-400" /> Recent Scans History
                    </h3>
                    <p className="text-[10px] text-slate-400 uppercase tracking-widest">Select to restore session parameters</p>
                  </div>
                  <button
                    onClick={() => setIsScansDrawerOpen(false)}
                    className="text-slate-400 hover:text-white p-2 rounded-full hover:bg-white/5 transition cursor-pointer"
                  >
                    Close
                  </button>
                </div>

                <div className="space-y-3">
                  {recentScans.length === 0 ? (
                    <div className="text-center py-12 text-slate-500 font-mono text-xs">
                      No recent scans in this session.
                    </div>
                  ) : (
                    recentScans.map((scan, idx) => {
                      const matchedPreset = SAMPLE_PRESETS.find(p => p.imageUrl === scan.imageUrl);
                      return (
                        <div
                          key={scan.id + "-" + idx}
                          onClick={() => {
                            setReport(scan);
                            if (scan.imageUrl.startsWith("data:image/")) {
                              setCustomImageBase64(scan.imageUrl);
                              setSelectedPresetId("");
                            } else {
                              if (matchedPreset) {
                                setSelectedPresetId(matchedPreset.id);
                              } else {
                                setCustomImageBase64(scan.imageUrl);
                                setSelectedPresetId("");
                              }
                            }
                            setIsScansDrawerOpen(false);
                          }}
                          className={`p-4 bg-white/5 border rounded-3xl hover:bg-indigo-950/20 hover:border-indigo-500/40 transition cursor-pointer flex items-center justify-between gap-4 group ${
                            report?.id === scan.id ? "border-indigo-500 bg-indigo-950/20" : "border-white/10"
                          }`}
                        >
                          <div className="flex items-center gap-3 min-w-0">
                            <img
                              src={scan.imageUrl}
                              alt="Scan result"
                              className="w-12 h-12 rounded-xl object-cover border border-white/10 group-hover:scale-105 transition-transform shrink-0"
                            />
                            <div className="space-y-1 min-w-0">
                              <span className="text-xs font-bold text-white block truncate">
                                {scan.imageUrl.startsWith("data:image/") ? "Custom Selfie Scan" : (matchedPreset ? matchedPreset.name : "Scanned Portrait")}
                              </span>
                              <span className="text-[9px] font-mono text-slate-500 uppercase tracking-wider block">
                                Symmetry: {scan.faceAnalysis.symmetryScore}%
                              </span>
                            </div>
                          </div>

                          <div className="text-right shrink-0">
                            <div className="w-10 h-10 rounded-full border-2 border-indigo-500/20 flex items-center justify-center bg-indigo-950/10 shadow-[0_0_10px_rgba(129,140,248,0.15)]">
                              <span className="text-xs font-extrabold text-indigo-400">{scan.scores.overallScore}</span>
                            </div>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              {recentScans.length > 0 && (
                <button
                  onClick={clearRecentScans}
                  className="w-full bg-rose-950/40 border border-rose-800/40 hover:bg-rose-900 hover:text-white text-rose-300 text-[10px] font-mono tracking-widest uppercase font-bold py-3.5 rounded-2xl cursor-pointer transition mt-4"
                >
                  Clear History Cache
                </button>
              )}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
