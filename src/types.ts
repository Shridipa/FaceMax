/**
 * TypeScript types for FaceMax AI: The AI Glow-Up Laboratory
 */

export interface LandmarkPoint {
  x: number; // 0 to 100 percent of image width
  y: number; // 0 to 100 percent of image height
  label: string;
}

export interface LandmarkMesh {
  leftEye: LandmarkPoint[];
  rightEye: LandmarkPoint[];
  noseBridge: LandmarkPoint[];
  noseTip: LandmarkPoint;
  mouth: LandmarkPoint[];
  jawline: LandmarkPoint[];
  leftEyebrow: LandmarkPoint[];
  rightEyebrow: LandmarkPoint[];
  faceContour: LandmarkPoint[];
}

export interface MetricScore {
  name: string;
  score: number;
  maxScore: number;
  assessment: string;
  recommendations: string[];
}

export interface FaceAnalysisData {
  symmetryScore: number; // 0 to 100
  symmetryDetails: string;
  symmetryHeatmapPoints: LandmarkPoint[]; // landmark points with symmetry deviations
  jawlineHarmony: string;
  jawlineType: string;
  eyeSpacingRatio: string;
  eyeSpacingStatus: "ideal" | "wide" | "close";
  poseAssessment: string;
  expressionFeedback: string;
}

export interface PhotoQualityData {
  lightingScore: number; // 0 to 100
  lightingType: string; // e.g., "Flat", "Harsh Side", "Diffused Studio", "Backlit"
  lightingDetails: string;
  compositionType: string; // e.g., "Rule of Thirds Ideal", "Too Close", "Centered"
  sharpnessScore: number; // 0 to 100
  photoQualityAssessment: string;
}

export interface StyleRecommendation {
  category: "grooming" | "accessories" | "eyewear" | "hairstyle" | "color_palette";
  title: string;
  description: string;
  priority: "high" | "medium" | "low";
}

export interface HairstyleSuggestion {
  id: string;
  name: string;
  description: string;
  suitabilityReason: string;
  silhouetteSvgPath?: string; // outline shape to render as simulation
}

export interface EyewearSuggestion {
  id: string;
  name: string;
  type: "square" | "aviator" | "round" | "browline" | "geometric";
  description: string;
  suitabilityReason: string;
}

export interface SkincareStep {
  step: string;
  productType: string;
  frequency: string;
  benefits: string;
}

export interface SkincareGuidance {
  skinTypeEstimate: string; // e.g. "Oily", "Dry", "Combination", "Sensitive"
  skincareScore: number; // 0 to 100 based on visible grooming/clarity
  primaryFocus: string;
  routine: SkincareStep[];
  wellnessHabits: string[];
}

export interface PresentationScores {
  overallScore: number; // overall presentation power
  photographyScore: number; // composition & lighting
  groomingScore: number; // hair & skin neatness
  styleScore: number; // accessories, eyewear compatibility, style alignment
}

export interface FaceMaxReport {
  id: string;
  timestamp: string;
  imageUrl: string;
  scores: PresentationScores;
  faceAnalysis: FaceAnalysisData;
  photoQuality: PhotoQualityData;
  styleRecommendations: StyleRecommendation[];
  hairstyles: HairstyleSuggestion[];
  eyewear: EyewearSuggestion[];
  skincare: SkincareGuidance;
  landmarks: LandmarkMesh;
}

export interface PresetSample {
  id: string;
  name: string;
  gender: string;
  imageUrl: string;
  description: string;
}
