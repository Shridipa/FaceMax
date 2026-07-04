import { FaceMaxReport, PresetSample } from "./types";

export const SAMPLE_PRESETS: PresetSample[] = [
  {
    id: "preset_1",
    name: "Julian (Tech Innovator)",
    gender: "male",
    imageUrl: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?q=80&w=600&auto=format&fit=crop",
    description: "Casual selfie with standard home lighting and direct gaze."
  },
  {
    id: "preset_2",
    name: "Sophia (Creative Director)",
    gender: "female",
    imageUrl: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?q=80&w=600&auto=format&fit=crop",
    description: "Outdoor natural lighting with a slight tilt and balanced smile."
  },
  {
    id: "preset_3",
    name: "Marcus (Executive Partner)",
    gender: "male",
    imageUrl: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?q=80&w=600&auto=format&fit=crop",
    description: "Sharp studio portrait with slight side shadow contrast."
  }
];

// Pre-computed professional reports matching the sample photos for instant loading
export const PRESET_REPORTS: Record<string, FaceMaxReport> = {
  preset_1: {
    id: "preset_1_session",
    timestamp: new Date().toISOString(),
    imageUrl: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?q=80&w=600&auto=format&fit=crop",
    scores: {
      overallScore: 84,
      photographyScore: 78,
      groomingScore: 86,
      styleScore: 88
    },
    faceAnalysis: {
      symmetryScore: 89,
      symmetryDetails: "Excellent vertical balance. Minimal horizontal deviation between the orbital planes. Chin matches the central nasal septum perfectly.",
      symmetryHeatmapPoints: [
        { x: 49.5, y: 32, label: "Pupil Alignment" },
        { x: 50, y: 48, label: "Nose Alignment" },
        { x: 50.5, y: 65, label: "Mouth Axis Symmetry" }
      ],
      jawlineHarmony: "Clean, defined angular jawline. Minimal submental fat contour. Grooming boundaries are sharply delineated.",
      jawlineType: "Soft Square",
      eyeSpacingRatio: "1.02:1 (Slightly wider than average)",
      eyeSpacingStatus: "ideal",
      poseAssessment: "Excellent direct 0-degree posture. Keep head level. Try lifting the camera 5cm higher to eliminate bottom-up focus distortion.",
      expressionFeedback: "Warm, professional neutral expression. Gentle corners-up lip posture. Avoid tension in the lower eyelids."
    },
    photoQuality: {
      lightingScore: 76,
      lightingType: "Warm Diffused Overhead",
      lightingDetails: "Good facial fill light. Overhead position creates minor shadowing under the eye orbit and nasal tip. Perfect candidate for standard studio-key rebalancing.",
      compositionType: "Centered Headshot (Rule of Thirds)",
      sharpnessScore: 85,
      photoQualityAssessment: "Strong resolution, minimal camera blur. Very low pixel noise. High professional profile rating."
    },
    styleRecommendations: [
      {
        category: "grooming",
        title: "Beard Border Optimization",
        description: "Trim the beard line 5mm higher on the neck to accentuate the natural mandibular angle and define jaw strength.",
        priority: "high"
      },
      {
        category: "eyewear",
        title: "Geometric Browline Frames",
        description: "To balance the strong square jawline, use browline or classic Clubmaster frames. The weighted upper border directs attention upward.",
        priority: "medium"
      },
      {
        category: "hairstyle",
        title: "Textured Mid-Fade Quiff",
        description: "Short sides with length on top will elongate the head slightly, softening the robust square jaw alignment.",
        priority: "high"
      }
    ],
    hairstyles: [
      { id: "hair_1", name: "Textured Mid-Fade Quiff", description: "Short tapered sides with 2-3 inches of textured length brushed up and slightly back.", suitabilityReason: "Elongates the vertical face profile to perfectly balance a square jaw." },
      { id: "hair_2", name: "Classic Side Part Crop", description: "Clean side parting with low volume and clean scissor-cut trim on the temples.", suitabilityReason: "Creates a highly neat, traditional executive look, matching symmetrical jaw borders." },
      { id: "hair_3", name: "Modern Textured Buzz Cut", description: "Uniform short trim with minor fade detail around the temples.", suitabilityReason: "Accentuates the natural high cheekbone contours and strong angular jawline." }
    ],
    eyewear: [
      { id: "eye_1", name: "Premium Clubmaster Acetate", type: "browline", description: "Thick horn-rimmed upper profile with light gold metal bottom lens rings.", suitabilityReason: "Frames the eyes with bold upper accents, neutralizing wide jaw structures." },
      { id: "eye_2", name: "Modern Hexagonal Titanium", type: "geometric", description: "Ultra-thin, lightweight dark matte titanium frames with soft geometric corners.", suitabilityReason: "Softens square chin borders while retaining a forward-thinking tech innovator vibe." },
      { id: "eye_3", name: "Classic Soft-Square Aviator", type: "aviator", description: "Thin gold wireframes with double bridge and teardrop-softened lenses.", suitabilityReason: "Timeless style that provides horizontal visual expansion across the cheekbones." }
    ],
    skincare: {
      skinTypeEstimate: "Combination (Slight oily t-zone)",
      skincareScore: 82,
      primaryFocus: "Hydration level balance and evening t-zone skin oiliness.",
      routine: [
        { step: "Morning 08:00 AM", productType: "Gentle Gel Cleanser", frequency: "Daily AM", benefits: "Removes night sebum without stripping hydration." },
        { step: "Morning 08:15 AM", productType: "Niacinamide Serum + Oil-free SPF 30+", frequency: "Daily AM", benefits: "Regulates pore oil production and prevents UV damage." },
        { step: "Night 10:00 PM", productType: "Hyaluronic Acid Hydrating Gel", frequency: "Daily PM", benefits: "Restores moisture levels during cell recovery." }
      ],
      wellnessHabits: [
        "Drink 2.5L of mineralized water to improve sub-orbital skin elasticity.",
        "Perform light lymphatic drainage massage along the jawline to decrease morning swelling."
      ]
    },
    landmarks: {
      leftEye: [
        { x: 38, y: 35, label: "Left Pupil" },
        { x: 34, y: 35, label: "Left Outer Corner" },
        { x: 42, y: 35, label: "Left Inner Corner" }
      ],
      rightEye: [
        { x: 62, y: 35, label: "Right Pupil" },
        { x: 58, y: 35, label: "Right Inner Corner" },
        { x: 66, y: 35, label: "Right Outer Corner" }
      ],
      noseBridge: [
        { x: 50, y: 34, label: "Nose Bridge Root" },
        { x: 50, y: 44, label: "Nose Bridge Mid" }
      ],
      noseTip: { x: 50, y: 49, label: "Nose Tip" },
      mouth: [
        { x: 44, y: 62, label: "Mouth Left Corner" },
        { x: 56, y: 62, label: "Mouth Right Corner" },
        { x: 50, y: 59, label: "Upper Lip Arch" },
        { x: 50, y: 65, label: "Lower Lip Center" }
      ],
      jawline: [
        { x: 28, y: 55, label: "Jaw Upper Left" },
        { x: 32, y: 68, label: "Jaw Mid Left" },
        { x: 41, y: 76, label: "Jaw Lower Left" },
        { x: 50, y: 79, label: "Chin Apex" },
        { x: 59, y: 76, label: "Jaw Lower Right" },
        { x: 68, y: 68, label: "Jaw Mid Right" },
        { x: 72, y: 55, label: "Jaw Upper Right" }
      ],
      leftEyebrow: [
        { x: 32, y: 30, label: "Left Eyebrow Tail" },
        { x: 38, y: 28, label: "Left Eyebrow Peak" },
        { x: 44, y: 30, label: "Left Eyebrow Head" }
      ],
      rightEyebrow: [
        { x: 56, y: 30, label: "Right Eyebrow Head" },
        { x: 62, y: 28, label: "Right Eyebrow Peak" },
        { x: 68, y: 30, label: "Right Eyebrow Tail" }
      ],
      faceContour: [
        { x: 26, y: 26, label: "Forehead Left" },
        { x: 50, y: 22, label: "Forehead Center" },
        { x: 74, y: 26, label: "Forehead Right" }
      ]
    }
  },
  preset_2: {
    id: "preset_2_session",
    timestamp: new Date().toISOString(),
    imageUrl: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?q=80&w=600&auto=format&fit=crop",
    scores: {
      overallScore: 89,
      photographyScore: 92,
      groomingScore: 88,
      styleScore: 87
    },
    faceAnalysis: {
      symmetryScore: 94,
      symmetryDetails: "Incredible bilateral harmony. Highly balanced zygomatic structures and matching orbital centers. Minimal side tilt ensures top precision.",
      symmetryHeatmapPoints: [
        { x: 40, y: 37, label: "Left Eye Focus" },
        { x: 60, y: 37, label: "Right Eye Focus" },
        { x: 50, y: 64, label: "Mouth Horizontal Plane" }
      ],
      jawlineHarmony: "Soft, smooth curved jawline line that tapers elegantly to the chin apex. Clear skin borders promote excellent visual clean-ness.",
      jawlineType: "Oval",
      eyeSpacingRatio: "1.00:1 (Mathematically Balanced)",
      eyeSpacingStatus: "ideal",
      poseAssessment: "Slight shoulder tilt with frontal level gaze. Creates a dynamic, approachable look perfect for leadership bios.",
      expressionFeedback: "Perfect genuine smiling muscle engagement (Duchenne smile) with matching orbicularis oculi contraction."
    },
    photoQuality: {
      lightingScore: 94,
      lightingType: "Natural Golden Hour Diffused",
      lightingDetails: "Outstanding soft, warm key illumination from front-left. Soft fill light from the right eliminates any harsh shadows. Excellent skin depth depiction.",
      compositionType: "Standard Headshot Upper Chest Segment",
      sharpnessScore: 92,
      photoQualityAssessment: "Pristine focus. Fine textures of hair and skin are sharply recorded. Background bokeh creates exceptional separation."
    },
    styleRecommendations: [
      {
        category: "accessories",
        title: "Gold Jewelry Accents",
        description: "To complement the warm natural lighting and skin tones, thin warm-toned gold frames or minimalist earrings harmonize perfectly.",
        priority: "medium"
      },
      {
        category: "eyewear",
        title: "Geometric Acetate Round",
        description: "To add a modern intellectual contrast to the soft oval facial contours, semi-structured geometric round frames look sublime.",
        priority: "medium"
      },
      {
        category: "hairstyle",
        title: "Face-framing Layered Waves",
        description: "Soft shoulder-length waves frame the symmetrical cheekbones, adding visual motion and depth.",
        priority: "high"
      }
    ],
    hairstyles: [
      { id: "hair_1b", name: "Face-framing Waves", description: "Voluminous, loose curls or layered waves parted in the middle.", suitabilityReason: "Highlights the highly balanced cheekbone levels." },
      { id: "hair_2b", name: "Sleek Low Bun with Center Part", description: "Polished hair pulled tight into a neat low knot behind the nape.", suitabilityReason: "Presents an authoritative, minimalist modern profile." },
      { id: "hair_3b", name: "Textured Lob with Curtain Bangs", description: "Chic shoulder-skimming bob paired with soft face-framing fringe split.", suitabilityReason: "Accentuates the eye orbit line and highlights jaw symmetry." }
    ],
    eyewear: [
      { id: "eye_1b", name: "Round Matte Gold Wireframes", type: "round", description: "Ultra-thin warm metal wires with comfortable temple tips.", suitabilityReason: "Accentuates the soft, approachable, classic aesthetic." },
      { id: "eye_2b", name: "Architectural Hexagonal Acetate", type: "geometric", description: "Transparent pastel-gray acetate frame with sharp outer borders.", suitabilityReason: "Adds structured, modern industrial contrast to an oval silhouette." },
      { id: "eye_3b", name: "Classic Clubmaster Amber Tortoise", type: "browline", description: "Spotted tortoise upper wing with clear prescription glasses rings.", suitabilityReason: "Directs visual interest upwards, matching professional portraits." }
    ],
    skincare: {
      skinTypeEstimate: "Normal to Hydrated",
      skincareScore: 91,
      primaryFocus: "Maintaining epidermal skin barrier integrity and natural glow.",
      routine: [
        { step: "Morning 07:30 AM", productType: "Vitamin C Serum", frequency: "Daily AM", benefits: "Enhances glow and acts as a powerful antioxidant barrier." },
        { step: "Morning 07:45 AM", productType: "Lightweight Daily Emulsion + SPF 50", frequency: "Daily AM", benefits: "Locks moisture and prevents premature photo-aging." },
        { step: "Night 09:30 PM", productType: "Squalane Night Oil", frequency: "3x Weekly PM", benefits: "Supports overnight skin cell rejuvenation." }
      ],
      wellnessHabits: [
        "Include rich omega fatty acids (nuts, salmon) in meals to feed natural skin barrier lipds.",
        "Ensure screen filters are in warm night mode to protect face from heavy blue lights."
      ]
    },
    landmarks: {
      leftEye: [
        { x: 39, y: 37, label: "Left Pupil" },
        { x: 35, y: 37, label: "Left Outer Corner" },
        { x: 43, y: 37, label: "Left Inner Corner" }
      ],
      rightEye: [
        { x: 61, y: 37, label: "Right Pupil" },
        { x: 57, y: 37, label: "Right Inner Corner" },
        { x: 65, y: 37, label: "Right Outer Corner" }
      ],
      noseBridge: [
        { x: 50, y: 36, label: "Nose Bridge Root" },
        { x: 50, y: 45, label: "Nose Bridge Mid" }
      ],
      noseTip: { x: 50, y: 50, label: "Nose Tip" },
      mouth: [
        { x: 42, y: 64, label: "Mouth Left Corner" },
        { x: 58, y: 64, label: "Mouth Right Corner" },
        { x: 50, y: 61, label: "Upper Lip Arch" },
        { x: 50, y: 67, label: "Lower Lip Center" }
      ],
      jawline: [
        { x: 31, y: 54, label: "Jaw Upper Left" },
        { x: 34, y: 66, label: "Jaw Mid Left" },
        { x: 41, y: 75, label: "Jaw Lower Left" },
        { x: 50, y: 78, label: "Chin Apex" },
        { x: 59, y: 75, label: "Jaw Lower Right" },
        { x: 66, y: 66, label: "Jaw Mid Right" },
        { x: 69, y: 54, label: "Jaw Upper Right" }
      ],
      leftEyebrow: [
        { x: 33, y: 32, label: "Left Eyebrow Tail" },
        { x: 39, y: 30, label: "Left Eyebrow Peak" },
        { x: 45, y: 32, label: "Left Eyebrow Head" }
      ],
      rightEyebrow: [
        { x: 55, y: 32, label: "Right Eyebrow Head" },
        { x: 61, y: 30, label: "Right Eyebrow Peak" },
        { x: 67, y: 32, label: "Right Eyebrow Tail" }
      ],
      faceContour: [
        { x: 28, y: 28, label: "Forehead Left" },
        { x: 50, y: 24, label: "Forehead Center" },
        { x: 72, y: 28, label: "Forehead Right" }
      ]
    }
  },
  preset_3: {
    id: "preset_3_session",
    timestamp: new Date().toISOString(),
    imageUrl: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?q=80&w=600&auto=format&fit=crop",
    scores: {
      overallScore: 82,
      photographyScore: 80,
      groomingScore: 81,
      styleScore: 85
    },
    faceAnalysis: {
      symmetryScore: 86,
      symmetryDetails: "Great overall proportion. High cheekbone alignment. Minor chin apex offset due to head posture, but easily compensated via grooming angles.",
      symmetryHeatmapPoints: [
        { x: 37, y: 36, label: "Eye Alignment Left" },
        { x: 63, y: 36, label: "Eye Alignment Right" },
        { x: 50, y: 51, label: "Nose Alignment Tip" }
      ],
      jawlineHarmony: "Strong angular jaw structure with high cheekbones. Well-sculpted facial shadow contrast.",
      jawlineType: "Diamond",
      eyeSpacingRatio: "1.04:1 (Balanced wide orbit)",
      eyeSpacingStatus: "ideal",
      poseAssessment: "Slight head tilt back. Try tilting the chin 2 degrees down and forward to define the mandibular border line.",
      expressionFeedback: "Laser-focused professional expression. Neutral lip tension with slightly raised eyebrows."
    },
    photoQuality: {
      lightingScore: 81,
      lightingType: "High Contrast Side Key",
      lightingDetails: "Key light illuminates 70% of the face, creating beautiful diamond definition. However, some minor shadow loss occurs under the chin fold.",
      compositionType: "Close Portrait Crop",
      sharpnessScore: 88,
      photoQualityAssessment: "Highly sharp. Excellent details on eyelashes and beard follicles. No camera movement detected."
    },
    styleRecommendations: [
      {
        category: "grooming",
        title: "Chin Volume Balancing",
        description: "Let the chin beard grow 3mm longer than the side stubbles to anchor the diamond chin and elongate the lower face.",
        priority: "high"
      },
      {
        category: "eyewear",
        title: "Classic Square Hornrims",
        description: "Classic black square or trapezoid frames contrast the angular cheeks perfectly and add a commanding executive structure.",
        priority: "high"
      },
      {
        category: "hairstyle",
        title: "Pompadour Fade with High Volume",
        description: "A modern styled pompadour draws attention up and softens diamond cheek widths beautifully.",
        priority: "medium"
      }
    ],
    hairstyles: [
      { id: "hair_1c", name: "Modern Styled Pompadour", description: "Swept-up front hair with short, clean taper-faded sides.", suitabilityReason: "Creates vertical lines that harmonize wide diamond cheek arches." },
      { id: "hair_2c", name: "Textured Taper Crop", description: "Short layered hair cropped neat on top with subtle texture details.", suitabilityReason: "Clean and low maintenance, frames high cheekbones excellently." },
      { id: "hair_3c", name: "Slick Back Classic", description: "Hair combed straight back with low-shine pomade and tapered sides.", suitabilityReason: "An elegant, traditional business standard that shows full forehead symmetry." }
    ],
    eyewear: [
      { id: "eye_1c", name: "Premium Square Acetate", type: "square", description: "Bold black high-grade acetate frames with clear polished bevels.", suitabilityReason: "Bold square silhouettes frame wide eye structures with exceptional executive strength." },
      { id: "eye_2c", name: "Rimless Double-Bridge Aviator", type: "aviator", description: "Thin platinum titanium wire borders, rimless bottom.", suitabilityReason: "Minimalist weight that keeps the diamond cheekbones clear and structured." },
      { id: "eye_3c", name: "D-Frame Soft Geometric", type: "geometric", description: "Distinct flat-top D-shape horn glasses in dark gray matte color.", suitabilityReason: "Perfect combination of retro design and modern architectural elegance." }
    ],
    skincare: {
      skinTypeEstimate: "Dry to Balanced",
      skincareScore: 84,
      primaryFocus: "Enhancing surface cellular hydration and softening dry zones.",
      routine: [
        { step: "Morning 08:30 AM", productType: "Ceramide Hydrating Wash", frequency: "Daily AM", benefits: "Gently purifies skin without destroying natural lipids." },
        { step: "Morning 08:45 AM", productType: "Hyaluronic Serums + Rich SPF 35", frequency: "Daily AM", benefits: "Draws atmospheric moisture into dry cellular levels." },
        { step: "Night 11:00 PM", productType: "Night Moisture Seal Balm", frequency: "Daily PM", benefits: "Creates a protective envelope to limit nocturnal dehydration." }
      ],
      wellnessHabits: [
        "Include avocados or healthy plant oils in diet to support inner cellular hydration.",
        "Keep bedroom humidity above 45% during winter/A/C usage."
      ]
    },
    landmarks: {
      leftEye: [
        { x: 37, y: 36, label: "Left Pupil" },
        { x: 32, y: 36, label: "Left Outer Corner" },
        { x: 41, y: 36, label: "Left Inner Corner" }
      ],
      rightEye: [
        { x: 63, y: 36, label: "Right Pupil" },
        { x: 59, y: 36, label: "Right Inner Corner" },
        { x: 68, y: 36, label: "Right Outer Corner" }
      ],
      noseBridge: [
        { x: 50, y: 35, label: "Nose Bridge Root" },
        { x: 50, y: 44, label: "Nose Bridge Mid" }
      ],
      noseTip: { x: 50, y: 51, label: "Nose Tip" },
      mouth: [
        { x: 43, y: 65, label: "Mouth Left Corner" },
        { x: 57, y: 65, label: "Mouth Right Corner" },
        { x: 50, y: 62, label: "Upper Lip Arch" },
        { x: 50, y: 68, label: "Lower Lip Center" }
      ],
      jawline: [
        { x: 29, y: 56, label: "Jaw Upper Left" },
        { x: 32, y: 69, label: "Jaw Mid Left" },
        { x: 41, y: 77, label: "Jaw Lower Left" },
        { x: 50, y: 80, label: "Chin Apex" },
        { x: 59, y: 77, label: "Jaw Lower Right" },
        { x: 68, y: 69, label: "Jaw Mid Right" },
        { x: 71, y: 56, label: "Jaw Upper Right" }
      ],
      leftEyebrow: [
        { x: 31, y: 31, label: "Left Eyebrow Tail" },
        { x: 37, y: 29, label: "Left Eyebrow Peak" },
        { x: 43, y: 31, label: "Left Eyebrow Head" }
      ],
      rightEyebrow: [
        { x: 57, y: 31, label: "Right Eyebrow Head" },
        { x: 63, y: 29, label: "Right Eyebrow Peak" },
        { x: 69, y: 31, label: "Right Eyebrow Tail" }
      ],
      faceContour: [
        { x: 27, y: 27, label: "Forehead Left" },
        { x: 50, y: 23, label: "Forehead Center" },
        { x: 73, y: 27, label: "Forehead Right" }
      ]
    }
  }
};
