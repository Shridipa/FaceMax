import { Link } from "react-router-dom";

export default function About() {
  return (
    <main className="pt-16 min-h-screen">
      <div className="max-w-2xl mx-auto px-6 py-16 md:py-24">
        <h1 className="text-3xl md:text-4xl font-serif mb-8">About FaceMaxx</h1>

        <div className="space-y-6 text-sm text-[var(--color-text-secondary)] leading-relaxed">
          <p>
            FaceMaxx is a personal color analysis tool that helps you discover which colors
            naturally complement your unique coloring.
          </p>

          <p>
            Using computer vision and color science, FaceMaxx analyzes several visual
            characteristics together:
          </p>

          <ul className="list-disc pl-5 space-y-1">
            <li>Skin depth and undertone</li>
            <li>Hair color and luminance</li>
            <li>Eye color</li>
            <li>Overall contrast between features</li>
            <li>Color saturation (chroma)</li>
          </ul>

          <p>
            It then identifies the closest color family in the 12-season system and
            creates a personalized palette of colors that work best with your natural tones.
          </p>

          <p className="font-medium text-[var(--color-text-primary)]">
            Your photo is used only to generate your analysis. It is not stored or shared.
          </p>

          <h2 className="text-lg font-semibold text-[var(--color-text-primary)] pt-4">Technical approach</h2>

          <p>
            The underlying system uses face detection, skin-region extraction, and color-space
            analysis (CIELAB) to measure your natural coloring. Machine learning models trained
            on annotated color-analysis data classify your undertone, depth, chroma and contrast,
            which are then combined to determine your season.
          </p>
        </div>

        <div className="mt-12">
          <Link
            to="/analyze"
            className="inline-flex items-center gap-2 px-6 py-3 bg-white text-[var(--color-text-primary)] text-sm font-medium rounded-full border border-[var(--color-border)]"
          >
            Try it now
          </Link>
        </div>
      </div>
    </main>
  );
}
