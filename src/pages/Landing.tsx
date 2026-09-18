import { Link } from "react-router-dom";
import { ArrowRight, Upload, Sparkles, Palette, Shirt, Paintbrush, Gem, Scissors } from "lucide-react";

/* ------------------------------------------------------------------ */
/*  Sample palette for the landing page preview                        */
/* ------------------------------------------------------------------ */
const SAMPLE_PALETTE = [
  { name: "Burgundy", hex: "#6E1F2A" },
  { name: "Forest", hex: "#234B35" },
  { name: "Deep Teal", hex: "#174F52" },
  { name: "Espresso", hex: "#3A2418" },
  { name: "Warm Plum", hex: "#64384F" },
  { name: "Cream", hex: "#F2E5CF" },
];

const ANALYSIS_AXES = [
  { num: "01", title: "Undertone", desc: "Whether your coloring leans warm, cool or neutral." },
  { num: "02", title: "Skin depth", desc: "The lightness or depth of your natural skin tone." },
  { num: "03", title: "Contrast", desc: "The visual difference between your features." },
  { num: "04", title: "Chroma", desc: "How vivid or muted your natural coloring appears." },
  { num: "05", title: "Hair", desc: "Your hair color and its warmth or coolness." },
  { num: "06", title: "Eyes", desc: "Eye color depth and tone." },
];

const STEPS = [
  { num: "01", title: "Upload", desc: "Add a clear photo of yourself." },
  { num: "02", title: "Analyze", desc: "FaceMaxx studies your natural coloring." },
  { num: "03", title: "Discover", desc: "Get your season, palette and recommendations." },
];

const STYLE_CATEGORIES = [
  { icon: Shirt, title: "Clothing", desc: "What to wear, from everyday to formal." },
  { icon: Paintbrush, title: "Makeup", desc: "Lips, eyes, blush and nails that complement you." },
  { icon: Gem, title: "Jewellery", desc: "Gold, silver or rose gold — and why." },
  { icon: Scissors, title: "Hair", desc: "Color directions that harmonize with your palette." },
];

/* ------------------------------------------------------------------ */
/*  Landing Page                                                       */
/* ------------------------------------------------------------------ */
export default function Landing() {
  return (
    <main className="pt-16">
      {/* ---- Hero ---- */}
      <section className="max-w-6xl mx-auto px-6 pt-20 pb-20 md:pt-28 md:pb-28">
        <div className="grid md:grid-cols-2 gap-12 md:gap-16 items-center">
          <div>
            <p className="text-[11px] font-medium tracking-[0.25em] uppercase text-[var(--color-text-tertiary)] mb-5">
              Personal Color Analysis
            </p>
            <h1 className="text-4xl md:text-5xl lg:text-[3.5rem] font-serif leading-[1.08] tracking-tight mb-6">
              Discover the colors
              <br />
              <span className="italic">that make you look</span>
              <br />
              your best.
            </h1>
            <p className="text-base text-[var(--color-text-secondary)] leading-relaxed max-w-md mb-8">
              FaceMaxx analyzes your natural coloring to identify your personal color season
              and create a palette tailored to you.
            </p>
            <div className="flex flex-wrap gap-3">
              <Link
                to="/analyze"
                className="inline-flex items-center gap-2.5 px-7 py-3.5 bg-white text-[var(--color-text-primary)] text-sm font-medium rounded-full border border-[var(--color-border)] hover:bg-[var(--color-surface-muted)] transition-colors"
              >
                Analyze my colors <ArrowRight className="w-4 h-4" />
              </Link>
              <a
                href="#how-it-works"
                className="inline-flex items-center gap-2 px-7 py-3.5 border border-[var(--color-border)] text-sm font-medium rounded-full hover:bg-[var(--color-surface-muted)] transition-colors text-[var(--color-text-primary)]"
              >
                See how it works
              </a>
            </div>
          </div>

          {/* Hero visual: sample palette card */}
          <div className="relative flex justify-center">
            <div className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-2xl p-7 w-full max-w-[320px] shadow-[0_1px_4px_rgba(0,0,0,0.06)]">
              <div className="flex items-baseline justify-between mb-5">
                <div>
                  <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-1">
                    Your Season
                  </p>
                  <p className="text-xl font-serif">Deep Autumn</p>
                </div>
                <p className="text-[10px] text-[var(--color-text-tertiary)] font-mono">EXAMPLE</p>
              </div>

              {/* Season tags */}
              <div className="flex gap-2 mb-5">
                <span className="px-2.5 py-1 text-[10px] font-medium bg-[var(--color-surface-muted)] rounded-full text-[var(--color-text-secondary)]">Warm</span>
                <span className="px-2.5 py-1 text-[10px] font-medium bg-[var(--color-surface-muted)] rounded-full text-[var(--color-text-secondary)]">Deep</span>
                <span className="px-2.5 py-1 text-[10px] font-medium bg-[var(--color-surface-muted)] rounded-full text-[var(--color-text-secondary)]">Rich</span>
              </div>

              {/* Palette swatches */}
              <div className="grid grid-cols-3 gap-2.5">
                {SAMPLE_PALETTE.map((c) => (
                  <div key={c.hex} className="text-center">
                    <div
                      className="w-full aspect-[4/3] rounded-lg mb-1.5 border border-[var(--color-border-light)]"
                      style={{ backgroundColor: c.hex }}
                    />
                    <p className="text-[9px] text-[var(--color-text-secondary)] leading-tight font-medium">{c.name}</p>
                  </div>
                ))}
              </div>

              <div className="mt-5 pt-4 border-t border-[var(--color-border-light)] text-center">
                <p className="text-[10px] text-[var(--color-text-tertiary)]">
                  + clothing, makeup, jewellery &amp; hair guidance
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ---- What We Analyze ---- */}
      <section className="border-y border-[var(--color-border-light)] bg-[var(--color-surface)]">
        <div className="max-w-6xl mx-auto px-6 py-20">
          <p className="text-[11px] font-medium tracking-[0.25em] uppercase text-[var(--color-text-tertiary)] mb-3 text-center">
            What Makes Your Palette Yours
          </p>
          <h2 className="text-3xl md:text-4xl font-serif text-center mb-4">
            More than a color season.
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] text-center max-w-lg mx-auto mb-14">
            Your result isn&apos;t based on skin tone alone. FaceMaxx considers the relationship
            between six key characteristics to find your most harmonious colors.
          </p>

          <div className="grid md:grid-cols-3 gap-x-8 gap-y-10">
            {ANALYSIS_AXES.map((a) => (
              <div key={a.num} className="flex gap-4">
                <span className="text-xs text-[var(--color-text-tertiary)] font-mono mt-0.5 shrink-0">{a.num}</span>
                <div>
                  <h3 className="text-sm font-semibold mb-1">{a.title}</h3>
                  <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{a.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ---- How It Works ---- */}
      <section id="how-it-works" className="max-w-6xl mx-auto px-6 py-20 md:py-24">
        <p className="text-[11px] font-medium tracking-[0.25em] uppercase text-[var(--color-text-tertiary)] mb-3 text-center">
          How It Works
        </p>
        <h2 className="text-3xl md:text-4xl font-serif text-center mb-14">Three simple steps.</h2>

        <div className="grid md:grid-cols-3 gap-10 md:gap-8">
          {STEPS.map((s, i) => (
            <div key={s.num} className="text-center">
              <div className="w-11 h-11 rounded-full bg-[var(--color-surface-muted)] border border-[var(--color-border-light)] flex items-center justify-center mx-auto mb-4">
                {i === 0 && <Upload className="w-4.5 h-4.5 text-[var(--color-text-secondary)]" />}
                {i === 1 && <Sparkles className="w-4.5 h-4.5 text-[var(--color-text-secondary)]" />}
                {i === 2 && <Palette className="w-4.5 h-4.5 text-[var(--color-text-secondary)]" />}
              </div>
              <p className="text-[10px] text-[var(--color-text-tertiary)] font-mono mb-1.5">{s.num}</p>
              <h3 className="text-base font-semibold mb-1.5">{s.title}</h3>
              <p className="text-sm text-[var(--color-text-secondary)]">{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ---- Palette Preview ---- */}
      <section className="bg-[var(--color-surface)] border-y border-[var(--color-border-light)]">
        <div className="max-w-6xl mx-auto px-6 py-20 text-center">
          <p className="text-[11px] font-medium tracking-[0.25em] uppercase text-[var(--color-text-tertiary)] mb-3">
            Your Palette
          </p>
          <h2 className="text-3xl md:text-4xl font-serif mb-3">
            Colors that work with you.
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] max-w-md mx-auto mb-10">
            Every season has a distinct palette. These are the shades that naturally complement your coloring.
          </p>

          <div className="flex justify-center gap-5 md:gap-7 flex-wrap">
            {SAMPLE_PALETTE.map((c) => (
              <div key={c.hex} className="flex flex-col items-center gap-2.5">
                <div
                  className="w-16 h-16 md:w-20 md:h-20 rounded-full border border-[var(--color-border-light)] shadow-sm hover:scale-110 transition-transform duration-200"
                  style={{ backgroundColor: c.hex }}
                />
                <span className="text-xs text-[var(--color-text-secondary)] font-medium">{c.name}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ---- Style Beyond Color ---- */}
      <section className="max-w-6xl mx-auto px-6 py-20 md:py-24">
        <p className="text-[11px] font-medium tracking-[0.25em] uppercase text-[var(--color-text-tertiary)] mb-3 text-center">
          Style Beyond Color
        </p>
        <h2 className="text-3xl md:text-4xl font-serif text-center mb-4">
          Your complete style guide.
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] text-center max-w-lg mx-auto mb-12">
          Beyond your palette, FaceMaxx gives you practical guidance for clothing, makeup, jewellery and hair.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {STYLE_CATEGORIES.map((cat) => {
            const Icon = cat.icon;
            return (
              <div key={cat.title} className="text-center p-5 rounded-xl bg-[var(--color-surface)] border border-[var(--color-border-light)]">
                <Icon className="w-5 h-5 text-[var(--color-text-tertiary)] mx-auto mb-3" />
                <h3 className="text-sm font-semibold mb-1">{cat.title}</h3>
                <p className="text-xs text-[var(--color-text-secondary)]">{cat.desc}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* ---- Final CTA ---- */}
      <section className="border-t border-[var(--color-border-light)] bg-[var(--color-surface)]">
        <div className="max-w-6xl mx-auto px-6 py-20 md:py-24 text-center">
          <h2 className="text-3xl md:text-4xl font-serif mb-3">Your best colors are personal.</h2>
          <p className="text-base text-[var(--color-text-secondary)] mb-8">Discover yours.</p>
          <Link
            to="/analyze"
            className="inline-flex items-center gap-2.5 px-8 py-4 bg-white text-[var(--color-text-primary)] text-sm font-medium rounded-full border border-[var(--color-border)] hover:bg-[var(--color-surface-muted)] transition-colors"
          >
            Analyze my colors <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>
    </main>
  );
}
