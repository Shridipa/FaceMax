import React, { useState, useEffect, useMemo, useRef } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, ChevronDown, Check, Shirt, Paintbrush, Gem, Scissors } from "lucide-react";
import type { ColourAnalysisResponse, ColourSwatch, SeasonPalette } from "../types/api";
import { SEASON_DESCRIPTIONS, AXIS_LABELS } from "../types/api";
import { getStyleGuide, type FullStyleGuide } from "../services/stylingRecommendations";

/* ------------------------------------------------------------------ */
/*  Large editorial swatch (fabric-chip style)                         */
/* ------------------------------------------------------------------ */
function PaletteSwatch({ swatch }: { swatch: ColourSwatch }) {
  return (
    <div className="group">
      <div
        className="w-full aspect-[4/3] rounded-lg border border-[var(--color-border-light)] group-hover:scale-[1.03] transition-transform duration-200 shadow-sm"
        style={{ backgroundColor: swatch.hex }}
      />
      <div className="mt-2.5">
        <p className="text-xs font-semibold uppercase tracking-wide">{swatch.name}</p>
        <p className="text-[10px] text-[var(--color-text-tertiary)] font-mono mt-0.5">{swatch.hex}</p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Small swatch for neutrals/caution                                  */
/* ------------------------------------------------------------------ */
function SmallSwatch({ swatch }: { swatch: ColourSwatch }) {
  return (
    <div className="flex items-center gap-3 group">
      <div
        className="w-12 h-12 rounded-lg border border-[var(--color-border-light)] shrink-0 group-hover:scale-110 transition-transform duration-200"
        style={{ backgroundColor: swatch.hex }}
      />
      <div>
        <p className="text-xs font-medium">{swatch.name}</p>
        <p className="text-[10px] text-[var(--color-text-tertiary)] font-mono">{swatch.hex}</p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Profile row                                                        */
/* ------------------------------------------------------------------ */
function ProfileRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between py-3.5 border-b border-[var(--color-border-light)] last:border-b-0">
      <span className="text-[11px] font-medium tracking-[0.15em] uppercase text-[var(--color-text-tertiary)]">{label}</span>
      <span className="text-sm font-medium text-[var(--color-text-primary)]">{AXIS_LABELS[value] || value}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Color Checker                                                      */
/* ------------------------------------------------------------------ */
function ColorChecker() {
  const [hex, setHex] = useState("#3A2418");
  const [result, setResult] = useState<string | null>(null);

  const check = () => {
    if (!/^#[0-9A-Fa-f]{6}$/.test(hex)) return;
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    const brightness = (r + g + b) / 3;
    if (brightness < 60 || brightness > 200) setResult("excellent");
    else if (brightness < 100 || brightness > 160) setResult("good");
    else setResult("neutral");
  };

  const labels: Record<string, { text: string; color: string }> = {
    excellent: { text: "Excellent match", color: "text-[var(--color-success)]" },
    good: { text: "Good match", color: "text-[var(--color-text-primary)]" },
    neutral: { text: "Use carefully", color: "text-[var(--color-caution)]" },
  };

  return (
    <div>
      <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-1">Interactive</p>
      <h2 className="text-2xl md:text-3xl font-serif mb-1">Wondering about a color?</h2>
      <p className="text-sm text-[var(--color-text-secondary)] mb-6">Enter any color to see how it relates to your palette.</p>
      <div className="flex items-center gap-3">
        <input type="color" value={hex} onChange={(e) => { setHex(e.target.value); setResult(null); }} className="w-12 h-12 rounded-lg border border-[var(--color-border-light)] cursor-pointer" aria-label="Color picker" />
        <input type="text" value={hex} onChange={(e) => { setHex(e.target.value); setResult(null); }} className="flex-1 px-4 py-2.5 border border-[var(--color-border)] rounded-lg text-sm font-mono bg-[var(--color-bg)] text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)]" placeholder="#7B1E2B" aria-label="Hex color" />
        <button onClick={check} className="px-5 py-2.5 text-sm font-medium bg-[var(--color-accent)] text-white rounded-lg hover:bg-[var(--color-text-primary)] transition-colors">Check</button>
      </div>
      {result && labels[result] && (
        <div className="mt-5 flex items-center gap-4">
          <div className="w-14 h-14 rounded-xl border border-[var(--color-border-light)] shadow-sm" style={{ backgroundColor: hex }} />
          <div>
            <p className={`text-base font-semibold ${labels[result].color}`}>{labels[result].text}</p>
            <p className="text-xs text-[var(--color-text-tertiary)] font-mono mt-0.5">{hex}</p>
          </div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Section divider                                                    */
/* ------------------------------------------------------------------ */
function Divider() {
  return <hr className="border-[var(--color-border-light)] my-14" />;
}

/* ------------------------------------------------------------------ */
/*  Results Page                                                       */
/* ------------------------------------------------------------------ */
export default function Results() {
  const [data, setData] = useState<ColourAnalysisResponse | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [activeStyleTab, setActiveStyleTab] = useState<"clothing" | "makeup" | "jewelry" | "hair">("clothing");
  const paletteRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem("fmax_analysis");
    const storedImage = sessionStorage.getItem("fmax_image");
    if (stored) { try { setData(JSON.parse(stored)); } catch {} }
    if (storedImage) setImage(storedImage);
  }, []);

  const styleGuide: FullStyleGuide | null = useMemo(() => data ? getStyleGuide(data) : null, [data]);

  if (!data) {
    return (
      <main className="pt-16 min-h-screen flex items-center justify-center">
        <div className="text-center px-6">
          <h1 className="text-2xl font-serif mb-4">Your result couldn&apos;t be loaded.</h1>
          <p className="text-sm text-[var(--color-text-secondary)] mb-6">We couldn&apos;t retrieve your color profile.</p>
          <div className="flex gap-3 justify-center">
            <Link to="/analyze" className="inline-flex items-center gap-2 px-6 py-3 bg-white text-[var(--color-text-primary)] text-sm font-medium rounded-full border border-[var(--color-border)] hover:bg-[var(--color-surface-muted)] transition-colors">Analyze another photo</Link>
          </div>
        </div>
      </main>
    );
  }

  const season = data.season.primary;
  const seasonInfo = SEASON_DESCRIPTIONS[season] || {
    family: "",
    description: `Your natural coloring is best complemented by the ${season} palette.`,
    palette: { best: [], neutrals: [], caution: [] } as SeasonPalette,
  };
  const palette = seasonInfo.palette;

  // Build descriptive phrase from axes
  const depthLabel = AXIS_LABELS[data.axes.skin_depth.prediction] || data.axes.skin_depth.prediction;
  const undertoneLabel = AXIS_LABELS[data.axes.undertone.prediction] || data.axes.undertone.prediction;
  const chromaLabel = AXIS_LABELS[data.axes.chroma.prediction] || data.axes.chroma.prediction;

  const styleTabs = [
    { key: "clothing" as const, label: "Clothing", icon: Shirt },
    { key: "makeup" as const, label: "Makeup", icon: Paintbrush },
    { key: "jewelry" as const, label: "Jewellery", icon: Gem },
    { key: "hair" as const, label: "Hair", icon: Scissors },
  ];

  return (
    <main className="pt-16 min-h-screen">
      {/* ---- HERO ---- */}
      <section className="bg-[var(--color-surface)] border-b border-[var(--color-border-light)]">
        <div className="max-w-3xl mx-auto px-6 py-14 md:py-20">
          <Link to="/analyze" className="inline-flex items-center gap-1.5 text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors mb-10">
            <ArrowLeft className="w-4 h-4" /> New analysis
          </Link>

          <div className="flex flex-col sm:flex-row items-start gap-8">
            {image && (
              <div className="w-28 h-28 sm:w-32 sm:h-32 rounded-2xl overflow-hidden border border-[var(--color-border-light)] shrink-0 shadow-sm">
                <img src={image} alt="Your analyzed photo" className="w-full h-full object-cover" />
              </div>
            )}
            <div className="flex-1">
              <p className="text-[10px] font-medium tracking-[0.25em] uppercase text-[var(--color-text-tertiary)] mb-3">Your Color Profile</p>
              <h1 className="text-5xl md:text-6xl lg:text-7xl font-serif leading-[1] tracking-tight mb-4">{season}</h1>
              <p className="text-base text-[var(--color-text-secondary)] leading-relaxed max-w-md mb-4">
                {seasonInfo.description}
              </p>
              <div className="flex flex-wrap gap-2 mb-4">
                {[undertoneLabel, depthLabel, chromaLabel].map((w) => (
                  <span key={w} className="px-3 py-1 text-[11px] font-medium bg-[var(--color-surface-muted)] border border-[var(--color-border-light)] rounded-full text-[var(--color-text-secondary)]">
                    {w}
                  </span>
                ))}
              </div>
              {data.season.secondary && (
                <p className="text-sm text-[var(--color-text-tertiary)]">
                  Nearest neighbor: <span className="text-[var(--color-text-secondary)] font-medium">{data.season.secondary}</span>
                </p>
              )}
            </div>
          </div>

          {/* Scroll to palette */}
          <button onClick={() => paletteRef.current?.scrollIntoView({ behavior: "smooth" })} className="mt-10 text-xs font-medium text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] transition-colors uppercase tracking-widest">
            Explore your palette ↓
          </button>
        </div>
      </section>

      {/* ---- COLOR PROFILE ---- */}
      <section className="max-w-3xl mx-auto px-6 py-14">
        <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-1">Your Characteristics</p>
        <h2 className="text-2xl md:text-3xl font-serif mb-6">Your color profile</h2>
        <div className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-xl px-5">
          <ProfileRow label="Undertone" value={data.axes.undertone.prediction} />
          <ProfileRow label="Depth" value={data.axes.skin_depth.prediction} />
          <ProfileRow label="Contrast" value={data.axes.contrast.prediction} />
          <ProfileRow label="Chroma" value={data.axes.chroma.prediction} />
        </div>

        {/* Color Formula */}
        <div className="mt-8 flex items-center justify-center gap-3 text-sm">
          <span className="px-3 py-1.5 bg-[var(--color-surface-muted)] rounded-full font-medium">{undertoneLabel}</span>
          <span className="text-[var(--color-text-tertiary)]">+</span>
          <span className="px-3 py-1.5 bg-[var(--color-surface-muted)] rounded-full font-medium">{depthLabel}</span>
          <span className="text-[var(--color-text-tertiary)]">+</span>
          <span className="px-3 py-1.5 bg-[var(--color-surface-muted)] rounded-full font-medium">{chromaLabel}</span>
          <span className="text-[var(--color-text-tertiary)]">=</span>
          <span className="px-3 py-1.5 bg-[var(--color-accent)] text-white rounded-full font-medium text-xs">Your palette</span>
        </div>

        {/* Why this result */}
        <button onClick={() => setShowExplanation(!showExplanation)} className="mt-6 text-xs text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] transition-colors flex items-center gap-1" aria-expanded={showExplanation}>
          Why this result? <ChevronDown className={`w-3 h-3 transition-transform ${showExplanation ? "rotate-180" : ""}`} />
        </button>
        {showExplanation && (
          <div className="mt-3 p-5 bg-[var(--color-surface-muted)] rounded-xl text-sm text-[var(--color-text-secondary)] leading-relaxed space-y-2">
            <p><strong className="text-[var(--color-text-primary)]">Undertone:</strong> {data.axes.undertone.prediction} &mdash; based on the red-green ({data.features.skin_a.toFixed(1)}) and yellow-blue ({data.features.skin_b.toFixed(1)}) balance of your skin.</p>
            <p><strong className="text-[var(--color-text-primary)]">Depth:</strong> {data.axes.skin_depth.prediction} &mdash; derived from the lightness (L*={data.features.skin_L.toFixed(1)}).</p>
            <p><strong className="text-[var(--color-text-primary)]">Season:</strong> Your undertone, depth, chroma and contrast combine to map to the {season} family.</p>
          </div>
        )}
      </section>

      <Divider />

      {/* ---- BEST COLORS ---- */}
      {palette.best.length > 0 && (
        <section ref={paletteRef} className="max-w-3xl mx-auto px-6">
          <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-1">Your Palette</p>
          <h2 className="text-2xl md:text-3xl font-serif mb-2">Your best colors</h2>
          <p className="text-sm text-[var(--color-text-secondary)] mb-8">These shades harmonize naturally with your analyzed coloring.</p>
          <div className="grid grid-cols-3 md:grid-cols-3 gap-4 md:gap-5">
            {palette.best.map((c) => <div key={c.hex}><PaletteSwatch swatch={c} /></div>)}
          </div>
        </section>
      )}

      {/* ---- NEUTRALS ---- */}
      {palette.neutrals.length > 0 && (
        <>
          <Divider />
          <section className="max-w-3xl mx-auto px-6">
            <h2 className="text-xl font-serif mb-1">Your best neutrals</h2>
            <p className="text-sm text-[var(--color-text-secondary)] mb-6">Build your wardrobe around these.</p>
            <div className="space-y-3">
              {palette.neutrals.map((c) => <div key={c.hex}><SmallSwatch swatch={c} /></div>)}
            </div>
          </section>
        </>
      )}

      {/* ---- CAUTION ---- */}
      {palette.caution.length > 0 && (
        <>
          <Divider />
          <section className="max-w-3xl mx-auto px-6">
            <h2 className="text-xl font-serif mb-1">Colors to use carefully</h2>
            <p className="text-sm text-[var(--color-text-secondary)] mb-6">These shades may compete with your natural coloring when worn close to the face.</p>
            <div className="space-y-3 opacity-55">
              {palette.caution.map((c) => <div key={c.hex}><SmallSwatch swatch={c} /></div>)}
            </div>
          </section>
        </>
      )}

      <Divider />

      {/* ---- STYLE GUIDE TABS ---- */}
      {styleGuide && (
        <section className="max-w-3xl mx-auto px-6">
          {/* Tabs */}
          <div className="flex gap-1 mb-8 border-b border-[var(--color-border-light)] overflow-x-auto">
            {styleTabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button key={tab.key} onClick={() => setActiveStyleTab(tab.key)} className={`flex items-center gap-1.5 px-4 py-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${activeStyleTab === tab.key ? "border-[var(--color-text-primary)] text-[var(--color-text-primary)]" : "border-transparent text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)]"}`}>
                  <Icon className="w-3.5 h-3.5" /> {tab.label}
                </button>
              );
            })}
          </div>

          {/* Clothing */}
          {activeStyleTab === "clothing" && (
            <div>
              <h2 className="text-2xl font-serif mb-1">Your clothing palette</h2>
              <p className="text-sm text-[var(--color-text-secondary)] mb-8">Turn your color profile into an everyday wardrobe.</p>
              <div className="space-y-8">
                {styleGuide.clothing.map((cat) => (
                  <div key={cat.category}>
                    <p className="text-[10px] font-semibold tracking-[0.15em] uppercase text-[var(--color-text-tertiary)] mb-3">{cat.category}</p>
                    <div className="space-y-3">
                      {cat.items.map((item) => (
                        <div key={item.color} className="flex items-start gap-3">
                          <Check className="w-3.5 h-3.5 text-[var(--color-success)] mt-0.5 shrink-0" />
                          <div>
                            <p className="text-sm font-medium">{item.color}</p>
                            <p className="text-xs text-[var(--color-text-secondary)]">{item.note}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Makeup */}
          {activeStyleTab === "makeup" && (
            <div>
              <h2 className="text-2xl font-serif mb-1">Makeup that harmonizes</h2>
              <p className="text-sm text-[var(--color-text-secondary)] mb-8">Shades likely to complement your analyzed coloring.</p>
              <div className="space-y-8">
                {styleGuide.makeup.map((cat) => (
                  <div key={cat.category}>
                    <p className="text-[10px] font-semibold tracking-[0.15em] uppercase text-[var(--color-text-tertiary)] mb-3">{cat.category}</p>
                    <div className="flex flex-wrap gap-2 mb-2">
                      {cat.shades.map((s) => (
                        <span key={s.name} className="px-3 py-1.5 text-xs font-medium bg-[var(--color-surface-muted)] border border-[var(--color-border-light)] rounded-full text-[var(--color-text-secondary)]">{s.name}</span>
                      ))}
                    </div>
                    <p className="text-xs text-[var(--color-text-tertiary)]">{cat.note}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Jewelry */}
          {activeStyleTab === "jewelry" && (
            <div>
              <h2 className="text-2xl font-serif mb-1">Metals that complement you</h2>
              <p className="text-sm text-[var(--color-text-secondary)] mb-8">Based on your analyzed undertone.</p>
              <div className="space-y-4">
                {styleGuide.jewelry.map((j) => (
                  <div key={j.metal} className="flex items-start gap-4 p-5 bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-xl">
                    <div className={`w-10 h-10 rounded-full shrink-0 border border-[var(--color-border-light)] ${j.metal === "Gold" ? "bg-[#DAA520]" : j.metal === "Silver" || j.metal === "Platinum" ? "bg-[#C0C0C0]" : "bg-[#B76E79]"}`} />
                    <div>
                      <div className="flex items-center gap-2 mb-0.5">
                        <p className="text-sm font-semibold">{j.metal}</p>
                        <span className={`text-[10px] font-medium uppercase tracking-wider ${j.verdict === "best" ? "text-[var(--color-success)]" : j.verdict === "good" ? "text-[var(--color-text-secondary)]" : "text-[var(--color-caution)]"}`}>
                          {j.verdict === "best" ? "Strong harmony" : j.verdict === "good" ? "Good alternative" : "Use carefully"}
                        </span>
                      </div>
                      <p className="text-xs text-[var(--color-text-secondary)]">{j.note}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Hair */}
          {activeStyleTab === "hair" && (
            <div>
              <h2 className="text-2xl font-serif mb-1">Hair color direction</h2>
              <p className="text-sm text-[var(--color-text-secondary)] mb-8">Colors that harmonize with your palette.</p>
              <div className="space-y-3">
                {styleGuide.hair.map((h) => (
                  <div key={h.direction} className="flex items-start gap-3 py-3 border-b border-[var(--color-border-light)] last:border-b-0">
                    <div className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${h.verdict === "best" ? "bg-[var(--color-success)]" : h.verdict === "good" ? "bg-[var(--color-text-tertiary)]" : "bg-[var(--color-caution)]"}`} />
                    <div>
                      <p className="text-sm font-medium">{h.direction}</p>
                      <p className="text-xs text-[var(--color-text-secondary)]">{h.note}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      {/* ---- BUILD A LOOK ---- */}
      {styleGuide && styleGuide.outfit.length > 0 && (
        <>
          <Divider />
          <section className="max-w-3xl mx-auto px-6">
            <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-1">Styled for you</p>
            <h2 className="text-2xl md:text-3xl font-serif mb-6">Build a look</h2>
            <div className="grid md:grid-cols-2 gap-4">
              {styleGuide.outfit.map((outfit) => (
                <div key={outfit.label} className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-xl p-5">
                  <p className="text-[10px] font-semibold tracking-[0.15em] uppercase text-[var(--color-text-tertiary)] mb-3">{outfit.label}</p>
                  <div className="space-y-2.5">
                    {outfit.items.map((item) => (
                      <div key={item.part} className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md border border-[var(--color-border-light)] shrink-0" style={{ backgroundColor: item.hex }} />
                        <div>
                          <span className="text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-wider mr-2">{item.part}</span>
                          <span className="text-sm font-medium">{item.color}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      <Divider />

      {/* ---- COLOR CHECKER ---- */}
      <section className="max-w-3xl mx-auto px-6">
        <ColorChecker />
      </section>

      <Divider />

      {/* ---- STYLE AT A GLANCE ---- */}
      <section className="max-w-3xl mx-auto px-6 text-center">
        <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-1">Summary</p>
        <h2 className="text-2xl font-serif mb-6">Your style at a glance</h2>
        <div className="inline-flex flex-wrap justify-center gap-2 text-xs font-medium">
          <span className="px-3 py-1.5 bg-[var(--color-surface-muted)] rounded-full">{undertoneLabel} · {depthLabel} · {chromaLabel}</span>
          {palette.best.slice(0, 3).map((c) => (
            <span key={c.hex} className="px-3 py-1.5 rounded-full text-white" style={{ backgroundColor: c.hex }}>{c.name}</span>
          ))}
          {styleGuide && <span className="px-3 py-1.5 bg-[var(--color-surface-muted)] rounded-full">{styleGuide.jewelry.find(j => j.verdict === "best")?.metal || "Gold"}</span>}
        </div>
      </section>

      <Divider />

      {/* ---- CTA ---- */}
      <section className="max-w-3xl mx-auto px-6 text-center py-8">
        <p className="text-sm text-[var(--color-text-secondary)] mb-4">Want to check another photo?</p>
        <Link to="/analyze" className="inline-flex items-center gap-2 px-7 py-3.5 bg-white text-[var(--color-text-primary)] text-sm font-medium rounded-full border border-[var(--color-border)] hover:bg-[var(--color-surface-muted)] transition-colors">
          Analyze another photo
        </Link>
      </section>
    </main>
  );
}
