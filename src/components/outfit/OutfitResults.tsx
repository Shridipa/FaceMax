/**
 * Outfit Studio — shared results rendering (all three input modes).
 *
 * Page-scoped component: uses only the existing FaceMaxx design tokens.
 * The suitability verdict and score are deterministic (computed server-side);
 * the AI text is already validated before it reaches here.
 */
import React, { Fragment } from "react";
import { Columns2, Check } from "lucide-react";
import type {
  OutfitAnalysisResponse, OutfitGarment, SuitabilityDimension, OutfitScore,
} from "../../types/outfit";

/** Palette-fit badge tone shared by garment rows. */
const FIT_TONE: Record<string, string> = {
  harmonious: "text-[var(--color-success)]",
  compatible: "text-[var(--color-text-primary)]",
  neutral: "text-[var(--color-text-secondary)]",
  contrasting: "text-[var(--color-caution)]",
};

export function SwatchRow({ swatches }: { swatches: { name: string; hex: string }[] }) {
  return (
    <div className="flex flex-wrap gap-2 mt-3">
      {swatches.map((s, i) => (
        <div
          key={`${s.hex}-${i}`}
          className="flex items-center gap-1.5 bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-full pl-1.5 pr-3 py-1"
        >
          <span
            className="w-5 h-5 rounded-full border border-[var(--color-border-light)]"
            style={{ backgroundColor: s.hex }}
            aria-hidden="true"
          />
          <span className="text-[10px] font-medium text-[var(--color-text-secondary)]">{s.name}</span>
        </div>
      ))}
    </div>
  );
}

function DimensionRow({ label, dimension }: { label: string; dimension: SuitabilityDimension }) {
  const tone =
    dimension.value === "harmonious" ? "text-[var(--color-success)]"
    : dimension.value === "compatible" ? "text-[var(--color-text-primary)]"
    : dimension.value === "contrasting" ? "text-[var(--color-caution)]"
    : dimension.value === "neutral" ? "text-[var(--color-text-secondary)]"
    : "text-[var(--color-text-tertiary)]";
  const dot =
    dimension.value === "harmonious" ? "bg-[var(--color-success)]"
    : dimension.value === "compatible" ? "bg-[var(--color-text-primary)]"
    : dimension.value === "contrasting" ? "bg-[var(--color-caution)]"
    : dimension.value === "neutral" ? "bg-[var(--color-text-tertiary)]"
    : "bg-[var(--color-border)]";
  return (
    <div className="flex gap-3 py-3 border-b border-[var(--color-border-light)] last:border-b-0">
      <span className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${dot}`} aria-hidden="true" />
      <div>
        <p className="text-xs font-semibold">
          {label}{" "}
          <span className={`text-[10px] uppercase tracking-wide font-medium ${tone}`}>
            {dimension.value}
          </span>
        </p>
        <p className="text-xs text-[var(--color-text-secondary)] mt-0.5 leading-relaxed">
          {dimension.reason}
        </p>
      </div>
    </div>
  );
}

function GarmentRow({
  garment,
  analysis,
  index,
}: {
  garment: OutfitGarment;
  analysis: OutfitAnalysisResponse["colour_analysis"];
  index: number;
}) {
  const perGarment = analysis?.garmentColours?.[index];
  return (
    <div className="flex items-center gap-3 py-3 border-b border-[var(--color-border-light)] last:border-b-0">
      <span
        className="w-9 h-9 rounded-lg border border-[var(--color-border-light)] shrink-0"
        style={{ backgroundColor: garment.dominantColour.hex }}
        aria-hidden="true"
      />
      <div>
        <p className="text-sm font-medium">
          {garment.category} · {garment.dominantColour.name}
          {perGarment?.paletteFit && (
            <span className={`ml-2 text-[10px] uppercase tracking-wide font-medium ${FIT_TONE[perGarment.paletteFit] || ""}`}>
              palette: {perGarment.paletteFit}
            </span>
          )}
        </p>
        <p className="text-[10px] text-[var(--color-text-tertiary)] font-mono">
          {garment.dominantColour.hex}
          {perGarment?.deltaEToNearestPaletteColour != null ? ` · ΔE ${perGarment.deltaEToNearestPaletteColour}` : ""}
          {garment.secondaryColour ? ` · accent ${garment.secondaryColour.name}` : ""}
          {garment.pattern !== "Solid" ? ` · ${garment.pattern}` : ""}
          {garment.confidence < 1 ? ` · confidence ${Math.round(garment.confidence * 100)}%` : " · from your input"}
        </p>
      </div>
    </div>
  );
}

function CombinationCard({ combo }: { combo: OutfitAnalysisResponse["outfit_combinations"][number] }) {
  return (
    <div className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-2xl p-6 shadow-[0_1px_4px_rgba(0,0,0,0.06)]">
      <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-1">
        {combo.aesthetic ? `${combo.aesthetic} · ` : ""}Anchor: {combo.anchorGarment}
      </p>
      <h3 className="text-lg font-serif mb-1">{combo.name}</h3>
      <p className="text-sm text-[var(--color-text-secondary)] mb-4">{combo.description}</p>
      <SwatchRow swatches={combo.palette} />
      <div className="mt-4 space-y-1.5">
        {combo.pieces.map((p, i) => (
          <p key={i} className="text-sm text-[var(--color-text-secondary)]">
            · {p.item} — {p.colourName}
            {p.owned === true && (
              <span className="ml-2 text-[10px] font-medium text-[var(--color-success)] uppercase tracking-wide">
                yours
              </span>
            )}
            {p.owned === false && (
              <span className="ml-2 text-[10px] font-medium text-[var(--color-text-tertiary)] uppercase tracking-wide">
                suggested · optional
              </span>
            )}
          </p>
        ))}
        {combo.footwear && (
          <p className="text-sm text-[var(--color-text-secondary)]">· Footwear: {combo.footwear}</p>
        )}
        {combo.accessories.length > 0 && (
          <p className="text-sm text-[var(--color-text-secondary)]">
            · Accessories: {combo.accessories.join(", ")}
          </p>
        )}
      </div>
      <p className="text-sm mt-4">
        <span className="font-medium">Why it works:</span> {combo.rationale}
      </p>
    </div>
  );
}

function ScoreBlock({ score }: { score: OutfitScore }) {
  return (
    <div>
      <div className="flex items-baseline gap-2">
        <span className="text-4xl font-serif">
          {score.score === null ? "n/a" : score.score}
        </span>
        <span className="text-xs text-[var(--color-text-tertiary)]">/ 100 palette compatibility</span>
      </div>
      <div className="mt-4 space-y-2.5">
        {score.breakdown.map((b) => (
          <div key={b.label}>
            <div className="flex items-center justify-between text-[11px] mb-1">
              <span className="text-[var(--color-text-secondary)]">
                {b.label} <span className="text-[var(--color-text-tertiary)]">· weight {Math.round(b.weight * 100)}%</span>
              </span>
              <span
                className={`font-medium ${
                  b.percent === null ? "text-[var(--color-text-tertiary)]"
                  : b.percent >= 80 ? "text-[var(--color-success)]"
                  : b.percent >= 55 ? "text-[var(--color-text-primary)]"
                  : "text-[var(--color-caution)]"
                }`}
              >
                {b.value === "unknown" ? "not measurable" : `${b.percent}%`}
              </span>
            </div>
            <div className="h-1.5 rounded-full bg-[var(--color-surface-muted)] overflow-hidden">
              <div
                className={`h-full rounded-full ${
                  b.percent === null ? "bg-[var(--color-border)]"
                  : b.percent >= 80 ? "bg-[var(--color-success)]"
                  : b.percent >= 55 ? "bg-[var(--color-text-primary)]"
                  : "bg-[var(--color-caution)]"
                }`}
                style={{ width: `${b.percent ?? 0}%` }}
                aria-hidden="true"
              />
            </div>
          </div>
        ))}
      </div>
      <p className="text-[10px] text-[var(--color-text-tertiary)] leading-relaxed mt-4">{score.note}</p>
    </div>
  );
}

/* ------------------------------------------------------------------ */

export function OutfitResults({
  result,
  photoPreview,
  onAddToComparison,
  addedToComparison,
}: {
  result: OutfitAnalysisResponse;
  photoPreview?: string | null;
  onAddToComparison?: () => void;
  addedToComparison?: boolean;
}) {
  const suitability = result.suitability;
  return (
    <div className="space-y-10">
      <div className="flex flex-col md:flex-row items-start gap-6">
        {photoPreview && (
          <img
            src={photoPreview}
            alt="Analysed outfit"
            className="w-full md:w-56 rounded-xl border border-[var(--color-border-light)]"
          />
        )}
        <div className="flex-1 w-full">
          <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-2">
            Suitability
          </p>
          <h2 className="text-2xl font-serif mb-2">{suitability?.verdict}</h2>
          {result.colour_analysis && (
            <p className="text-xs text-[var(--color-text-secondary)]">
              Outfit contrast (L*): {result.colour_analysis.outfitContrast ?? "n/a"}
              {result.colour_analysis.paletteMatches.length > 0 && (
                <>
                  {" "}· closest palette match: {result.colour_analysis.paletteMatches[0].swatch.name}
                </>
              )}
            </p>
          )}
          {result.garments.length > 0 && (
            <div className="mt-4">
              <SwatchRow swatches={result.garments.map((g) => g.dominantColour)} />
            </div>
          )}
          <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mt-4 mb-1">
            Your analysed outfit ({result.garments.length} garment{result.garments.length === 1 ? "" : "s"})
          </p>
          <div className="mt-1">
            {result.garments.map((g, i) => (
              <Fragment key={g.id || `${g.category}-${g.dominantColour.hex}-${i}`}>
                <GarmentRow garment={g} analysis={result.colour_analysis} index={i} />
              </Fragment>
            ))}
          </div>
          {result.rejectedGarments && result.rejectedGarments.length > 0 && (
            <ul className="mt-3 space-y-1">
              {result.rejectedGarments.slice(0, 4).map((r, i) => (
                <li key={i} className="text-[11px] text-[var(--color-caution)] leading-relaxed">
                  · {r.category}: {r.reason}
                </li>
              ))}
            </ul>
          )}
          {result.limitations?.length > 0 && (
            <ul className="mt-4 space-y-1">
              {result.limitations.slice(0, 5).map((l, i) => (
                <li key={i} className="text-[11px] text-[var(--color-text-tertiary)] leading-relaxed">
                  · {l}
                </li>
              ))}
            </ul>
          )}
          {onAddToComparison && (
            <button
              onClick={onAddToComparison}
              disabled={addedToComparison}
              className="mt-5 inline-flex items-center gap-2 px-5 py-2.5 border border-[var(--color-border)] text-xs font-medium rounded-full hover:bg-[var(--color-surface-muted)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {addedToComparison ? (
                <>
                  <Check className="w-3.5 h-3.5" /> Added to comparison
                </>
              ) : (
                <>
                  <Columns2 className="w-3.5 h-3.5" /> Add to comparison
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {suitability && (
        <section>
          <h3 className="text-xl font-serif mb-4">Why this verdict</h3>
          <div className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-2xl p-6">
            <DimensionRow label="Colour harmony" dimension={suitability.dimensions.colourHarmony} />
            <DimensionRow label="Temperature" dimension={suitability.dimensions.temperature} />
            <DimensionRow label="Depth" dimension={suitability.dimensions.depth} />
            <DimensionRow label="Chroma" dimension={suitability.dimensions.chroma} />
            <DimensionRow label="Contrast" dimension={suitability.dimensions.contrast} />
            <DimensionRow label="Colour combination" dimension={suitability.dimensions.colourCombination} />
          </div>
        </section>
      )}

      {result.score && (
        <section>
          <h3 className="text-xl font-serif mb-4">Compatibility score</h3>
          <div className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-2xl p-6">
            <ScoreBlock score={result.score} />
          </div>
        </section>
      )}

      {result.styling_advice && (
        <section>
          <h3 className="text-xl font-serif mb-4">How to style it</h3>
          <div className="grid md:grid-cols-2 gap-6">
            {(
              [
                ["Current outfit", result.styling_advice.currentOutfit],
                ["Footwear", result.styling_advice.footwear],
                ["Accessories", result.styling_advice.accessories],
                ["Layering", result.styling_advice.layering],
              ] as [string, string[]][]
            )
              .filter(([, items]) => items.length > 0)
              .map(([label, items]) => (
                <div
                  key={label}
                  className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-2xl p-6"
                >
                  <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-3">
                    {label}
                  </p>
                  <ul className="space-y-2">
                    {items.map((s, i) => (
                      <li key={i} className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
                        · {s}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
          </div>
        </section>
      )}

      {result.outfit_combinations.length > 0 && (
        <section>
          <h3 className="text-xl font-serif mb-4">Ways to style it</h3>
          <div className="grid md:grid-cols-2 gap-6">
            {result.outfit_combinations.map((combo) => (
              <Fragment key={combo.name}>
                <CombinationCard combo={combo} />
              </Fragment>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
