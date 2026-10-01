/**
 * Outfit Studio — comparison view.
 *
 * Renders deterministic per-option scores, advantages and colour conflicts
 * side by side. Never declares a universal winner — the summary states the
 * score differences and explicitly leaves the choice to the user.
 */
import { Fragment } from "react";
import { X } from "lucide-react";
import type { CompareOptionResult, CompareResponse } from "../../types/outfit";
import { SwatchRow } from "./OutfitResults";

function OptionCard({ option }: { option: CompareOptionResult }) {
  return (
    <div className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-2xl p-6">
      <div className="flex items-baseline justify-between gap-3 mb-1">
        <h3 className="text-base font-serif truncate">{option.label}</h3>
        <span className="text-2xl font-serif shrink-0">
          {option.score.score === null ? "n/a" : option.score.score}
          <span className="text-[10px] text-[var(--color-text-tertiary)] font-sans font-normal"> /100</span>
        </span>
      </div>
      <p className="text-xs text-[var(--color-text-secondary)] mb-1">{option.suitability.verdict}</p>

      {option.garments?.length > 0 && (
        <div className="mt-3 space-y-1.5">
          <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)]">
            Outfit ({option.garments.length} garment{option.garments.length === 1 ? "" : "s"})
          </p>
          {option.garments.map((g, i) => (
            <div key={g.id || i} className="flex items-center gap-2">
              <span
                className="w-5 h-5 rounded border border-[var(--color-border-light)] shrink-0"
                style={{ backgroundColor: g.dominantColour.hex }}
                aria-hidden="true"
              />
              <span className="text-xs text-[var(--color-text-secondary)] truncate">
                {g.dominantColour.name} {g.category}
              </span>
            </div>
          ))}
        </div>
      )}

      <SwatchRow swatches={option.palette} />

      {option.advantages.length > 0 && (
        <div className="mt-4">
          <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-1.5">
            Works well
          </p>
          <ul className="space-y-1">
            {option.advantages.slice(0, 4).map((a, i) => (
              <li key={i} className="text-xs text-[var(--color-text-secondary)] leading-relaxed">· {a}</li>
            ))}
          </ul>
        </div>
      )}

      {option.conflicts.length > 0 && (
        <div className="mt-3">
          <p className="text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-1.5">
            Colour tensions
          </p>
          <ul className="space-y-1">
            {option.conflicts.slice(0, 4).map((c, i) => (
              <li key={i} className="text-xs text-[var(--color-text-secondary)] leading-relaxed">· {c}</li>
            ))}
          </ul>
        </div>
      )}

      {option.limitations.length > 0 && (
        <ul className="mt-3 space-y-1">
          {option.limitations.map((l, i) => (
            <li key={i} className="text-[10px] text-[var(--color-text-tertiary)] leading-relaxed">· {l}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function ComparisonView({
  comparison,
  comparing,
  compareError,
  onClear,
}: {
  comparison: CompareResponse | null;
  comparing: boolean;
  compareError: string | null;
  onClear: () => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-serif">Compare outfit options</h2>
        <button
          type="button"
          onClick={onClear}
          className="inline-flex items-center gap-1.5 text-xs font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors"
        >
          <X className="w-3.5 h-3.5" /> Clear comparison
        </button>
      </div>
      <p className="text-xs text-[var(--color-text-tertiary)] mb-6">
        Scores come from documented palette-compatibility rules — they measure colour consistency with your
        saved palette, not which look is "best". Style and occasion are yours to weigh.
      </p>

      {comparing && (
        <p className="text-sm text-[var(--color-text-secondary)]" role="status">
          Comparing options…
        </p>
      )}
      {compareError && <p className="text-xs text-[var(--color-error)]">{compareError}</p>}

      {comparison && (
        <>
          {comparison.summary && (
            <div className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-2xl p-5 mb-6">
              <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{comparison.summary}</p>
            </div>
          )}
          <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
            {comparison.options.map((option) => (
              <Fragment key={option.label}>
                <OptionCard option={option} />
              </Fragment>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
