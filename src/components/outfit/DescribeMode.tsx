/**
 * Outfit Studio — Mode 2: describe clothes in natural language.
 *
 * The backend (Gemini) extracts structured garments from the description;
 * the user can correct every field before generating recommendations.
 * Colours supplied/confirmed by the user are treated as ground truth, so
 * the later suitability verdict is deterministic.
 */
import { useState } from "react";
import { Plus, Trash2, Wand2 } from "lucide-react";
import type { GarmentInput } from "../../types/outfit";

const EXAMPLES = [
  "I have a black oversized shirt and beige wide-leg trousers.",
  "I own a dark blue denim jacket, white T-shirt and black jeans.",
  "I have a maroon kurti with gold embroidery.",
  "I own a cream sweater, brown trousers and tan boots.",
];

const COLOUR_SWATCHES: { name: string; hex: string }[] = [
  { name: "Black", hex: "#000000" }, { name: "White", hex: "#FFFFFF" },
  { name: "Grey", hex: "#808080" }, { name: "Navy", hex: "#1A1A3E" },
  { name: "Cobalt Blue", hex: "#0047AB" }, { name: "Sky Blue", hex: "#87CEEB" },
  { name: "Teal", hex: "#008080" }, { name: "Forest Green", hex: "#234B35" },
  { name: "Olive", hex: "#6B8E23" }, { name: "Maroon", hex: "#6E1F2A" },
  { name: "Red", hex: "#FF0000" }, { name: "Orange", hex: "#FFA500" },
  { name: "Marigold", hex: "#EAA221" }, { name: "Camel", hex: "#C19A6B" },
  { name: "Brown", hex: "#8B6914" }, { name: "Tan", hex: "#D2B48C" },
  { name: "Beige", hex: "#F5F5DC" }, { name: "Cream", hex: "#FFFDD0" },
  { name: "Light Pink", hex: "#FFB6C1" }, { name: "Mauve", hex: "#8B668B" },
  { name: "Indigo", hex: "#4B0082" }, { name: "Burgundy", hex: "#6E1F2A" },
];

function emptyGarment(): GarmentInput {
  return { id: `g-${Date.now()}-${Math.floor(Math.random() * 1e6)}`, category: "", colourName: "", hex: "#000000", pattern: null, fit: null, owned: true };
}

export function DescribeMode({
  profile,
  onAnalyse,
  analyzing,
}: {
  profile: NonNullable<import("../../types/outfit").OutfitAnalysisRequest["profile"]>;
  onAnalyse: (garments: GarmentInput[], occasion: string, stylePreference: string) => void;
  analyzing: boolean;
}) {
  const [description, setDescription] = useState("");
  const [extracting, setExtracting] = useState(false);
  const [extractError, setExtractError] = useState<string | null>(null);
  const [extracted, setExtracted] = useState<GarmentInput[] | null>(null);
  const [assumptions, setAssumptions] = useState<string[]>([]);
  const [occasion, setOccasion] = useState("");
  const [stylePreference, setStylePreference] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);

  const runExtraction = async () => {
    if (description.trim().length < 3) {
      setExtractError("Describe at least one garment first.");
      return;
    }
    setExtracting(true);
    setExtractError(null);
    try {
      const { describeClothes } = await import("../../services/outfit");
      const res = await describeClothes(description.trim(), profile);
      if (!res.success) {
        // show the server's honest reason (e.g. missing API key, no colour found)
        const reason = res.limitations?.length ? res.limitations.join(" ") : res.message;
        setExtractError(reason || "Could not interpret the description. Please try again.");
        setExtracted(null);
      } else {
        setExtracted(res.garments);
        setAssumptions(res.assumptions);
        setExtractError(null);
      }
    } catch (err) {
      setExtractError((err as Error).message || "Could not reach the styling service. Please try again.");
    } finally {
      setExtracting(false);
    }
  };

  const patchGarment = (id: string, patch: Partial<GarmentInput>) => {
    setExtracted((prev) => (prev || []).map((g) => (g.id === id ? { ...g, ...patch } : g)));
  };

  const validExtracted = (extracted || []).filter((g) => g.category.trim() && /^#[0-9a-fA-F]{6}$/.test(g.hex));

  return (
    <div className="space-y-6">
      <div>
        <label htmlFor="outfit-description" className="block text-sm font-medium mb-2">
          Describe the clothes you own
        </label>
        <textarea
          id="outfit-description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={3}
          maxLength={2000}
          placeholder="Tell me what you have in your wardrobe. Include colours, clothing types, patterns or fabrics if you know them."
          className="w-full text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded-xl px-4 py-3 text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:border-[var(--color-accent)] transition-colors resize-y"
        />
        <div className="flex flex-wrap gap-2 mt-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => setDescription(ex)}
              className="text-[11px] px-3 py-1.5 rounded-full border border-[var(--color-border-light)] text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-muted)] transition-colors"
            >
              e.g. {ex.length > 42 ? ex.slice(0, 42) + "…" : ex}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={runExtraction}
          disabled={extracting || analyzing}
          className="mt-4 inline-flex items-center gap-2 px-6 py-3 bg-[var(--color-accent)] text-white text-sm font-medium rounded-lg hover:bg-[var(--color-text-primary)] transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
        >
          <Wand2 className="w-4 h-4" /> {extracting ? "Reading your description…" : "Extract my garments"}
        </button>
        {extractError && <p className="text-xs text-[var(--color-error)] mt-3">{extractError}</p>}
        {extracting && (
          <p className="text-xs text-[var(--color-text-secondary)] mt-3" role="status">
            Interpreting colours and garment types…
          </p>
        )}
      </div>

      {extracted && extracted.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium">
              Check the extracted items ({validExtracted.length} ready)
            </p>
            <button
              type="button"
              onClick={() => setExtracted((prev) => [...(prev || []), emptyGarment()])}
              className="inline-flex items-center gap-1.5 text-xs font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors"
            >
              <Plus className="w-3.5 h-3.5" /> Add garment
            </button>
          </div>
          <p className="text-[11px] text-[var(--color-text-tertiary)]">
            The AI may misread a colour or category — correct anything before generating outfits.
          </p>

          {extracted.map((g) => (
            <div
              key={g.id}
              className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-xl p-4"
            >
              <div className="flex items-center gap-3">
                <span
                  className="w-9 h-9 rounded-lg border border-[var(--color-border-light)] shrink-0"
                  style={{ backgroundColor: g.hex }}
                  aria-hidden="true"
                />
                <input
                  value={g.category}
                  onChange={(e) => patchGarment(g.id, { category: e.target.value })}
                  placeholder="Garment (e.g. Shirt)"
                  aria-label="Garment category"
                  className="flex-1 min-w-0 text-sm bg-transparent border-b border-[var(--color-border-light)] py-1 focus:outline-none focus:border-[var(--color-accent)]"
                />
                <select
                  value={COLOUR_SWATCHES.some((c) => c.hex === g.hex) ? g.hex : ""}
                  onChange={(e) => {
                    const c = COLOUR_SWATCHES.find((x) => x.hex === e.target.value);
                    if (c) patchGarment(g.id, { hex: c.hex, colourName: c.name });
                  }}
                  aria-label="Garment colour"
                  className="text-xs bg-transparent border border-[var(--color-border-light)] rounded-lg px-2 py-1.5 focus:outline-none focus:border-[var(--color-accent)]"
                >
                  <option value="">Colour…</option>
                  {COLOUR_SWATCHES.map((c) => (
                    <option key={c.hex} value={c.hex}>{c.name}</option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => setEditingId(editingId === g.id ? null : g.id)}
                  className="text-xs font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors shrink-0"
                  aria-expanded={editingId === g.id}
                >
                  {editingId === g.id ? "Hide" : "More"}
                </button>
                <button
                  type="button"
                  onClick={() => setExtracted((prev) => (prev || []).filter((x) => x.id !== g.id))}
                  aria-label={`Remove ${g.category || "garment"}`}
                  className="text-[var(--color-text-tertiary)] hover:text-[var(--color-error)] transition-colors shrink-0"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
              {editingId === g.id && (
                <div className="mt-3 grid sm:grid-cols-2 gap-3 pl-12">
                  <label className="block">
                    <span className="block text-[10px] uppercase tracking-[0.15em] text-[var(--color-text-tertiary)] mb-1">Colour name</span>
                    <input
                      value={g.colourName}
                      onChange={(e) => patchGarment(g.id, { colourName: e.target.value })}
                      className="w-full text-xs bg-transparent border border-[var(--color-border-light)] rounded-lg px-3 py-2 focus:outline-none focus:border-[var(--color-accent)]"
                    />
                  </label>
                  <label className="block">
                    <span className="block text-[10px] uppercase tracking-[0.15em] text-[var(--color-text-tertiary)] mb-1">Hex</span>
                    <input
                      type="color"
                      value={/^#[0-9a-fA-F]{6}$/.test(g.hex) ? g.hex : "#000000"}
                      onChange={(e) => patchGarment(g.id, { hex: e.target.value.toUpperCase() })}
                      className="w-full h-9 cursor-pointer bg-transparent border border-[var(--color-border-light)] rounded-lg px-1"
                      aria-label="Pick exact colour"
                    />
                  </label>
                  <label className="block">
                    <span className="block text-[10px] uppercase tracking-[0.15em] text-[var(--color-text-tertiary)] mb-1">Pattern</span>
                    <select
                      value={g.pattern || ""}
                      onChange={(e) => patchGarment(g.id, { pattern: e.target.value || null })}
                      className="w-full text-xs bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-lg px-3 py-2 focus:outline-none focus:border-[var(--color-accent)]"
                    >
                      {["", "Solid", "Striped", "Checked", "Floral", "Geometric", "Printed"].map((p) => (
                        <option key={p} value={p}>{p || "Not specified"}</option>
                      ))}
                    </select>
                  </label>
                  <label className="block">
                    <span className="block text-[10px] uppercase tracking-[0.15em] text-[var(--color-text-tertiary)] mb-1">Fit</span>
                    <input
                      value={g.fit || ""}
                      onChange={(e) => patchGarment(g.id, { fit: e.target.value || null })}
                      placeholder="e.g. oversized, straight"
                      className="w-full text-xs bg-transparent border border-[var(--color-border-light)] rounded-lg px-3 py-2 focus:outline-none focus:border-[var(--color-accent)]"
                    />
                  </label>
                </div>
              )}
            </div>
          ))}

          {assumptions.length > 0 && (
            <ul className="text-[11px] text-[var(--color-text-tertiary)] space-y-1">
              {assumptions.map((a, i) => (
                <li key={i}>· Assumed: {a}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {validExtracted.length > 0 && (
        <div className="grid sm:grid-cols-2 gap-4">
          <label className="block">
            <span className="block text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-2">
              Occasion <span className="normal-case tracking-normal">(optional)</span>
            </span>
            <select
              value={occasion}
              onChange={(e) => setOccasion(e.target.value)}
              className="w-full text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg px-4 py-3 focus:outline-none focus:border-[var(--color-accent)] transition-colors"
            >
              <option value="">Any occasion</option>
              {["Everyday", "College", "Office", "Formal event", "Party", "Date", "Wedding", "Travel", "Traditional"].map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="block text-[10px] font-medium tracking-[0.2em] uppercase text-[var(--color-text-tertiary)] mb-2">
              Style preference <span className="normal-case tracking-normal">(optional)</span>
            </span>
            <input
              type="text"
              value={stylePreference}
              maxLength={80}
              placeholder="e.g. minimal, tonal, bold"
              onChange={(e) => setStylePreference(e.target.value)}
              className="w-full text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg px-4 py-3 placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:border-[var(--color-accent)] transition-colors"
            />
          </label>
        </div>
      )}

      {validExtracted.length > 0 && (
        <button
          type="button"
          onClick={() => onAnalyse(validExtracted, occasion, stylePreference)}
          disabled={analyzing}
          className="inline-flex items-center gap-2 px-7 py-3.5 bg-[var(--color-accent)] text-white text-sm font-medium rounded-lg hover:bg-[var(--color-text-primary)] transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
        >
          <Wand2 className="w-4 h-4" /> {analyzing ? "Generating…" : "Generate My Outfit Ideas"}
        </button>
      )}
    </div>
  );
}
