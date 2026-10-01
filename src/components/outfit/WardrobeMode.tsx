/**
 * Outfit Studio — Mode 3: My Wardrobe.
 *
 * A lightweight digital wardrobe persisted in this browser's localStorage
 * (no account system exists in the app, so no server-side user records are
 * invented). Items are added/edited/removed locally, then combined into an
 * outfit and analysed against the saved colour profile.
 */
import { useEffect, useMemo, useState } from "react";
import { Plus, Search, Trash2, Pencil, Shirt } from "lucide-react";
import type { GarmentInput } from "../../types/outfit";
import { loadWardrobe, saveWardrobe, type WardrobeItem } from "../../services/outfit";

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

const CATEGORIES = ["Tops", "Bottoms", "Dresses", "Outerwear", "Footwear", "Accessories"];

export function WardrobeMode({
  onAnalyse,
  analyzing,
}: {
  onAnalyse: (garments: GarmentInput[], occasion: string, stylePreference: string) => void;
  analyzing: boolean;
}) {
  const [items, setItems] = useState<WardrobeItem[]>(() => loadWardrobe());
  const [filter, setFilter] = useState("");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [occasion, setOccasion] = useState("");
  const [stylePreference, setStylePreference] = useState("");
  const [adding, setAdding] = useState(false);
  const [editId, setEditId] = useState<string | null>(null);
  const [form, setForm] = useState({ category: "", colourName: "Black", hex: "#000000", pattern: "", fit: "" });
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    saveWardrobe(items);
  }, [items]);

  const visible = useMemo(
    () =>
      items.filter(
        (i) =>
          (!filter || i.category === filter) &&
          (!query ||
            i.category.toLowerCase().includes(query.toLowerCase()) ||
            i.colourName.toLowerCase().includes(query.toLowerCase()) ||
            (i.pattern || "").toLowerCase().includes(query.toLowerCase())),
      ),
    [items, filter, query],
  );

  const openAdd = () => {
    setAdding(true);
    setEditId(null);
    setForm({ category: "", colourName: "Black", hex: "#000000", pattern: "", fit: "" });
  };
  const openEdit = (item: WardrobeItem) => {
    setAdding(true);
    setEditId(item.id);
    setForm({
      category: item.category,
      colourName: item.colourName,
      hex: item.hex,
      pattern: item.pattern || "",
      fit: item.fit || "",
    });
  };
  const submitForm = () => {
    if (!form.category.trim()) return;
    const cleaned = {
      category: form.category.trim().slice(0, 60),
      colourName: form.colourName.trim().slice(0, 40) || "Unrecognised",
      hex: /^#[0-9a-fA-F]{6}$/.test(form.hex) ? form.hex.toUpperCase() : "#000000",
      pattern: form.pattern.trim() || null,
      fit: form.fit.trim() || null,
    };
    setItems((prev) =>
      editId
        ? prev.map((i) => (i.id === editId ? { ...i, ...cleaned } : i))
        : [...prev, { id: `w-${Date.now()}-${Math.floor(Math.random() * 1e6)}`, ...cleaned }],
    );
    setAdding(false);
    setEditId(null);
  };
  const removeItem = (id: string) => {
    setItems((prev) => prev.filter((i) => i.id !== id));
    setSelected((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  };
  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectedGarments: GarmentInput[] = items
    .filter((i) => selected.has(i.id))
    .map((i) => ({
      id: i.id,
      category: i.category,
      colourName: i.colourName,
      hex: i.hex,
      pattern: i.pattern,
      fit: i.fit,
      owned: true,
    }));

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-[180px]">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-tertiary)]" aria-hidden="true" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search your wardrobe"
            aria-label="Search wardrobe"
            className="w-full text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg pl-9 pr-3 py-2.5 placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:border-[var(--color-accent)] transition-colors"
          />
        </div>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Filter by category"
          className="text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg px-3 py-2.5 focus:outline-none focus:border-[var(--color-accent)] transition-colors"
        >
          <option value="">All categories</option>
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <button
          type="button"
          onClick={openAdd}
          className="inline-flex items-center gap-2 px-4 py-2.5 bg-white border border-[var(--color-border)] text-sm font-medium rounded-lg hover:bg-[var(--color-surface-muted)] transition-colors"
        >
          <Plus className="w-4 h-4" /> Add item
        </button>
      </div>

      {notice && (
        <p className="text-[11px] text-[var(--color-text-tertiary)]">{notice}</p>
      )}
      <p className="text-[11px] text-[var(--color-text-tertiary)] -mt-3">
        Saved in this browser only (localStorage) — there is no account system in this app, so items are not stored server-side.
      </p>

      {adding && (
        <div className="bg-[var(--color-surface)] border border-[var(--color-border-light)] rounded-2xl p-5">
          <p className="text-sm font-medium mb-4">{editId ? "Edit item" : "Add a garment"}</p>
          <div className="grid sm:grid-cols-2 gap-4">
            <label className="block">
              <span className="block text-[10px] uppercase tracking-[0.15em] text-[var(--color-text-tertiary)] mb-1.5">Category</span>
              <select
                value={form.category}
                onChange={(e) => setForm((f) => ({ ...f, category: e.target.value }))}
                className="w-full text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg px-3 py-2.5 focus:outline-none focus:border-[var(--color-accent)]"
              >
                <option value="">Select…</option>
                {CATEGORIES.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
                <option value="Other">Other</option>
              </select>
            </label>
            <label className="block">
              <span className="block text-[10px] uppercase tracking-[0.15em] text-[var(--color-text-tertiary)] mb-1.5">Colour</span>
              <div className="flex gap-2">
                <select
                  value={COLOUR_SWATCHES.some((c) => c.hex === form.hex) ? form.hex : ""}
                  onChange={(e) => {
                    const c = COLOUR_SWATCHES.find((x) => x.hex === e.target.value);
                    if (c) setForm((f) => ({ ...f, hex: c.hex, colourName: c.name }));
                  }}
                  className="flex-1 text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg px-3 py-2.5 focus:outline-none focus:border-[var(--color-accent)]"
                >
                  <option value="">Custom…</option>
                  {COLOUR_SWATCHES.map((c) => (
                    <option key={c.hex} value={c.hex}>{c.name}</option>
                  ))}
                </select>
                <input
                  type="color"
                  value={/^#[0-9a-fA-F]{6}$/.test(form.hex) ? form.hex : "#000000"}
                  onChange={(e) => setForm((f) => ({ ...f, hex: e.target.value.toUpperCase(), colourName: "Custom" }))}
                  className="w-12 h-11 cursor-pointer bg-transparent border border-[var(--color-border)] rounded-lg px-1"
                  aria-label="Pick custom colour"
                />
              </div>
            </label>
            <label className="block">
              <span className="block text-[10px] uppercase tracking-[0.15em] text-[var(--color-text-tertiary)] mb-1.5">Colour name</span>
              <input
                value={form.colourName}
                onChange={(e) => setForm((f) => ({ ...f, colourName: e.target.value }))}
                className="w-full text-sm bg-transparent border border-[var(--color-border)] rounded-lg px-3 py-2.5 focus:outline-none focus:border-[var(--color-accent)]"
              />
            </label>
            <label className="block">
              <span className="block text-[10px] uppercase tracking-[0.15em] text-[var(--color-text-tertiary)] mb-1.5">Pattern (optional)</span>
              <select
                value={form.pattern}
                onChange={(e) => setForm((f) => ({ ...f, pattern: e.target.value }))}
                className="w-full text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg px-3 py-2.5 focus:outline-none focus:border-[var(--color-accent)]"
              >
                {["", "Solid", "Striped", "Checked", "Floral", "Geometric", "Printed"].map((p) => (
                  <option key={p} value={p}>{p || "Not specified"}</option>
                ))}
              </select>
            </label>
            <label className="block sm:col-span-2">
              <span className="block text-[10px] uppercase tracking-[0.15em] text-[var(--color-text-tertiary)] mb-1.5">Fit (optional)</span>
              <input
                value={form.fit}
                onChange={(e) => setForm((f) => ({ ...f, fit: e.target.value }))}
                placeholder="e.g. oversized, straight, tailored"
                className="w-full text-sm bg-transparent border border-[var(--color-border)] rounded-lg px-3 py-2.5 placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:border-[var(--color-accent)]"
              />
            </label>
          </div>
          <div className="flex gap-3 mt-5">
            <button
              type="button"
              onClick={submitForm}
              disabled={!form.category.trim()}
              className="px-6 py-2.5 bg-[var(--color-accent)] text-white text-sm font-medium rounded-lg hover:bg-[var(--color-text-primary)] transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {editId ? "Save changes" : "Add to wardrobe"}
            </button>
            <button
              type="button"
              onClick={() => { setAdding(false); setEditId(null); }}
              className="px-6 py-2.5 border border-[var(--color-border)] text-sm font-medium rounded-lg hover:bg-[var(--color-surface-muted)] transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {items.length === 0 ? (
        <div className="border-2 border-dashed border-[var(--color-border)] rounded-2xl px-6 py-12 text-center bg-[var(--color-surface)]">
          <Shirt className="w-6 h-6 text-[var(--color-text-tertiary)] mx-auto mb-3" />
          <p className="text-sm font-medium mb-1">Your wardrobe is empty</p>
          <p className="text-xs text-[var(--color-text-secondary)] mb-5">
            Add the pieces you own — category and a colour are enough to start.
          </p>
          <button
            type="button"
            onClick={openAdd}
            className="inline-flex items-center gap-2 px-6 py-3 bg-white border border-[var(--color-border)] text-sm font-medium rounded-lg hover:bg-[var(--color-surface-muted)] transition-colors"
          >
            <Plus className="w-4 h-4" /> Add your first item
          </button>
        </div>
      ) : (
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {visible.map((item) => {
            const isSelected = selected.has(item.id);
            return (
              <div
                key={item.id}
                className={`bg-[var(--color-surface)] border rounded-2xl p-4 transition-colors ${
                  isSelected
                    ? "border-[var(--color-accent)] ring-1 ring-[var(--color-accent)]"
                    : "border-[var(--color-border-light)]"
                }`}
              >
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => toggleSelect(item.id)}
                    aria-label={`Select ${item.colourName} ${item.category}`}
                    className="mt-1 w-4 h-4 accent-[var(--color-accent)] cursor-pointer"
                  />
                  <span
                    className="w-10 h-10 rounded-lg border border-[var(--color-border-light)] shrink-0"
                    style={{ backgroundColor: item.hex }}
                    aria-hidden="true"
                  />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium truncate">
                      {item.colourName} {item.category}
                    </p>
                    <p className="text-[10px] text-[var(--color-text-tertiary)] font-mono truncate">
                      {item.hex}
                      {item.pattern && item.pattern !== "Solid" ? ` · ${item.pattern}` : ""}
                      {item.fit ? ` · ${item.fit}` : ""}
                    </p>
                  </div>
                  <div className="flex gap-1 shrink-0">
                    <button
                      type="button"
                      onClick={() => openEdit(item)}
                      aria-label={`Edit ${item.colourName} ${item.category}`}
                      className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] transition-colors p-1"
                    >
                      <Pencil className="w-3.5 h-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => removeItem(item.id)}
                      aria-label={`Remove ${item.colourName} ${item.category}`}
                      className="text-[var(--color-text-tertiary)] hover:text-[var(--color-error)] transition-colors p-1"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {items.length > 0 && (
        <div className="space-y-4">
          <p className="text-xs text-[var(--color-text-secondary)]">
            {selected.size} selected — tick 2+ items (a top and a bottom, or a dress) to analyse an outfit. You can also analyse a single hero piece.
          </p>
          {selectedGarments.length > 0 && (
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
          <button
            type="button"
            onClick={() => onAnalyse(selectedGarments, occasion, stylePreference)}
            disabled={analyzing || selectedGarments.length === 0}
            className="inline-flex items-center gap-2 px-7 py-3.5 bg-[var(--color-accent)] text-white text-sm font-medium rounded-lg hover:bg-[var(--color-text-primary)] transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {analyzing ? "Analysing…" : "Analyse Selected Outfit"}
          </button>
        </div>
      )}
    </div>
  );
}
