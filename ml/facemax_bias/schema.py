"""Row-level schema validation for the FaceMaxx annotation table."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

try:  # pandas is optional; validate plain dicts when absent
    from pandas import DataFrame  # type: ignore
except ImportError:  # pragma: no cover - exercised in CI without pandas
    DataFrame = Any  # type: ignore

from .vocab import (
    CAMERA_ORDER,
    CHROMA_ORDER,
    CONTRAST_ORDER,
    LIGHTING_ORDER,
    MAKEUP_ORDER,
    QUALITY_ORDER,
    SEASON_ORDER,
    SKIN_DEPTH_ORDER,
    SOURCE_ORDER,
    UNDERTONE_ORDER,
    normalize_lighting,
    normalize_quality,
)

REQUIRED_COLUMNS = (
    "image_id",
    "person_id",
    "season_primary",
)

OPTIONAL_NUMERIC = (
    "skin_L",
    "skin_a",
    "skin_b",
    "undertone_confidence",
    "olive_score",
    "contrast_score",
    "chroma_score",
    "season_confidence",
    "annotation_agreement",
)

_CATEGORICAL = {
    "skin_depth": SKIN_DEPTH_ORDER,
    "undertone": UNDERTONE_ORDER,
    "contrast": CONTRAST_ORDER,
    "chroma": CHROMA_ORDER,
    "image_quality": QUALITY_ORDER,
    "lighting_type": LIGHTING_ORDER,
    "camera_type": CAMERA_ORDER,
    "source": SOURCE_ORDER,
    "makeup": MAKEUP_ORDER,
    "season_primary": SEASON_ORDER,
    "season_secondary": SEASON_ORDER,
}


@dataclass
class SchemaReport:
    """Result of validating an annotation table."""

    total_rows: int = 0
    missing_required: List[str] = field(default_factory=list)
    unknown_values: Dict[str, Dict[str, int]] = field(default_factory=dict)
    person_leakage: List[str] = field(default_factory=list)
    duplicate_images: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_required and not self.duplicate_images

    def summary_lines(self) -> List[str]:
        lines = [
            f"Rows: {self.total_rows}",
            f"Missing required columns: {', '.join(self.missing_required) or 'none'}",
            f"Duplicate image_ids: {len(self.duplicate_images)}",
            f"person_id reuse across splits (leakage): {len(self.person_leakage)}",
        ]
        for column, counts in self.unknown_values.items():
            preview = ", ".join(f"{k!r} x{v}" for k, v in sorted(counts.items())[:6])
            lines.append(f"Unknown {column} values: {preview}")
        lines.extend(f"warning: {w}" for w in self.warnings)
        return lines


def _iter_records(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, DataFrame):
        for _, row in data.iterrows():
            yield {k: row[k] for k in data.columns}
    elif isinstance(data, dict):
        first = next(iter(data), None)
        if first is None:
            return
        length = len(data[first])
        for i in range(length):
            yield {col: data[col][i] for col in data}
    else:
        yield from data


def validate(data: Any, split: Optional[str] = None) -> SchemaReport:
    """Validate records, optionally restricted to one split (e.g. the test split)."""
    report = SchemaReport()
    seen_images: Dict[str, str] = {}
    seen_persons: Dict[str, str] = {}
    missing_required: set = set()

    for row in _iter_records(data):
        if split is not None and row.get("split") not in (None, split):
            continue
        report.total_rows += 1

        for column in REQUIRED_COLUMNS:
            value = row.get(column)
            if value is None or (isinstance(value, float) and value != value) or str(value).strip() == "":
                missing_required.add(column)

        image_id = str(row.get("image_id", "")).strip()
        person_id = str(row.get("person_id", "")).strip()
        row_split = str(row.get("split") or "unsplit")

        if image_id:
            if image_id in seen_images:
                report.duplicate_images.append(image_id)
            seen_images[image_id] = row_split

        if person_id and image_id:
            if person_id in seen_persons and seen_persons[person_id] != row_split:
                report.person_leakage.append(
                    f"{person_id} appears in {seen_persons[person_id]} and {row_split}"
                )
            else:
                seen_persons.setdefault(person_id, row_split)

        for column, vocabulary in _CATEGORICAL.items():
            raw = row.get(column)
            if raw is None or str(raw).strip() == "":
                continue
            if column == "lighting_type":
                normalized = normalize_lighting(str(raw))
            elif column == "image_quality":
                normalized = normalize_quality(str(raw))
            else:
                normalized = str(raw).strip()
            if normalized not in vocabulary:
                report.unknown_values.setdefault(column, {})
                report.unknown_values[column][normalized] = (
                    report.unknown_values[column].get(normalized, 0) + 1
                )

    report.missing_required = sorted(missing_required)
    report.duplicate_images = sorted(set(report.duplicate_images))
    report.person_leakage = sorted(set(report.person_leakage))

    if report.total_rows and len(report.duplicate_images) / report.total_rows > 0.01:
        report.warnings.append(
            "more than 1% duplicate image_ids; check upstream annotation pipeline"
        )
    return report
