"""Per-row annotation linting for real data entering the pipeline.

Complements the frozen evaluator's dataset-level schema.validate() with a
ROW-level lint: for every row, which required fields are empty, which values
fall outside the controlled vocabularies (with suggestions), which numeric
fields are out of range. Emits the report as text and as machine-readable
JSON (for the eventual annotation UI).

Controlled vocabularies come from the frozen evaluator's vocab (single source
of truth). Free-form fields (notes, best_colors, ...) are format-checked only.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from facemax_bias.vocab import (
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

from .dataset import HARMONY_COLOURS

# Columns that MUST be non-empty for a row to enter training/evaluation.
REQUIRED = ("image_id", "person_id", "season_primary", "undertone", "skin_depth")

# Numeric fields with acceptable ranges.
NUMERIC_RANGES = {
    "skin_L": (0.0, 100.0),
    "skin_a": (-128.0, 127.0),  # LAB a* theoretical range
    "skin_b": (-128.0, 127.0),
    "undertone_confidence": (0.0, 1.0),
    "olive_score": (0.0, 1.0),
    "contrast_score": (0.0, 1.0),
    "chroma_score": (0.0, 1.0),
    "season_confidence": (0.0, 1.0),
    "annotation_agreement": (0.0, 1.0),
}

# Semicolon-separated multi-value fields (values must be draping colour names
# from the harmony vocabulary - these are colour-compatibility annotations).
MULTI_COLOUR = ("best_colors", "neutral_colors", "caution_colors")

# Free-form fields (format-checked only, no vocabulary).
FREE_FORM = ("notes", "license", "image_path")

VOCAB_MAP = {
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
class RowIssues:
    image_id: str
    row_number: int
    missing_required: List[str] = field(default_factory=list)
    unknown_values: List[str] = field(default_factory=list)  # "column: value (did you mean X?)"
    out_of_range: List[str] = field(default_factory=list)    # "column: value not in [lo, hi]"

    @property
    def ok(self) -> bool:
        return not (self.missing_required or self.unknown_values or self.out_of_range)


@dataclass
class LintReport:
    total_rows: int = 0
    valid_rows: int = 0
    issues: List[RowIssues] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.valid_rows == self.total_rows

    def to_json(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "ok": self.ok,
            "issues": [
                {
                    "row_number": i.row_number,
                    "image_id": i.image_id,
                    "missing_required": i.missing_required,
                    "unknown_values": i.unknown_values,
                    "out_of_range": i.out_of_range,
                }
                for i in self.issues
            ],
        }

    def render(self, max_rows: int = 30) -> str:
        lines = [
            "ANNOTATION ROW LINT",
            "=" * 40,
            f"rows: {self.total_rows}   valid: {self.valid_rows}   "
            f"invalid: {self.total_rows - self.valid_rows}",
            f"overall: {'PASS' if self.ok else 'FAIL'}",
        ]
        for issue in self.issues[:max_rows]:
            lines.append(f"\nrow {issue.row_number} ({issue.image_id}):")
            for m in issue.missing_required:
                lines.append(f"  missing required: {m}")
            lines.extend(f"  unknown value: {u}" for u in issue.unknown_values)
            lines.extend(f"  out of range: {o}" for o in issue.out_of_range)
        if len(self.issues) > max_rows:
            lines.append(f"\n... {len(self.issues) - max_rows} more rows with issues")
        return "\n".join(lines)


def _nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _suggest(value: str, vocabulary) -> Optional[str]:
    matches = difflib.get_close_matches(value, list(vocabulary), n=1, cutoff=0.6)
    return matches[0] if matches else None


def lint_row(row: Dict[str, Any], row_number: int) -> RowIssues:
    image_id = str(row.get("image_id") or f"<row {row_number}>")
    issues = RowIssues(image_id=image_id, row_number=row_number)

    for column in REQUIRED:
        if not _nonempty(row.get(column)):
            issues.missing_required.append(column)

    for column, vocabulary in VOCAB_MAP.items():
        raw = row.get(column)
        if not _nonempty(raw):
            continue
        if column == "lighting_type":
            normalized = normalize_lighting(str(raw))
        elif column == "image_quality":
            normalized = normalize_quality(str(raw))
        else:
            normalized = str(raw).strip()
        if normalized not in vocabulary:
            suggestion = _suggest(normalized, vocabulary)
            suffix = f" (did you mean {suggestion!r}?)" if suggestion else ""
            issues.unknown_values.append(f"{column}: {normalized}{suffix}")

    for column, (lo, hi) in NUMERIC_RANGES.items():
        raw = row.get(column)
        if not _nonempty(raw):
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            issues.unknown_values.append(f"{column}: {raw} is not numeric")
            continue
        if not lo <= number <= hi:
            issues.out_of_range.append(f"{column}: {number} not in [{lo}, {hi}]")

    for column in MULTI_COLOUR:
        raw = row.get(column)
        if not _nonempty(raw):
            continue
        for token in str(raw).split(";"):
            token = token.strip()
            if token and token not in HARMONY_COLOURS:
                suggestion = _suggest(token, HARMONY_COLOURS)
                suffix = f" (did you mean {suggestion!r}?)" if suggestion else ""
                issues.unknown_values.append(f"{column}: {token}{suffix}")

    return issues


def lint_rows(rows: List[Dict[str, Any]]) -> LintReport:
    report = LintReport()
    for number, row in enumerate(rows, start=2):  # row 1 is the CSV header
        issues = lint_row(row, number)
        report.total_rows += 1
        if issues.ok:
            report.valid_rows += 1
        else:
            report.issues.append(issues)
    return report


def save_json(report: LintReport, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
