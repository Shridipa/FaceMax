"""Real-data quality audit + person-grouped split manifests.

Connects real annotation rows to the FROZEN bias evaluator (facemax_bias):
schema validation, leakage, class/subgroup distributions, annotator agreement,
registry completeness. Emits split manifests under data/splits/ and a report
under data/reports/data_quality_report.txt.

Data rules enforced here (and nowhere bypassed):
- labels are never fabricated (rows with missing truth are counted, not guessed)
- skin tone is never used as a proxy for race; no race/ethnicity columns exist
- minority samples are never duplicated to fake balance (duplicates are flagged)
- all images of one person stay in exactly one split (grouped_split)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from facemax_bias.distribution import distribution, flag_underrepresented
from facemax_bias.labels import agreement_distribution
from facemax_bias.leakage import grouped_split, leakage_check, split_by_person, split_stats
from facemax_bias.schema import validate as schema_validate
from facemax_bias.vocab import MAJOR_AXES

from .dataset import EXTENDED_COLUMNS, HARMONY_COLOURS, REGISTRY_FILENAME, read_annotations

QUALITY_DISTRIBUTION_AXES = [
    "skin_depth",
    "undertone",
    # chroma/contrast live in the EXTENDED schema as *_level categorical
    # columns (the frozen evaluator's `chroma`/`contrast` vocab applies).
    "chroma_level",
    "contrast_level",
    "lighting_type",
    "camera_type",
    "image_quality",
    "source",
    "makeup",
    "season_primary",
]


@dataclass
class AuditResult:
    total_rows: int
    schema_missing_required: List[str]
    schema_unknown_values: Dict[str, Dict[str, int]]
    duplicate_image_ids: List[str]
    person_leakage_pairs: List[Any]
    missing_truth: int
    single_annotator_rows: int
    agreement_distribution: Dict[str, int]
    distributions: Dict[str, Dict[str, Any]]
    representation_flags: List[str]
    registry_issues: List[str]
    harmony_persons: int = 0
    harmony_rows: int = 0
    missing_raw_images: List[str] = field(default_factory=list)
    duplicate_files: List[str] = field(default_factory=list)
    row_lint_invalid: int = 0
    row_lint_total: int = 0
    holdout_status: Dict[str, Any] = field(default_factory=dict)
    ingestion_warnings: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def ok(self) -> bool:
        return (
            not self.schema_missing_required
            and not self.duplicate_image_ids
            and not self.person_leakage_pairs
            and not self.registry_issues
            and not self.missing_raw_images
            and not self.duplicate_files
            and not (self.holdout_status and self.holdout_status.get("locked") and not self.holdout_status.get("ok"))
        )

    def render(self) -> str:
        lines = ["FACEAMAXX DATA QUALITY AUDIT".replace("FACEAMAXX", "FACEMAXX"), "=" * 40]
        lines.append(f"total annotated rows:      {self.total_rows}")
        lines.append(f"rows missing season truth: {self.missing_truth} (left unlabelled, never fabricated)")
        lines.append(f"single-annotator rows:     {self.single_annotator_rows}")
        lines.append("")
        lines.append("schema:")
        lines.append(f"  missing required:        {', '.join(self.schema_missing_required) or 'none'}")
        lines.append(f"  duplicate image_ids:     {len(self.duplicate_image_ids)}")
        unknown_total = sum(sum(c.values()) for c in self.schema_unknown_values.values())
        lines.append(f"  unknown label values:    {unknown_total}")
        lines.append(f"  person leakage pairs:    {len(self.person_leakage_pairs)}")
        lines.append("")
        lines.append("annotator agreement:")
        for key in ("unanimous", "majority", "split"):
            count = self.agreement_distribution.get(key, 0)
            total = max(sum(self.agreement_distribution.values()), 1)
            lines.append(f"  {key:<10} {count:>6}  ({count / total * 100:.1f}%)")
        lines.append("")
        lines.append("class/subgroup distributions (counts):")
        for column, info in self.distributions.items():
            counts = ", ".join(f"{label}:{count}" for label, count in list(info["counts"].items())[:10])
            lines.append(f"  {column}: {counts}")
        if self.representation_flags:
            lines.append("")
            lines.append("under-represented subgroups:")
            for flag in self.representation_flags[:10]:
                lines.append(f"  ! {flag}")
        lines.append("")
        lines.append(f"harmony annotations:       {self.harmony_rows} scores / {self.harmony_persons} persons")
        lines.append(f"registry issues:           {len(self.registry_issues)}")
        for issue in self.registry_issues:
            lines.append(f"  ! {issue}")
        if self.missing_raw_images:
            preview = ", ".join(self.missing_raw_images[:8])
            lines.append(f"missing raw images:        {len(self.missing_raw_images)} ({preview}...)")
        if self.duplicate_files:
            lines.append(f"duplicate image files:     {len(self.duplicate_files)}")
            for pair in self.duplicate_files[:5]:
                lines.append(f"  = {pair}")
        if self.row_lint_total:
            valid = self.row_lint_total - self.row_lint_invalid
            lines.append(f"row lint:                  {valid}/{self.row_lint_total} valid")
        if self.holdout_status:
            state = "locked+verified" if self.holdout_status.get("ok") else (
                "LOCK BROKEN" if self.holdout_status.get("locked") else "not locked"
            )
            lines.append(f"holdout lock:              {state}")
            for problem in self.holdout_status.get("problems", [])[:5]:
                lines.append(f"  ! {problem}")
        for warning in self.warnings:
            lines.append(f"warning: {warning}")
        lines.append("")
        lines.append(f"overall: {'PASS' if self.ok() else 'FAIL'}")
        return "\n".join(lines)


def _registry_issues(registry_path: Path) -> List[str]:
    """Lint DATASET_REGISTRY.md: every tracked dataset block must have license info."""
    issues: List[str] = []
    if not registry_path.exists():
        return [f"{REGISTRY_FILENAME} missing - create it before any download"]
    text = registry_path.read_text(encoding="utf-8")
    blocks = [block for block in text.split("\n---\n") if block.strip().startswith("## ")]
    if not blocks:
        return ["registry has no dataset blocks"]
    for block in blocks:
        title = block.strip().splitlines()[0].lstrip("# ").strip()
        for required in ("**license:**", "**intended use:**", "**downloaded:**"):
            if required not in block:
                issues.append(f"{title}: missing '{required.strip('*:').title()}' field")
    # External datasets marked downloaded must record a checksum.
    for block in blocks:
        title = block.strip().splitlines()[0].lstrip("# ").strip()
        downloaded_line = next((l for l in block.splitlines() if "**downloaded:**" in l), "")
        if "yes" in downloaded_line.lower() and "**checksum" not in block:
            issues.append(f"{title}: marked downloaded but checksum missing")
    return issues


def audit_dataset(
    root: Path,
    rows: Optional[List[Dict]] = None,
    harmony_rows: Optional[List[Dict]] = None,
) -> AuditResult:
    """Run every real-data quality check on the annotation table."""
    from .dataset import init_dataset

    root = Path(root)
    init_dataset(root)  # ensure tree (incl. DATASET_REGISTRY.md) exists before linting
    rows = rows if rows is not None else read_annotations(root / "metadata" / "facemaxx_color_annotations.csv")
    if harmony_rows is None:
        harmony_path = root / "annotations" / "harmony_scores.csv"
        harmony_rows = read_annotations(harmony_path) if harmony_path.exists() else []

    schema_report = schema_validate(rows)
    # Empty season_primary is EXPECTED during annotation-in-progress and is
    # reported separately as missing_truth (never fabricated to pass checks).
    # Only identity columns (image_id/person_id) are hard schema failures.
    schema_report.missing_required = [
        c for c in schema_report.missing_required if c not in ("season_primary",)
    ]
    leakage = leakage_check(rows)
    truth_present = [r for r in rows if (r.get("season_primary") or "").strip()]
    missing_truth = len(rows) - len(truth_present)
    single_annotator = sum(
        1 for r in rows if not any((r.get(f"annotator_{i}") or "").strip() for i in (1, 2, 3))
    )
    dist = distribution(rows, columns=QUALITY_DISTRIBUTION_AXES)
    registry_path = root / REGISTRY_FILENAME

    # --- file-layer checks: raw images present? identical content? ---------
    from .dataset import find_image

    raw_dir = root / "raw"
    missing_raw: List[str] = []
    hashes: Dict[str, List[str]] = {}
    for row in rows:
        image_id = str(row.get("image_id") or "")
        if not image_id:
            continue
        image_path = find_image(raw_dir, image_id)
        if image_path is None:
            missing_raw.append(image_id)
        else:
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            hashes.setdefault(digest, []).append(image_id)
    duplicate_files = [
        "=".join(sorted(ids)) for ids in hashes.values() if len(ids) > 1
    ]

    # --- row lint + holdout lock + ingestion receipts ----------------------
    from .validate import lint_rows

    lint = lint_rows(rows)
    from .ingest import audit_sources

    result = AuditResult(
        total_rows=len(rows),
        schema_missing_required=schema_report.missing_required,
        schema_unknown_values=schema_report.unknown_values,
        duplicate_image_ids=schema_report.duplicate_images,
        person_leakage_pairs=leakage["pairs"],
        missing_truth=missing_truth,
        single_annotator_rows=single_annotator,
        agreement_distribution=agreement_distribution(rows),
        distributions=dist,
        representation_flags=flag_underrepresented(dist),
        registry_issues=_registry_issues(registry_path),
        harmony_persons=len({r.get("person_id") for r in harmony_rows if r.get("person_id")}),
        harmony_rows=len(harmony_rows),
        missing_raw_images=missing_raw,
        duplicate_files=duplicate_files,
        row_lint_total=lint.total_rows,
        row_lint_invalid=lint.total_rows - lint.valid_rows,
        holdout_status=verify_holdout(root),
        ingestion_warnings=audit_sources(root),
    )
    if missing_truth:
        result.warnings.append(
            f"{missing_truth} rows have no season_primary yet; they are excluded from training, not guessed"
        )
    if result.total_rows and result.single_annotator_rows / result.total_rows > 0.5:
        result.warnings.append(
            "majority of rows single-annotated; agreement statistics unavailable until 2nd/3rd annotators added"
        )
    if harmony_rows:
        colours_seen = {str(r.get("colour") or "").strip() for r in harmony_rows}
        missing = [c for c in HARMONY_COLOURS if c not in colours_seen]
        if missing:
            result.warnings.append(f"harmony scores missing colours: {', '.join(missing[:8])}")
    for ing_warning in result.ingestion_warnings:
        result.warnings.append(f"ingest: {ing_warning}")
    if missing_raw:
        result.warnings.append(
            f"{len(missing_raw)} rows reference raw images that do not exist; "
            "annotations without files cannot be trained or evaluated"
        )
    if duplicate_files:
        result.warnings.append(
            "duplicate image files detected (identical content under different ids); "
            "these must be resolved, never counted twice"
        )
    return result


def write_split_manifests(
    root: Path,
    rows: List[Dict],
    test_frac: float = 0.2,
    seed: int = 7,
    label_column: Optional[str] = None,
) -> Dict[str, int]:
    """Person-grouped split + manifests under data/splits/. Returns counts.

    Uses facemax_bias.leakage (frozen evaluator logic) so the split policy is
    identical everywhere. Manifests are the artifact to freeze: later runs must
    reuse them rather than re-split.

    ``label_column``: when given (e.g. "season_primary"), the CLASS-AWARE
    person-grouped splitter is used so val/test keep every label class
    represented (a val split missing a class can never support calibration).
    Coverage is attempted with whole people only; when impossible it falls
    back to the plain grouped policy - never manufactured (spec #6).
    """
    root = Path(root)
    if label_column:
        from facemax_bias.leakage import grouped_split_class_aware

        split_rows = grouped_split_class_aware(
            [dict(r) for r in rows], test_frac=test_frac, seed=seed, label_column=label_column
        )
    else:
        split_rows = grouped_split([dict(r) for r in rows], test_frac=test_frac, seed=seed)
    counts: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        ids = sorted(str(r.get("image_id")) for r in split_rows if r.get("split") == split)
        counts[split] = len(ids)
        (root / "splits" / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    people = split_by_person(split_rows)
    (root / "splits" / "person_map.json").write_text(
        json.dumps({split: sorted(people[split]) for split in sorted(people)}, indent=2),
        encoding="utf-8",
    )
    stats = split_stats(split_rows)
    (root / "splits" / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return counts


SPLIT_FILES = ("train.txt", "val.txt", "test.txt", "person_map.json")
HOLDOUT_LOCK_FILENAME = "holdout.lock.json"


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_holdout(root: Path, rows: List[Dict], test_frac: float = 0.15, seed: int = 7, label_column: Optional[str] = None) -> Dict[str, Any]:
    """Re-split, write manifests, then FREEZE them with a tamper-evident lock.

    The lock file (data/splits/holdout.lock.json) stores sha256 digests of every
    manifest plus the split policy. From then on:
      - training/tuning may only use train/val;
      - the test split is the LOCKED HOLDOUT for the fairness evaluation;
      - any later modification of a manifest fails verify_holdout();
      - re-splitting requires audit --regen-splits --force (an explicit,
        logged human decision - never done implicitly).
    """
    import time

    root = Path(root)
    counts = write_split_manifests(root, rows, test_frac=test_frac, seed=seed, label_column=label_column)
    if label_column:
        from facemax_bias.leakage import grouped_split_class_aware

        split_rows = grouped_split_class_aware(
            [dict(r) for r in rows], test_frac=test_frac, seed=seed, label_column=label_column
        )
    else:
        split_rows = grouped_split([dict(r) for r in rows], test_frac=test_frac, seed=seed)
    check = leakage_check(split_rows)
    if not check["clean"]:
        raise RuntimeError(f"refusing to lock: split has person leakage: {check['pairs'][:3]}")
    lock = {
        "locked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_frac": test_frac,
        "seed": seed,
        "counts": counts,
        "digests": {name: _digest_file(root / "splits" / name) for name in SPLIT_FILES},
    }
    (root / "splits" / HOLDOUT_LOCK_FILENAME).write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    return lock


def verify_holdout(root: Path) -> Dict[str, Any]:
    """Verify manifests are unmodified since locking and still leakage-free."""
    root = Path(root)
    lock_path = root / "splits" / HOLDOUT_LOCK_FILENAME
    if not lock_path.exists():
        return {"locked": False, "ok": False, "reason": "no holdout lock file - splits are not frozen"}
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    problems: List[str] = []
    for name, digest in lock.get("digests", {}).items():
        path = root / "splits" / name
        if not path.exists():
            problems.append(f"{name}: missing (deleted after lock?)")
        elif _digest_file(path) != digest:
            problems.append(f"{name}: modified since lock")
    # Re-verify leakage from the manifest itself.
    person_map_path = root / "splits" / "person_map.json"
    if person_map_path.exists():
        person_map = json.loads(person_map_path.read_text(encoding="utf-8"))
        seen: Dict[str, str] = {}
        for split, people in person_map.items():
            for person in people:
                if person in seen and seen[person] != split:
                    problems.append(f"person {person} in {seen[person]} and {split}")
                seen.setdefault(person, split)
    return {"locked": True, "ok": not problems, "problems": problems,
            "locked_at": lock.get("locked_at"), "counts": lock.get("counts")}


def audit_from_root(root: Path) -> AuditResult:
    """Convenience wrapper reading everything from the dataset tree."""
    return audit_dataset(root)
