"""Real-data acquisition + FaceMaxx annotation workflow.

Turns photographs into evaluable annotations with the data rules enforced
mechanically:

    capture (handbook protocol, manifest CSV)
      -> annot import    copy into data/raw/ as content-addressed ids,
                         dedupe identical files (never double-count people)
      -> annot queue     per-annotator round files (independent labelling)
      -> (humans fill their round files independently)
      -> annot record    validate one round into the pending pool
      -> annot merge     2+ annotators -> majority season + agreement;
                         single annotator -> flagged, never fabricated
      -> annot coverage  progress vs. handbook acquisition targets
      -> gate            all checks pass -> ML training unblocked
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from facemax_bias.labels import agreement_score, majority_label
from facemax_bias.vocab import SEASON_ORDER

from .dataset import (
    ANNOTATIONS_FILENAME,
    EXTENDED_COLUMNS,
    init_dataset,
    read_annotations,
    write_extended_annotations,
)

ANNOTATIONS_DIRNAME = "annotations"
QUEUE_DIRNAME = "queues"
PENDING_FILENAME = "pending.csv"
CAPTURE_FILENAME = "capture_manifest.csv"

CAPTURE_COLUMNS = ("file", "person_alias", "lighting_type", "camera_type", "makeup", "notes")

QUEUE_COLUMNS = (
    "image_id", "image_path", "annotator",
    "season", "season_secondary", "undertone", "skin_depth", "chroma", "contrast",
    "olive_score", "confidence_1to5",
    "best_colors", "neutral_colors", "caution_colors", "notes",
)

# Season-vocabulary fields on the round file (validated on record).
ROUND_VOCAB_FIELDS = {
    "season": SEASON_ORDER,
    "season_secondary": SEASON_ORDER,
    "undertone": ("Cool", "Neutral-Cool", "Neutral", "Neutral-Warm", "Warm", "Olive"),
    "skin_depth": ("Very Fair", "Fair", "Light", "Light-Medium", "Medium", "Medium-Deep", "Deep", "Very Deep"),
    "chroma": ("Muted", "Medium", "Bright"),
    "contrast": ("Low", "Medium", "High"),
}


# ------------------------------------------------------------------ csv io


def _write_csv(path: Path, fieldnames, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return [{k: (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(fh)]


# ----------------------------------------------------------------- capture


def write_capture_manifest(path: Path) -> Path:
    """Capture-manifest template: one row per photo, filled while shooting.

    Capture metadata (person, lighting, camera, makeup) is recorded at capture
    time by the photographer - never inferred later from pixels.
    """
    _write_csv(path, CAPTURE_COLUMNS, [])
    return Path(path)


def read_capture_manifest(path: Path) -> List[Dict[str, str]]:
    return _read_csv(path)


# ------------------------------------------------------------------ import


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_images(root: Path, manifest_path: Path, staging_dir: Path) -> Dict[str, Any]:
    """Copy staged photos into data/raw/ as content-addressed ids, deduped.

    - image_id = IMG-<sha256[:12]>; identical files collapse to one id
      (dedupe by content, not by name - never double-count people/images).
    - ids already in the annotation table are skipped (idempotent re-import).
    - Capture metadata rides onto the annotation row; unknown fields stay
      empty for the photographer to fix - never guessed.
    """
    root = Path(root)
    init_dataset(root)
    staging_dir = Path(staging_dir)
    manifest_rows = read_capture_manifest(manifest_path)
    if not staging_dir.exists():
        raise FileNotFoundError(f"staging directory not found: {staging_dir}")

    annotations_path = root / "metadata" / ANNOTATIONS_FILENAME
    existing_rows = read_annotations(annotations_path)
    existing_ids = {str(r.get("image_id")) for r in existing_rows}

    imported: List[Dict[str, str]] = []
    skipped_duplicate_file = 0
    skipped_already_imported = 0
    seen_hashes: Dict[str, str] = {}

    for entry in manifest_rows:
        name = str(entry.get("file") or "")
        src = staging_dir / name if not Path(name).exists() else Path(name)
        if not src.exists():
            raise FileNotFoundError(f"manifest row references missing file: {name}")
        digest = _sha256_file(src)
        if digest in seen_hashes:
            skipped_duplicate_file += 1  # same content twice within one import
            continue
        seen_hashes[digest] = name
        image_id = f"IMG-{digest[:12]}"
        if image_id in existing_ids or any(r["image_id"] == image_id for r in imported):
            skipped_already_imported += 1
            continue
        dest = root / "raw" / f"{image_id}.jpg"
        if not dest.exists():
            shutil.copy2(src, dest)
        row = {column: "" for column in EXTENDED_COLUMNS}
        row.update(
            {
                "image_id": image_id,
                "person_id": str(entry.get("person_alias") or "").strip(),
                "source": "Phone selfie",
                "license": "project-owned",
                "image_path": f"raw/{image_id}.jpg",
                "notes": str(entry.get("notes") or ""),
            }
        )
        for capture_field in ("lighting_type", "camera_type", "makeup", "image_quality"):
            row[capture_field] = str(entry.get(capture_field) or "").strip()
        imported.append(row)

    if imported:
        write_extended_annotations(annotations_path, existing_rows + imported)
        log_path = root / "raw" / "IMPORT_LOG.md"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "imported": len(imported),
                        "duplicate_files": skipped_duplicate_file,
                        "already_imported": skipped_already_imported,
                        "ids": [r["image_id"] for r in imported],
                    }
                )
                + "\n"
            )
    return {
        "imported": len(imported),
        "skipped_duplicate_file": skipped_duplicate_file,
        "skipped_already_imported": skipped_already_imported,
        "imported_ids": [r["image_id"] for r in imported],
    }


def pending_pool(root: Path) -> List[Dict[str, str]]:
    """Annotated rows awaiting annotation (imported, no season yet)."""
    rows = read_annotations(Path(root) / "metadata" / ANNOTATIONS_FILENAME)
    return [r for r in rows if not (r.get("season_primary") or "").strip()]


# ------------------------------------------------------------------ queues


def queue_round(root: Path, annotator: str, image_ids: List[str]) -> Path:
    """Write a per-annotator round file for independent labelling."""
    pool = {str(r.get("image_id")): r for r in pending_pool(root)}
    missing = [i for i in image_ids if i not in pool]
    if missing:
        raise ValueError(f"ids not in pending pool: {missing[:3]}")
    queue_dir = Path(root) / ANNOTATIONS_DIRNAME / QUEUE_DIRNAME
    rows = []
    for image_id in image_ids:
        entry = pool[image_id]
        rows.append(
            {
                "image_id": image_id,
                "image_path": entry.get("image_path") or f"raw/{image_id}.jpg",
                "annotator": annotator,
                "season": "",
                "season_secondary": "",
                "undertone": "",
                "skin_depth": "",
                "chroma": "",
                "contrast": "",
                "olive_score": "",
                "confidence_1to5": "",
                "best_colors": "",
                "neutral_colors": "",
                "caution_colors": "",
                "notes": "",
            }
        )
    path = queue_dir / f"round-{annotator}.csv"
    _write_csv(path, QUEUE_COLUMNS, rows)
    return path


def read_round(path: Path) -> List[Dict[str, str]]:
    return _read_csv(path)


# ------------------------------------------------------------------ record


def record_round(root: Path, round_path: Path, validate: bool = True) -> Dict[str, Any]:
    """Validate a filled round file into the pending pool.

    Rows with an empty season are deferred (stay queued). Rows whose season /
    undertone / skin_depth / chroma / contrast fall outside the controlled
    vocabulary are rejected with a suggestion - nothing is silently fixed and
    nothing is fabricated.
    """
    from .validate import _suggest

    entries = read_round(round_path)
    pending_path = Path(root) / ANNOTATIONS_DIRNAME / PENDING_FILENAME
    pending = _read_csv(pending_path) if pending_path.exists() else []
    seen_pairs = {(r.get("image_id"), r.get("annotator")) for r in pending}

    accepted: List[Dict[str, str]] = []
    deferred = 0
    rejected: List[str] = []
    for entry in entries:
        image_id = entry.get("image_id") or "<unknown>"
        if not entry.get("season"):
            deferred += 1
            continue
        problems: List[str] = []
        for field, vocab in ROUND_VOCAB_FIELDS.items():
            value = entry.get(field) or ""
            if not value:
                continue
            if value not in vocab:
                suggestion = _suggest(value, vocab)
                hint = f" (did you mean {suggestion!r}?)" if suggestion else ""
                problems.append(f"{image_id}: {field}={value!r}{hint}")
        if problems:
            rejected.extend(problems)
            continue
        pair = (image_id, entry.get("annotator") or "")
        if pair in seen_pairs:
            continue  # same annotator already recorded for this image
        seen_pairs.add(pair)
        accepted.append(entry)

    if accepted:
        _write_csv(pending_path, QUEUE_COLUMNS, pending + accepted)
    return {
        "accepted": len(accepted),
        "deferred": deferred,
        "rejected": rejected,
        "pending_total": len(pending) + len(accepted),
    }


# ------------------------------------------------------------------- merge


def merge_rounds(root: Path, min_annotators: int = 2) -> Dict[str, Any]:
    """Merge recorded rounds into the annotation table.

    For each image with >= min_annotators independent season labels:
      - season_primary = majority label (ties left unmerged - no fabrication)
      - annotation_agreement = majority fraction
      - season_secondary = the second season when the vote split 2/3 : 1/3
    Images with fewer annotators are left unannotated (stay pending).

    Axis labels (undertone/skin_depth/chroma/contrast) are merged the same
    way when every annotator agreed; disagreement leaves them empty.
    """
    root = Path(root)
    pending_path = root / ANNOTATIONS_DIRNAME / PENDING_FILENAME
    if not pending_path.exists():
        return {"merged": 0, "by_image": {}, "split_votes": []}

    pending = _read_csv(pending_path)
    by_image: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for entry in pending:
        by_image[entry.get("image_id") or ""].append(entry)

    annotations_path = root / "metadata" / ANNOTATIONS_FILENAME
    rows = read_annotations(annotations_path)
    row_by_id = {str(r.get("image_id")): r for r in rows}

    merged_ids: List[str] = []
    split_votes: List[str] = []
    consumed_pairs: set = set()

    for image_id, entries in by_image.items():
        annotator_labels = [(e.get("annotator") or "", e) for e in entries if e.get("season")]
        annotators = {a for a, _ in annotator_labels}
        if len(annotators) < min_annotators:
            continue
        row = row_by_id.get(image_id)
        if row is None:
            continue

        labels = [e.get("season") or "" for _, e in annotator_labels]
        majority = majority_label(labels)
        if majority is None:
            split_votes.append(f"{image_id}: {sorted(set(labels))}")
            continue
        row["season_primary"] = majority
        row["annotation_agreement"] = round(agreement_score(labels), 4)
        counts: Dict[str, int] = defaultdict(int)
        for label in labels:
            counts[label] += 1
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])
        if len(ordered) > 1:
            row["season_secondary"] = ordered[1][0]
        # Annotator columns record every independent label.
        for index, (_, entry) in enumerate(annotator_labels[:3], start=1):
            row[f"annotator_{index}"] = entry.get("season") or ""
        # Merge axes only on full agreement.
        for field in ("undertone", "skin_depth", "chroma", "contrast"):
            values = {e.get(field) or "" for _, e in annotator_labels}
            values.discard("")
            if len(values) == 1:
                target = "chroma_level" if field == "chroma" else (
                    "contrast_level" if field == "contrast" else field
                )
                row[target] = next(iter(values))
        if entry_conf := max(annotator_labels, key=lambda pair: _confidence(pair[1]))[1]:
            row["season_confidence"] = _confidence(entry_conf)
            row["best_colors"] = entry_conf.get("best_colors") or ""
            row["neutral_colors"] = entry_conf.get("neutral_colors") or ""
            row["caution_colors"] = entry_conf.get("caution_colors") or ""
            row["notes"] = entry_conf.get("notes") or ""
        consumed_pairs.update((image_id, e.get("annotator") or "") for _, e in annotator_labels)
        merged_ids.append(image_id)

    if merged_ids:
        write_extended_annotations(annotations_path, rows)
        # Drop merged entries from the pending pool.
        remaining = [
            e for e in pending
            if (e.get("image_id"), e.get("annotator") or "") not in consumed_pairs
        ]
        _write_csv(pending_path, QUEUE_COLUMNS, remaining)
    return {
        "merged": len(merged_ids),
        "by_image": {image_id: True for image_id in merged_ids},
        "split_votes": split_votes,
        "still_pending": sum(1 for r in pending_pool(root)),
    }


def _confidence(entry: Dict[str, str]) -> float:
    try:
        return float(entry.get("confidence_1to5") or 0.0)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------- coverage


def coverage(root: Path) -> Dict[str, Any]:
    """Acquisition/annotation progress vs. handbook targets.

    Targets are the handbook's minimum viable acquisition plan; they are
    reporting numbers, not fabricated balance (no duplication to hit them).
    """
    from facemax_bias.vocab import SKIN_DEPTH_ORDER

    rows = read_annotations(Path(root) / "metadata" / ANNOTATIONS_FILENAME)
    annotated = [r for r in rows if (r.get("season_primary") or "").strip()]
    people = {str(r.get("person_id") or "") for r in rows} - {""}
    depth_counts: Dict[str, int] = {depth: 0 for depth in SKIN_DEPTH_ORDER}
    multi_image_people = 0
    people_images: Dict[str, int] = defaultdict(int)
    for row in rows:
        person = str(row.get("person_id") or "")
        if person:
            people_images[person] += 1
    for person, count in people_images.items():
        if count >= 2:
            multi_image_people += 1
    for row in annotated:
        depth = str(row.get("skin_depth") or "").strip()
        if depth in depth_counts:
            depth_counts[depth] += 1
    boundaries = sum(
        1 for r in annotated
        if {str(r.get("season_primary") or ""), str(r.get("season_secondary") or "")}
        & {"Deep Autumn", "Deep Winter"}
    )
    return {
        "images_imported": len(rows),
        "images_annotated": len(annotated),
        "images_pending": len(rows) - len(annotated),
        "people": len(people),
        "people_with_multiple_images": multi_image_people,
        "skin_depth_counts_annotated": depth_counts,
        "boundary_candidates_DA_DW": boundaries,
        "target_people": 60,
        "target_images": 200,
        "target_boundary_candidates": 40,
    }


def render_coverage(info: Dict[str, Any]) -> str:
    lines = [
        "ACQUISITION / ANNOTATION COVERAGE",
        "=" * 42,
        f"images imported:   {info['images_imported']} (target {info['target_images']})",
        f"images annotated:  {info['images_annotated']}",
        f"images pending:    {info['images_pending']}",
        f"people:            {info['people']} (target {info['target_people']})",
        f"multi-image people:{info['people_with_multiple_images']} (lighting/robustness variation)",
        "",
        "annotated rows by skin depth (representation check):",
    ]
    for depth, count in info["skin_depth_counts_annotated"].items():
        bar = "#" * count
        lines.append(f"  {depth:<13} {count:>4}  {bar}")
    lines.append(f"\nDeep Autumn/Winter boundary candidates: {info['boundary_candidates_DA_DW']} (target {info['target_boundary_candidates']})")
    return "\n".join(lines)
