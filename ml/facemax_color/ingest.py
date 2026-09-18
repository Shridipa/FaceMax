"""External-dataset ingestion: official sources only, license-gated, traceable.

This module deliberately does NOT download anything. The rule for FaceMaxx:

    1. The human reviews the dataset's official terms and completes the
       ``license`` field of its DATASET_REGISTRY.md block.
    2. The human downloads the archive from the OFFICIAL source themselves.
    3. ``python -m facemax_color ingest record --dataset NAME --file PATH``
       computes sha256, writes an ingestion receipt, and marks the source
       downloaded in data/ingest/sources.json.

Nothing enters data/raw/ without a receipt; no scraping, no unlicensed
copying, no manufactured labels. External datasets are NEVER seasonal ground
truth (see DATASET_REGISTRY.md rules).
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataset import REGISTRY_FILENAME, init_dataset

INGEST_DIRNAME = "ingest"
SOURCES_FILENAME = "sources.json"


class IngestError(RuntimeError):
    """Raised when an ingestion step violates the data rules."""


# External datasets under consideration. ``license_gate`` must be flipped to
# "confirmed" by a human (--license-confirmed) AFTER reading official terms.
EXTERNAL_SOURCE_TEMPLATE: List[Dict[str, Any]] = [
    {
        "name": "CelebAMask-HQ",
        "official_url": "https://github.com/switchablenorms/CelebAMask-HQ",
        "intended_use": "face parsing masks (skin/hair/eyes/lips) only - NEVER seasonal ground truth",
        "license_gate": "unconfirmed",
        "status": "not_downloaded",
        "checksum": None,
        "receipt": None,
    },
    {
        "name": "FFHQ",
        "official_url": "https://github.com/NVlabs/ffhq-dataset",
        "intended_use": "preprocessing/robustness research; per-image Flickr licensing respected",
        "license_gate": "unconfirmed",
        "status": "not_downloaded",
        "checksum": None,
        "receipt": None,
    },
    {
        "name": "Fitzpatrick17k",
        "official_url": "https://github.com/mattgroh/fitzpatrick17k",
        "intended_use": "skin-depth robustness evaluation only; Fitzpatrick labels are NOT undertone/season labels",
        "license_gate": "unconfirmed",
        "status": "not_downloaded",
        "checksum": None,
        "receipt": None,
    },
]


def _ingest_dir(root: Path) -> Path:
    return Path(root) / INGEST_DIRNAME


def sources_path(root: Path) -> Path:
    return _ingest_dir(root) / SOURCES_FILENAME


def ensure_sources(root: Path) -> Path:
    """Create sources.json from the template if absent. Returns its path."""
    init_dataset(root)  # guarantees ingest/ dir exists
    path = sources_path(root)
    if not path.exists():
        write_sources(root, EXTERNAL_SOURCE_TEMPLATE)
    return path


def read_sources(root: Path) -> List[Dict[str, Any]]:
    ensure_sources(root)
    return json.loads(sources_path(root).read_text(encoding="utf-8"))


def write_sources(root: Path, sources: List[Dict[str, Any]]) -> None:
    path = sources_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streamed sha256 so multi-GB archives never load into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _registry_license(root: Path, dataset: str) -> Optional[str]:
    """Return the license field text of the dataset's registry block (or None)."""
    registry = Path(root) / REGISTRY_FILENAME
    if not registry.exists():
        return None
    text = registry.read_text(encoding="utf-8")
    for block in text.split("\n---\n"):
        head = next((line for line in block.splitlines() if line.strip().startswith("## ")), "")
        if dataset.lower() in head.lower():
            license_line = next(
                (line for line in block.splitlines() if line.strip().startswith("- **license:**")),
                "",
            )
            # Parse markdown properly: "- **license:** some text" -> "some text".
            match = re.search(r"\*\*license:\*\*\s*(.*)", license_line)
            return match.group(1).strip() if match else ""
    return None


def _safe_name(dataset: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", dataset).strip("_") or "unknown"


def record_download(
    root: Path,
    dataset: str,
    archive: Path,
    license_confirmed: bool = False,
) -> Dict[str, Any]:
    """Register a human-made download: sha256 receipt + sources.json update.

    Refuses (IngestError) unless:
      - the dataset is a known external source,
      - its DATASET_REGISTRY.md block has a completed license field,
      - the caller passes license_confirmed=True (the --license-confirmed flag),
        certifying a human reviewed the official terms.
    """
    root = Path(root)
    archive = Path(archive)
    sources = read_sources(root)
    entry = next((s for s in sources if s["name"].lower() == dataset.lower()), None)
    if entry is None:
        known = ", ".join(s["name"] for s in sources)
        raise IngestError(f"unknown dataset {dataset!r}; known sources: {known}")
    if not archive.exists():
        raise IngestError(f"archive not found: {archive}")

    license_text = _registry_license(root, entry["name"])
    if license_text is None or not license_text:
        raise IngestError(
            f"complete the 'license' field of the {entry['name']} block in "
            f"{REGISTRY_FILENAME} before recording a download"
        )
    if not license_confirmed:
        raise IngestError(
            "license not confirmed: a human must review the official terms, then "
            "re-run with --license-confirmed"
        )

    checksum = sha256_file(archive)
    receipt_name = f"{_safe_name(entry['name'])}.receipt.json"
    receipt = {
        "dataset": entry["name"],
        "archive": str(archive),
        "sha256": checksum,
        "bytes": archive.stat().st_size,
        "license": license_text,
        "recorded": date.today().isoformat(),
    }
    receipt_path = _ingest_dir(root) / receipt_name
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    entry.update(
        {
            "status": "downloaded",
            "checksum": checksum,
            "receipt": receipt_name,
            "license_gate": "confirmed",
        }
    )
    write_sources(root, sources)
    return receipt


def verify_receipt(root: Path, dataset: str, rehash: bool = False) -> Dict[str, Any]:
    """Verify a recorded download. With rehash=True the archive is re-hashed.

    Returns {"ok": bool, "locked"... "reason": str|None, "stored"/"computed": ...}.
    """
    root = Path(root)
    sources = read_sources(root)
    entry = next((s for s in sources if s["name"].lower() == dataset.lower()), None)
    if entry is None:
        return {"ok": False, "reason": f"unknown dataset {dataset!r}"}
    if entry.get("status") != "downloaded" or not entry.get("receipt"):
        return {"ok": False, "reason": "not downloaded / no receipt recorded"}
    receipt_path = _ingest_dir(root) / entry["receipt"]
    if not receipt_path.exists():
        return {"ok": False, "reason": f"receipt missing: {receipt_path}"}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    stored = receipt.get("sha256")
    if not rehash:
        return {"ok": bool(stored), "reason": None if stored else "receipt has no checksum",
                "stored": stored}
    archive = Path(receipt.get("archive", ""))
    if not archive.exists():
        return {"ok": False, "reason": f"archive missing: {archive}", "stored": stored}
    computed = sha256_file(archive)
    if computed != stored:
        return {"ok": False, "reason": "checksum mismatch - archive changed after receipt",
                "stored": stored, "computed": computed}
    return {"ok": True, "reason": None, "stored": stored, "computed": computed}


def ingestion_plan(root: Path) -> List[Dict[str, Any]]:
    """Status of every external source (for the `ingest plan` command)."""
    sources = read_sources(root)
    plan: List[Dict[str, Any]] = []
    for entry in sources:
        license_text = _registry_license(root, entry["name"])
        plan.append(
            {
                "name": entry["name"],
                "status": entry.get("status", "not_downloaded"),
                "license_gate": entry.get("license_gate", "unconfirmed"),
                "registry_license": license_text or "(empty - complete before download)",
                "receipt": entry.get("receipt"),
                "official_url": entry.get("official_url", ""),
                "intended_use": entry.get("intended_use", ""),
            }
        )
    return plan


def render_plan(plan: List[Dict[str, Any]]) -> str:
    lines = ["EXTERNAL SOURCE INGESTION PLAN", "=" * 46]
    for item in plan:
        lines.append(
            f"\n{item['name']}  [{item['status']}]  license gate: {item['license_gate']}"
        )
        lines.append(f"  official source: {item['official_url']}")
        lines.append(f"  intended use:    {item['intended_use']}")
        lines.append(f"  registry license: {item['registry_license']}")
        lines.append(f"  receipt: {item['receipt'] or '-'}")
    lines.append(
        "\nworkflow: review official terms -> complete registry license field ->\n"
        "  download from the official source -> ingest record --license-confirmed\n"
        "  (external datasets are NEVER seasonal ground truth)"
    )
    return "\n".join(lines)


def audit_sources(root: Path) -> List[str]:
    """Warnings for inconsistent ingestion state (used by the data audit)."""
    warnings: List[str] = []
    for entry in read_sources(root):
        if entry.get("status") == "downloaded":
            receipt = entry.get("receipt")
            if not receipt or not (_ingest_dir(root) / receipt).exists():
                warnings.append(f"{entry['name']}: marked downloaded but receipt missing")
    return warnings
