"""Resumable CelebAMask-HQ mask extraction (tool, run manually).

Extracts mask PNGs from the verified archive into the designated real-data
directory, skipping files that already exist so interrupted runs resume.
Reports remaining count per run. Read-only w.r.t. the archive.
"""
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP = ROOT / "data/raw/external/CelebAMask-HQ/CelebAMask-HQ.zip"
DEST = ROOT / "data/raw/external/CelebAMask-HQ"


def main() -> int:
    t0 = time.time()
    with zipfile.ZipFile(ZIP) as z:
        masks = [
            n for n in z.namelist()
            if n.startswith("CelebAMask-HQ/CelebAMask-HQ-mask-anno/")
            and n.endswith(".png")
        ]
        remaining = [
            n for n in masks if not (DEST / n).exists()
        ]
        print(f"total masks: {len(masks)}  already extracted: {len(masks) - len(remaining)}  remaining: {len(remaining)}", flush=True)
        done = 0
        for name in remaining:
            z.extract(name, DEST)
            done += 1
            if done % 20000 == 0:
                elapsed = time.time() - t0
                print(f"  {done}/{len(remaining)} extracted ({done/elapsed:.0f}/s)", flush=True)
    print(f"run complete: extracted {done} masks in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
