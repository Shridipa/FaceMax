"""CLI: python -m facemax_bias selfcheck | demo | report

    selfcheck  run a fast end-to-end smoke test (no data required)
    demo       generate a synthetic biased dataset (CSV + JSONL)
    report     build the Model Bias & Robustness Report
               (from --csv/--jsonl input, or the synthetic demo if none given)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

# Force UTF-8 output so Windows consoles (cp1252) do not choke on any
# non-ASCII character that might slip into a report.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from .demo import demo_summary, generate_rows
from .report import generate_report


def _read_rows(path: Path) -> list:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_rows(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")


def cmd_selfcheck() -> int:
    rows = generate_rows(n_people=60, images_per_person=2, seed=3)
    report = generate_report(rows)
    text = report.text
    required = (
        "FACEMAXX",
        "Overall Accuracy",
        "SUBGROUP PERFORMANCE",
        "REPRESENTATION",
        "CONFUSION",
        "ROBUSTNESS",
        "SPLIT INTEGRITY",
    )
    missing = [token for token in required if token not in text]
    assert len(rows) == 120, f"expected 120 rows, got {len(rows)}"
    assert not missing, f"report missing sections: {missing}"
    print("selfcheck OK -", demo_summary(rows))
    print("report rendered:", len(text.splitlines()), "lines")
    return 0


def cmd_demo(outdir: Path, n_people: int, seed: int) -> int:
    rows = generate_rows(n_people=n_people, seed=seed)
    _write_rows(rows, outdir / "annotations.csv")
    _write_rows(rows, outdir / "annotations.jsonl")
    report = generate_report(rows)
    (outdir / "sample_report.txt").write_text(report.text, encoding="utf-8")
    report.save_json(str(outdir / "sample_report.json"))
    print(f"wrote {outdir}/annotations.csv, annotations.jsonl, sample_report.txt, sample_report.json")
    print(demo_summary(rows))
    return 0


def cmd_report(inputs: list, out: Path, title: str) -> int:
    rows: list = []
    for path in inputs:
        rows.extend(_read_rows(Path(path)))
    if not rows:
        print("no input rows; generating synthetic demo data instead", file=sys.stderr)
        rows = generate_rows(n_people=1000)
    report = generate_report(rows, title=title)
    print(report.text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.text, encoding="utf-8")
    report.save_json(str(out.with_suffix(".json")))
    print(f"\nsaved: {out} and {out.with_suffix('.json')}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="facemax_bias", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("selfcheck", help="fast end-to-end smoke test")

    demo_p = sub.add_parser("demo", help="generate synthetic biased dataset")
    demo_p.add_argument("--out", default="ml/data", help="output directory")
    demo_p.add_argument("--people", type=int, default=1000)
    demo_p.add_argument("--seed", type=int, default=7)

    report_p = sub.add_parser("report", help="build the bias & robustness report")
    report_p.add_argument("inputs", nargs="*", help="annotation CSV/JSONL files")
    report_p.add_argument("--out", default="ml/data/report.txt")
    report_p.add_argument("--title", default="FACEMAXX MODEL FAIRNESS REPORT")

    args = parser.parse_args(argv)
    if args.command == "selfcheck":
        return cmd_selfcheck()
    if args.command == "demo":
        return cmd_demo(Path(args.out), args.people, args.seed)
    if args.command == "report":
        return cmd_report(args.inputs, Path(args.out), args.title)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
