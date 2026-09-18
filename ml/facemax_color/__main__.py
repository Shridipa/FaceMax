"""FaceMaxx colour pipeline CLI.

    python -m facemax_color init-data   [--root data]
    python -m facemax_color extract     [--root data] [--images data/raw]
    python -m facemax_color train       [--root data] [--out models] [--fast]
    python -m facemax_color evaluate    [--root data] [--out reports] [--fast] [--lock-split]
    python -m facemax_color demo-image  [--root data] [--people 40]
    python -m facemax_color validate    [--root data] [--out data/reports]
    python -m facemax_color ingest plan | record | verify
    python -m facemax_color annot manifest | import | queue | record | merge | coverage | pending
    python -m facemax_color gate        [--root data]   # ML-training gate
    python -m facemax_color audit       [--root data] [--splits | --regen-splits [--force]]

Typical flow:
    1. init-data   create the dataset tree + annotation template
    2. (human)     fill metadata/facemaxx_color_annotations.csv; drop images in raw/
    3. extract     run face detection + skin LAB extraction on every image
    4. train       fit per-axis models, print the LogReg/RF/HGB comparison
    5. evaluate    person-split holdout -> predictions -> FINAL BIAS REPORT

`demo-image` builds a small synthetic image dataset (Haar-detectable cartoon
faces with planted ground truth) and runs the ENTIRE loop end to end, proving
the CV pipeline, models and bias evaluator work together before real data lands.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

from .dataset import ANNOTATIONS_FILENAME


def _load_annotations(root: Path):
    from .dataset import ANNOTATIONS_FILENAME, read_annotations

    return read_annotations(root / "metadata" / ANNOTATIONS_FILENAME)


def cmd_init_data(args) -> int:
    from .dataset import init_dataset
    from .ingest import ensure_sources

    root = Path(args.root)
    annotations = init_dataset(root)
    ensure_sources(root)  # external-source ingestion gates (sources.json)
    print(f"dataset tree ready under {root}")
    print(f"annotation template: {annotations}")
    print(f"ingestion gates: {root / 'ingest' / 'sources.json'} (see `ingest plan`)")
    print("next: fill the CSV, put images in data/raw/ as <image_id>.jpg|png, then run `extract`")
    return 0


def cmd_extract(args) -> int:
    from .colour import extract_batch
    from .dataset import ANNOTATIONS_FILENAME, init_dataset, write_annotations

    root = Path(args.root)
    init_dataset(root)
    rows = _load_annotations(root)
    if not rows:
        print("annotation CSV has no data rows yet - fill the template first", file=sys.stderr)
        return 1

    ids = [str(r.get("image_id")) for r in rows if r.get("image_id")]
    print(f"extracting colour features for {len(ids)} images ...")
    results = extract_batch(ids, Path(args.images), progress_every=100)

    ok = sum(1 for r in results.values() if r.ok)
    errors: dict = {}
    for r in results.values():
        if not r.ok:
            key = r.error or "unknown"
            errors[key] = errors.get(key, 0) + 1
    print(f"extraction finished: {ok}/{len(results)} ok")
    for err, count in sorted(errors.items(), key=lambda kv: -kv[1]):
        print(f"  {err}: {count}")

    # Persist feature table (processed/).
    features_path = root / "processed" / "colour_features.csv"
    fieldnames = ["image_id"] + list(next(iter(results.values())).as_feature_row().keys())
    with features_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for image_id in ids:
            row = {"image_id": image_id}
            row.update(results[image_id].as_feature_row())
            writer.writerow(row)
    print(f"feature table: {features_path}")

    # Rewrite annotations, preserving the extended schema (traceability,
    # harmony targets, notes) plus the extraction status flag.
    from .dataset import EXTENDED_COLUMNS

    existing_columns = list(EXTENDED_COLUMNS) + ["features_ok"]
    for row in rows:
        res = results.get(str(row.get("image_id")))
        row["features_ok"] = bool(res and res.ok)
    annotations_path = root / "metadata" / ANNOTATIONS_FILENAME
    with annotations_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=existing_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"annotations updated with features_ok: {annotations_path}")
    return 0


def _training_paths(root: Path, demo: bool) -> Dict[str, Path]:  # noqa: ANN201
    """Resolve annotation/feature/model paths for real vs ISOLATED demo mode.

    Demo mode never reads or writes anything outside ``root/demo/``: demo
    annotations, demo features and demo model artifacts are fully separate
    from the real annotation table, real features and data/models.
    """
    if demo:
        demo_dir = root / "demo"
        return {
            "data_root": demo_dir,
            "annotations": demo_dir / "annotations.csv",
            "features": demo_dir / "colour_features.csv",
            "models": demo_dir / "models",
            "reports": demo_dir,
        }
    return {
        "data_root": root,
        "annotations": root / "metadata" / ANNOTATIONS_FILENAME,
        "features": root / "processed" / "colour_features.csv",
        "models": root / "models",
        "reports": root / "reports",
    }


def _gate_allows_training(root: Path, smoke: bool) -> tuple:
    """(allowed, status, reasons) from the readiness audit. Never modified,
    never bypassed: NOT READY always blocks real-data training."""
    from .readiness import readiness

    report = readiness(root)
    if report.status == "NOT READY":
        return False, "NOT READY", list(report.blockers)
    if report.status == "SMOKE-READY" and not smoke:
        reasons = [
            "dataset is SMOKE-READY only: gate minimums met but training tier not reached",
            "SMOKE-READY data may only produce wiring smoke tests (artifacts marked smoke)",
            "re-run with --smoke to acknowledge this, or collect data to reach TRAIN-READY",
        ] + list(report.train_gaps[:5])
        return False, "SMOKE-READY", reasons
    return True, report.status, []


def _print_training_blocked(status: str, reasons: List[str], demo_flag: str = "--demo") -> None:
    print("=" * 46)
    print("ML TRAINING BLOCKED")
    print("=" * 46)
    print(f"Dataset status: {status}")
    print()
    print("Resolve:")
    for reason in reasons:
        print(f"  - {reason}")
    print()
    print("No real-data model was trained.")
    print(f"Pipeline testing on isolated synthetic data remains available: train {demo_flag}")


def _ensure_demo_splits(demo_dir: Path, seed: int) -> Dict[str, int]:
    """Create demo-tree split manifests (person-grouped, deterministic) once.

    The demo tree is synthetic and isolated; this bootstrap simply mirrors the
    real flow (manifest-driven prep + holdout evaluation) using the SAME
    frozen grouped_split logic. Real data NEVER gets splits from here.
    """
    from facemax_bias.leakage import grouped_split, leakage_check, split_stats

    splits_dir = demo_dir / "splits"
    if (splits_dir / "train.txt").exists():
        return {}
    try:
        from .dataset import read_annotations

        rows = read_annotations(demo_dir / "annotations.csv")  # demo-local table
    except FileNotFoundError:
        return {}  # no demo tree yet; per-target handling reports it clearly
    if not rows:
        return {}
    split_rows = grouped_split([dict(r) for r in rows], test_frac=0.2, seed=seed)
    check = leakage_check(split_rows)
    if not check["clean"]:  # defensive: grouped_split is person-grouped
        raise RuntimeError(f"demo split has person leakage: {check['pairs'][:3]}")
    counts: Dict[str, int] = {}
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        ids = sorted(str(r.get("image_id")) for r in split_rows if r.get("split") == split)
        counts[split] = len(ids)
        (splits_dir / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    stats = split_stats(split_rows)
    (splits_dir / "split_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"demo split manifests written to {splits_dir}: {counts}")
    return counts


def cmd_train(args) -> int:
    """Supervised training, gated by the readiness audit (or --demo)."""
    from .training import load_config, run_training
    from .training_data import TARGETS, prepare_training_data, save_report

    root = Path(args.root)
    demo = bool(getattr(args, "demo", False))
    paths = _training_paths(root, demo)

    smoke = demo or bool(getattr(args, "smoke", False))
    if demo:
        print("DEMO/SYNTHETIC training mode: everything stays under", root / "demo")
        print("DEMO / SYNTHETIC RESULT - NOT A FACEMAXX PERFORMANCE BENCHMARK")
        print()
    else:
        allowed, status, reasons = _gate_allows_training(root, smoke)
        if not allowed:
            _print_training_blocked(status, reasons)
            return 1
        print(f"dataset status: {status} - real-data training allowed")
        print()

    config_path = Path(args.config) if getattr(args, "config", None) else None
    if config_path is not None and not config_path.exists():
        print(f"config not found ({config_path}); using built-in defaults", file=sys.stderr)
        config_path = None
    config = load_config(config_path)
    if getattr(args, "seed", None) is not None:
        config["random_seed"] = int(args.seed)
    if getattr(args, "folds", None):
        config["cv_folds"] = int(args.folds)
    if demo:
        # The synthetic demo is a WIRING smoke test: its artifacts are marked
        # smoke and are never benchmarks, so a lower rare-class threshold is
        # acceptable HERE ONLY to exercise the full CV chain. Real data keeps
        # the strict default - collect data instead of lowering thresholds.
        user_min = None
        if config_path is not None:
            try:
                user_min = json.loads(config_path.read_text(encoding="utf-8")).get("min_class_samples")
            except Exception:
                user_min = None
        if user_min is None:
            config["min_class_samples"] = 3
        _ensure_demo_splits(paths["data_root"], int(config["random_seed"]))

    requested = [t.strip() for t in (args.targets or ",").split(",") if t.strip()]
    unknown = [t for t in requested if t not in TARGETS]
    if unknown:
        print(f"unknown target(s) {unknown}; valid: {sorted(TARGETS)}", file=sys.stderr)
        return 1

    results: Dict[str, object] = {}
    for target in requested:
        print("-" * 60)
        print(f"TARGET: {target}")
        print("-" * 60)
        try:
            data = prepare_training_data(
                paths["annotations"],
                paths["features"],
                target,
                data_root=paths["data_root"],
                source="demo" if demo else "real",
            )
        except FileNotFoundError as exc:
            print(f"  skipped: {exc}", file=sys.stderr)
            continue
        print(data.summary())
        save_report(data, paths["reports"] / f"training_data_{target}.json")
        if not data.rows:
            print("  no usable rows for this target - nothing to train (see exclusions above)")
            results[target] = {"skipped": "no usable rows"}
            continue

        result = run_training(data, config, paths["models"], smoke=smoke)
        results[target] = result
        cv = result.get("cv") or {}
        if cv.get("skipped"):
            print(f"  CV SKIPPED: {cv.get('reason')}")
            print("  DATA INSUFFICIENT for supervised training of this target")
            continue
        best = cv.get("best")
        best_mean = cv.get("candidates", {}).get(best, {}).get("mean")
        print(f"  best baseline: {best} (CV {config.get('selection_metric')}={best_mean:.3f})")
        if result.get("season_approaches"):
            for name, info in result["season_approaches"].items():
                mean = info.get("mean")
                text = f"{mean:.3f}" if isinstance(mean, float) else f"skipped: {info.get('reason')}"
                print(f"  season approach {name:<13} {text}")
            print(f"  selected approach: {result.get('selected_approach')}")
        print(f"  calibration: {result['calibration'].get('method')} ({result['calibration'].get('note', '')[:60]})")
        print(f"  artifact: {result['model_id']}  ({result['artifact_dir']})")
        if smoke:
            print("  [smoke artifact - wiring test, not a benchmark]")

    manifest = {
        "mode": "demo" if demo else "real",
        "smoke": smoke,
        "config": config,
        "results": {
            t: {k: r.get(k) for k in ("model_id", "selected_approach", "n_train_val_rows", "n_people") if isinstance(r, dict) and r.get(k) is not None}
            for t, r in results.items() if isinstance(r, dict)
        },
    }
    (paths["reports"] / "training_run.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print()
    print(f"run manifest: {paths['reports'] / 'training_run.json'}")
    return 0


def cmd_prepare(args) -> int:
    """Training-data preparation report only (no training)."""
    from .training_data import TARGETS, prepare_training_data, save_report

    root = Path(args.root)
    demo = bool(getattr(args, "demo", False))
    paths = _training_paths(root, demo)
    out_dir = Path(args.out) if getattr(args, "out", None) else paths["reports"]
    requested = [t.strip() for t in (args.targets or ",").split(",") if t.strip()]
    bad = [t for t in requested if t not in TARGETS]
    if bad:
        print(f"unknown target(s) {bad}; valid: {sorted(TARGETS)}", file=sys.stderr)
        return 1
    rc = 0
    for target in requested:
        print("=" * 60)
        print(f"{target} (source: {'demo' if demo else 'real'})")
        try:
            data = prepare_training_data(
                paths["annotations"],
                paths["features"],
                target,
                data_root=paths["data_root"],
                source="demo" if demo else "real",
            )
        except FileNotFoundError as exc:
            print(f"  {exc}", file=sys.stderr)
            rc = 1
            continue
        print(data.summary())
        save_report(data, out_dir / f"training_data_{target}.json")
    print(f"\nreports: {out_dir}/training_data_<target>.json")
    return rc


def cmd_evaluate(args) -> int:
    """Score the LOCKED holdout with a trained artifact (final, run once)."""
    from .evaluate_model import EvaluationError, load_predictor, render_holdout_report, run_holdout_evaluation
    from .training_data import prepare_training_data

    root = Path(args.root)
    demo = bool(getattr(args, "demo", False))
    paths = _training_paths(root, demo)

    if not demo:
        allowed, status, reasons = _gate_allows_training(root, bool(getattr(args, "smoke", False)))
        if not allowed:
            _print_training_blocked(status, reasons)
            return 1

    try:
        predictor, metadata = load_predictor(paths["models"], args.model)
        target = str(metadata.get("target"))
        data = prepare_training_data(
            paths["annotations"],
            paths["features"],
            target,
            data_root=paths["data_root"],
            source="demo" if demo else "real",
        )
        evaluated = run_holdout_evaluation(
            data,
            paths["models"],
            args.model,
            data_source_expected="demo" if demo else "real",
        )
    except (EvaluationError, FileNotFoundError) as exc:
        print(f"evaluation refused: {exc}", file=sys.stderr)
        return 1
    print(render_holdout_report(evaluated, target, demo=demo))
    report_path = paths["models"] / target / args.model / "holdout_report.txt"
    report_path.write_text(render_holdout_report(evaluated, target, demo=demo) + "\n", encoding="utf-8")
    print(f"\nreport saved: {report_path}")
    return 0





def cmd_demo_image(args) -> int:
    """Full loop on synthetic images: synth -> extract -> train -> evaluate."""
    from .demo import run_demo

    summary = run_demo(
        Path(args.root),
        n_people=args.people,
        images_per_person=args.images,
        seed=args.seed,
    )
    print(summary.render())
    print()
    print("artifacts: data/demo/bias_report.txt, data/demo/demo_summary.txt, data/demo/fallback_boxes.json")
    return 0


def cmd_audit(args) -> int:
    import json as _json

    from .audit import (
        audit_dataset,
        lock_holdout,
        verify_holdout,
        write_split_manifests,
    )

    root = Path(args.root)
    rows = _load_annotations(root)

    # A frozen holdout refuses implicit re-splitting; --force is the explicit,
    # logged human decision to break the lock and re-split.
    lock_path = root / "splits" / "holdout.lock.json"
    if (args.splits or args.regen_splits) and lock_path.exists() and not args.force:
        print(
            "REFUSING to re-split: a locked holdout exists (holdout.lock.json).\n"
            "The locked test split must stay untouched. To intentionally break the\n"
            "lock and re-split, re-run with --force (this is logged in the audit).",
            file=sys.stderr,
        )
        return 1

    if args.regen_splits or args.splits:
        if args.regen_splits:
            lock = lock_holdout(root, rows, test_frac=args.test_frac, seed=args.seed)
            print(f"holdout LOCKED: {lock['counts']}")
            print("the test split is now the frozen fairness holdout - never train on it")
        else:
            counts = write_split_manifests(root, rows, test_frac=args.test_frac, seed=args.seed)
            print(f"\nsplit manifests written to {root / 'splits'}: {counts}")
            print("consider `audit --regen-splits` to also freeze them with a holdout lock")
    elif lock_path.exists():
        status = verify_holdout(root)
        print(f"holdout lock: {'OK' if status['ok'] else 'BROKEN'} (locked {status.get('locked_at', '?')})")
        for problem in status.get("problems", []):
            print(f"  ! {problem}")

    result = audit_dataset(root, rows=rows)
    print(result.render())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_quality_report.txt").write_text(result.render() + "\n", encoding="utf-8")
    (out / "row_lint.json").write_text(
        _json.dumps(_lint_summary(root), indent=2) + "\n", encoding="utf-8"
    )
    return 0 if result.ok() else 1


def _lint_summary(root: Path) -> dict:
    """Machine-readable per-row lint results for the annotation UI."""
    from .validate import lint_rows, save_json

    report = lint_rows(_load_annotations(root))
    save_json(report, Path(root) / "reports" / "row_lint.json")
    return report.to_json()


def cmd_ingest(args) -> int:
    from .ingest import (
        IngestError,
        ingestion_plan,
        read_sources,
        record_download,
        render_plan,
        verify_receipt,
    )

    root = Path(args.root)
    try:
        if args.ingest_cmd == "plan":
            print(render_plan(ingestion_plan(root)))
            return 0
        if args.ingest_cmd == "record":
            receipt = record_download(
                root,
                args.dataset,
                Path(args.file),
                license_confirmed=args.license_confirmed,
            )
            print(f"receipt written: {receipt['dataset']} sha256={receipt['sha256']}")
            print(f"  archive: {receipt['archive']} ({receipt['bytes']} bytes)")
            return 0
        if args.ingest_cmd == "verify":
            status = verify_receipt(root, args.dataset, rehash=args.rehash)
            if status["ok"]:
                print(f"verify OK: {args.dataset} (sha256 {status.get('stored', '')[:16]}...)")
                return 0
            print(f"verify FAILED: {status.get('reason')}", file=sys.stderr)
            return 1
    except IngestError as exc:
        print(f"ingest error: {exc}", file=sys.stderr)
        return 1
    print(f"unknown ingest subcommand: {args.ingest_cmd}", file=sys.stderr)
    return 1


def cmd_validate(args) -> int:
    from .validate import lint_rows, save_json

    rows = _load_annotations(Path(args.root))
    report = lint_rows(rows)
    print(report.render())
    out = Path(args.out)
    save_json(report, out / "row_lint.json")
    print(f"\nJSON report: {out / 'row_lint.json'}")
    return 0 if report.ok else 1


def cmd_annot(args) -> int:
    from .annotation import (
        coverage,
        import_images,
        merge_rounds,
        pending_pool,
        queue_round,
        read_round,
        record_round,
        render_coverage,
        write_capture_manifest,
    )

    root = Path(args.root)
    if args.annot_cmd == "manifest":
        path = write_capture_manifest(root / "annotations" / "capture_manifest.csv")
        print(f"capture manifest template: {path}")
        print("fill one row per photo while shooting (see docs/ANNOTATION_HANDBOOK.md)")
        return 0
    if args.annot_cmd == "import":
        result = import_images(root, Path(args.manifest), Path(args.staging))
        print(
            f"imported {result['imported']} images | duplicate files skipped: "
            f"{result['skipped_duplicate_file']} | already imported: "
            f"{result['skipped_already_imported']}"
        )
        return 0
    if args.annot_cmd == "queue":
        ids = [line.strip() for line in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        path = queue_round(root, args.annotator, ids)
        print(f"round file for annotator {args.annotator!r}: {path} ({len(ids)} images)")
        return 0
    if args.annot_cmd == "record":
        result = record_round(root, Path(args.round_file), validate=not args.no_validate)
        print(f"accepted: {result['accepted']} | deferred (no season): {result['deferred']}")
        for problem in result["rejected"][:10]:
            print(f"  REJECTED {problem}")
        if result["rejected"][10:]:
            print(f"  ... {len(result['rejected']) - 10} more rejections")
        return 0 if not result["rejected"] else 1
    if args.annot_cmd == "merge":
        result = merge_rounds(root, min_annotators=args.min_annotators)
        print(f"merged images: {result['merged']} | still pending: {result['still_pending']}")
        for split in result["split_votes"][:10]:
            print(f"  SPLIT VOTE {split}")
        return 0
    if args.annot_cmd == "coverage":
        print(render_coverage(coverage(root)))
        return 0
    if args.annot_cmd == "pending":
        pool = pending_pool(root)
        print(f"{len(pool)} images awaiting annotation")
        for row in pool[:10]:
            print(f"  {row['image_id']}  person={row.get('person_id')}  raw/{row['image_id']}.jpg")
        return 0
    print(f"unknown annot subcommand: {args.annot_cmd}", file=sys.stderr)
    return 1


def cmd_ready(args) -> int:
    """Real-data readiness audit: the last checkpoint before ML training."""
    from .readiness import readiness, save_report

    root = Path(args.root)
    report = readiness(root)
    print(report.render())
    out = Path(args.out)
    save_report(report, out)
    print(f"\nreport files: {out / 'readiness_report.txt'} + .json")
    return {
        "NOT READY": 1,
        "SMOKE-READY": 2,
        "TRAIN-READY": 0,
    }[report.status]


def cmd_predict(args) -> int:
    """Single-image prediction: CV extraction -> trained models -> JSON to stdout.

    Used by the Express backend /api/color-analysis endpoint.
    If no trained models exist, uses rule-based heuristics ("model zero").
    """
    from pathlib import Path as _P

    from .colour import extract_image
    from .models import (
        ColourModelBundle,
        AxisModel,
        feature_matrix,
        predict_season_from_axes,
        season_from_axes,
    )

    image_path = _P(args.image)
    if not image_path.is_file():
        print(json.dumps({"ok": False, "error": f"Image not found: {image_path}"}), file=sys.stdout)
        return 1

    # 1. CV extraction
    ext = extract_image(image_path.stem, image_path)
    if not ext.ok:
        print(json.dumps({"ok": False, "error": ext.error or "extraction failed"}), file=sys.stdout)
        return 1

    features = ext.as_feature_row()

    # 2. Load trained models if available
    root = Path(args.root)
    model_dir = Path(args.model_dir) if args.model_dir else root / "models"
    # Fall back to demo models if real models dir is empty
    if not model_dir.is_dir() or not any(model_dir.iterdir()):
        demo_models = root / "demo" / "models"
        if demo_models.is_dir() and any(demo_models.iterdir()):
            model_dir = demo_models

    axes_predictions: dict = {}
    axes_probabilities: dict = {}
    model_used: dict = {}

    for axis in ("undertone", "skin_depth", "chroma", "contrast"):
        # Look for any trained model for this axis
        axis_dir = model_dir / axis
        if axis_dir.is_dir():
            model_files = sorted(axis_dir.glob("*/model.joblib"))
            if model_files:
                try:
                    import joblib
                    model_path = model_files[0]
                    predictor = joblib.load(model_path)
                    # SeasonPredictor has predict_row; raw pipeline has predict_proba
                    if hasattr(predictor, "predict_row"):
                        result = predictor.predict_row(features)
                        axes_predictions[axis] = result["prediction"]
                        axes_probabilities[axis] = result.get("probabilities", {result["prediction"]: 1.0})
                    else:
                        X = feature_matrix([features])
                        proba = predictor.predict_proba(X)[0]
                        classes = [str(c) for c in predictor.classes_]
                        probs = {c: float(p) for c, p in zip(classes, proba)}
                        axes_predictions[axis] = max(probs, key=probs.get)
                        axes_probabilities[axis] = probs
                    model_used[axis] = model_path.parent.name
                    continue
                except Exception:
                    pass  # fall through to heuristic

        # Heuristic fallback
        from .models import HEURISTICS
        pred, probs = AxisModel(axis=axis).predict(features)
        axes_predictions[axis] = pred
        axes_probabilities[axis] = probs
        model_used[axis] = "heuristic"

    # 3. Season from axes
    season_primary, season_secondary, season_confidence = predict_season_from_axes(
        axes_predictions, axes_probabilities
    )

    # 4. Build response
    result = {
        "ok": True,
        "season": {
            "primary": season_primary,
            "secondary": season_secondary,
            "confidence": round(season_confidence, 3),
        },
        "axes": {
            "undertone": {"prediction": axes_predictions.get("undertone", ""), "probabilities": axes_probabilities.get("undertone", {}), "model": model_used.get("undertone", "")},
            "skin_depth": {"prediction": axes_predictions.get("skin_depth", ""), "probabilities": axes_probabilities.get("skin_depth", {}), "model": model_used.get("skin_depth", "")},
            "chroma": {"prediction": axes_predictions.get("chroma", ""), "probabilities": axes_probabilities.get("chroma", {}), "model": model_used.get("chroma", "")},
            "contrast": {"prediction": axes_predictions.get("contrast", ""), "probabilities": axes_probabilities.get("contrast", {}), "model": model_used.get("contrast", "")},
        },
        "features": {
            "skin_L": round(features.get("feat_skin_L", 0), 1),
            "skin_a": round(features.get("feat_skin_a", 0), 1),
            "skin_b": round(features.get("feat_skin_b", 0), 1),
            "ita": round(features.get("feat_ita", 0), 1),
            "hair_L": round(features.get("feat_hair_L", 0), 1),
            "eye_L": round(features.get("feat_eye_L", 0), 1),
            "contrast_proxy": round(features.get("feat_contrast_proxy", 0), 1),
            "brightness": features.get("feat_exposure_flag", "ok"),
            "skin_pixels": features.get("feat_skin_pixels", 0),
        },
    }

    print(json.dumps(result, indent=2))
    return 0


def cmd_gate(args) -> int:
    """The ML-training gate: OPEN only when real annotated data is audit-clean."""
    import json as _json

    from .annotation import coverage, render_coverage
    from .audit import audit_dataset, verify_holdout
    from .dataset import init_dataset
    from .validate import lint_rows

    root = Path(args.root)
    init_dataset(root)  # empty template is a valid (closed) gate state
    rows = _load_annotations(root)
    cov = coverage(root)
    lint = lint_rows(rows)
    audit = audit_dataset(root, rows=rows)
    holdout = verify_holdout(root)

    blocked: List[str] = []
    if cov["images_annotated"] < 40:
        blocked.append(
            f"only {cov['images_annotated']} annotated images (< 40); acquisition/annotation phase"
        )
    if cov["people"] < 20:
        blocked.append(f"only {cov['people']} people (< 20); person-grouped splits need people, not images")
    if lint.total_rows and not lint.ok:
        blocked.append(f"row lint: {lint.total_rows - lint.valid_rows} invalid rows (run `validate`)")
    if not audit.ok():
        blocked.append("data audit FAIL (see `audit` report)")
    if not holdout.get("locked") or not holdout.get("ok"):
        blocked.append("holdout not locked (run `audit --regen-splits` once the audit passes)")

    gate = {
        "gate": "ML_TRAINING",
        "status": "OPEN" if not blocked else "CLOSED",
        "reasons": blocked,
        "coverage": cov,
        "holdout": {"locked": holdout.get("locked", False), "ok": holdout.get("ok", False)},
        "audit_ok": audit.ok(),
    }
    print(
        "=" * 46,
        "FACEMAXX ML-TRAINING GATE",
        "=" * 46,
        sep="\n",
    )
    print(f"status: {gate['status']}")
    for reason in blocked:
        print(f"  BLOCKED: {reason}")
    if not blocked:
        print("real annotated data is audit-clean, leakage-free, holdout locked.")
        print("Baseline training (`train`/`evaluate`) is now justified.")
    print()
    print(render_coverage(cov))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ml_gate.json").write_text(_json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(f"\ngate state saved: {out / 'ml_gate.json'}")
    return 0 if not blocked else 1


def _load_features(root: Path) -> dict:
    """image_id -> feature dict from processed/colour_features.csv."""
    path = root / "processed" / "colour_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"run `python -m facemax_color extract` first ({path} missing)")
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["image_id"]: row for row in csv.DictReader(fh)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="facemax_color", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-data", help="create dataset tree + annotation template")
    p.add_argument("--root", default="data")
    p.set_defaults(func=cmd_init_data)

    p = sub.add_parser("extract", help="run colour extraction on all annotated images")
    p.add_argument("--root", default="data")
    p.add_argument("--images", default="data/raw")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser(
        "prepare",
        help="training-data preparation report (join + exclusions + class balance)",
    )
    p.add_argument("--root", default="data")
    p.add_argument("--targets", default="season,undertone,skin_depth,chroma,contrast")
    p.add_argument("--out", default=None, help="report dir (default: data/reports or demo dir)")
    p.add_argument("--demo", action="store_true", help="operate on the isolated demo tree only")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser(
        "train",
        help="gated supervised training (readiness-checked; --demo for the isolated synthetic tree)",
    )
    p.add_argument("--root", default="data")
    p.add_argument("--targets", default="season,undertone,skin_depth,chroma,contrast")
    p.add_argument("--config", default=None, help="training config JSON (default: built-in defaults)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="allow SMOKE-READY data (artifacts are marked smoke: wiring test, not a benchmark)",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="train on the ISOLATED demo tree (data/demo) - synthetic smoke test only",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser(
        "evaluate",
        help="score the LOCKED holdout with a trained artifact (final; never used for tuning)",
    )
    p.add_argument("--root", default="data")
    p.add_argument("--model", required=True, help="model_id from the registry (e.g. season-random_forest-v001)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--demo", action="store_true", help="evaluate a demo artifact on the demo holdout")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("demo-image", help="synthetic image dataset + full end-to-end run")
    p.add_argument("--root", default="data")
    p.add_argument("--people", type=int, default=40)
    p.add_argument("--images", type=int, default=2)
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=cmd_demo_image)

    p = sub.add_parser("audit", help="data-quality audit + optional split manifests")
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="data/reports")
    p.add_argument("--splits", action="store_true", help="write person-grouped split manifests")
    p.add_argument(
        "--regen-splits",
        action="store_true",
        help="re-split AND lock the holdout (refused if a lock exists without --force)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="explicitly break an existing holdout lock (logged; use sparingly)",
    )
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("validate", help="per-row annotation lint (vocab/ranges/required)")
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="data/reports")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("ingest", help="external dataset ingestion (plan/record/verify)")
    ing = ing_sub = p.add_subparsers(dest="ingest_cmd", required=True)
    p_plan = ing_sub.add_parser("plan", help="show official sources, license gates, status")
    p_plan.add_argument("--root", default="data")
    p_plan.set_defaults(func=cmd_ingest)
    p_rec = ing_sub.add_parser("record", help="register a human-made download (sha256 receipt)")
    p_rec.add_argument("--root", default="data")
    p_rec.add_argument("--dataset", required=True)
    p_rec.add_argument("--file", required=True, help="path to the downloaded archive")
    p_rec.add_argument(
        "--license-confirmed",
        action="store_true",
        help="certify a human reviewed the official license terms",
    )
    p_rec.set_defaults(func=cmd_ingest)
    p_ver = ing_sub.add_parser("verify", help="verify a recorded download (optionally re-hash)")
    p_ver.add_argument("--root", default="data")
    p_ver.add_argument("--dataset", required=True)
    p_ver.add_argument("--rehash", action="store_true", help="re-hash the archive now")
    p_ver.set_defaults(func=cmd_ingest)

    p = sub.add_parser("annot", help="annotation workflow (import/queue/record/merge/coverage)")
    annot_sub = p.add_subparsers(dest="annot_cmd", required=True)
    p_man = annot_sub.add_parser("manifest", help="write capture-manifest template")
    p_man.add_argument("--root", default="data")
    p_man.set_defaults(func=cmd_annot)
    p_imp = annot_sub.add_parser("import", help="import staged photos into data/raw (content-deduped)")
    p_imp.add_argument("--root", default="data")
    p_imp.add_argument("--manifest", required=True, help="capture_manifest.csv path")
    p_imp.add_argument("--staging", required=True, help="folder holding the photo files")
    p_imp.set_defaults(func=cmd_annot)
    p_que = annot_sub.add_parser("queue", help="write a per-annotator round file")
    p_que.add_argument("--root", default="data")
    p_que.add_argument("--annotator", required=True)
    p_que.add_argument("--ids-file", required=True, help="text file with image_ids, one per line")
    p_que.set_defaults(func=cmd_annot)
    p_rec = annot_sub.add_parser("record", help="validate a filled round into the pending pool")
    p_rec.add_argument("--root", default="data")
    p_rec.add_argument("--round-file", required=True)
    p_rec.add_argument("--no-validate", action="store_true")
    p_rec.set_defaults(func=cmd_annot)
    p_mrg = annot_sub.add_parser("merge", help="merge pending rounds into the annotation table")
    p_mrg.add_argument("--root", default="data")
    p_mrg.add_argument("--min-annotators", type=int, default=2)
    p_mrg.set_defaults(func=cmd_annot)
    p_cov = annot_sub.add_parser("coverage", help="acquisition/annotation progress vs. targets")
    p_cov.add_argument("--root", default="data")
    p_cov.set_defaults(func=cmd_annot)
    p_pen = annot_sub.add_parser("pending", help="list images awaiting annotation")
    p_pen.add_argument("--root", default="data")
    p_pen.set_defaults(func=cmd_annot)

    p = sub.add_parser(
        "ready", help="data-readiness audit: is the real dataset usable for training?"
    )
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="data/reports")
    p.set_defaults(func=cmd_ready)

    p = sub.add_parser("gate", help="ML-training gate: OPEN only when real data passes every check")
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="data/reports")
    p.set_defaults(func=cmd_gate)

    # --- predict (single-image colour analysis for API integration) ---
    p = sub.add_parser("predict", help="Run CV extraction + model prediction on a single image (JSON stdout)")
    p.add_argument("image", help="Path to image file (JPEG/PNG)")
    p.add_argument("--root", default="data", help="Dataset root (for model lookup)")
    p.add_argument("--model-dir", default=None, help="Explicit model directory (overrides root)")
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
