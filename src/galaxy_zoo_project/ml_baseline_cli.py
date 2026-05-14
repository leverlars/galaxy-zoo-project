from __future__ import annotations

import argparse
from pathlib import Path

from galaxy_zoo_project.ml_baseline import FEATURE_MODES, MLBaselineConfig, run_ml_baseline


def resolve_project_path(project_root: Path, path: Path | None, default: Path) -> Path:
    selected = path or default
    if selected.is_absolute():
        return selected
    return project_root / selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a manifest-driven Galaxy Zoo ML baseline.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--processed-manifest-path", type=Path, default=None)
    parser.add_argument("--degraded-manifest-path", type=Path, default=None)
    parser.add_argument("--reconstruction-manifest-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test"])
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="rgb")
    parser.add_argument("--feature-size", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    processed_manifest_path = resolve_project_path(
        project_root,
        args.processed_manifest_path,
        project_root / "data" / "processed" / "galaxy_zoo_128" / "manifest.csv",
    )
    degraded_manifest_path = resolve_project_path(
        project_root,
        args.degraded_manifest_path,
        project_root / "data" / "degraded" / "galaxy_zoo_128_moderate" / "manifest.csv",
    )
    reconstruction_manifest_path = resolve_project_path(
        project_root,
        args.reconstruction_manifest_path,
        project_root / "data" / "reconstructed" / "galaxy_zoo_128_baselines" / "manifest.csv",
    )
    default_output_name = f"class1_logistic_{args.feature_mode}{args.feature_size}"
    output_dir = resolve_project_path(
        project_root,
        args.output_dir,
        project_root / "data" / "ml_baselines" / default_output_name,
    )

    config = MLBaselineConfig(
        project_root=project_root,
        processed_manifest_path=processed_manifest_path,
        degraded_manifest_path=degraded_manifest_path,
        reconstruction_manifest_path=reconstruction_manifest_path,
        output_dir=output_dir,
        train_split=args.train_split,
        eval_splits=tuple(args.eval_splits),
        feature_mode=args.feature_mode,
        feature_size=args.feature_size,
        max_iter=args.max_iter,
        seed=args.seed,
        train_limit=args.train_limit,
    )

    summary = run_ml_baseline(config)
    print(f"Trained on {summary['num_train_images']:,} clean images")
    print(f"Metrics: {summary['metrics_path']}")
    print(f"Predictions: {summary['predictions_path']}")
    print(f"Model: {summary['model_path']}")


if __name__ == "__main__":
    main()
