from __future__ import annotations

import argparse
from pathlib import Path

from galaxy_zoo_project.correlation_analysis import CorrelationAnalysisConfig, run_correlation_analysis


def resolve_project_path(project_root: Path, path: Path | None, default: Path) -> Path:
    selected = path or default
    if selected.is_absolute():
        return selected
    return project_root / selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether reconstruction fidelity metrics align with downstream task performance."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--degraded-manifest-path", type=Path, default=None)
    parser.add_argument("--reconstruction-manifest-path", type=Path, default=None)
    parser.add_argument("--predictions-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    config = CorrelationAnalysisConfig(
        project_root=project_root,
        degraded_manifest_path=resolve_project_path(
            project_root,
            args.degraded_manifest_path,
            project_root / "data" / "degraded" / "galaxy_zoo_128_moderate_eval" / "manifest.csv",
        ),
        reconstruction_manifest_path=resolve_project_path(
            project_root,
            args.reconstruction_manifest_path,
            project_root / "data" / "reconstructed" / "galaxy_zoo_128_eval_baselines" / "manifest.csv",
        ),
        predictions_path=resolve_project_path(
            project_root,
            args.predictions_path,
            project_root / "data" / "ml_baselines" / "class1_logistic_rgb32_eval" / "predictions.csv",
        ),
        output_dir=resolve_project_path(
            project_root,
            args.output_dir,
            project_root / "data" / "analysis" / "reconstruction_task_correlation",
        ),
    )

    summary = run_correlation_analysis(config)
    print(f"Analyzed {summary['num_reconstructed_rows']:,} reconstructed image-method rows")
    print(f"Per-image analysis: {summary['outputs']['per_image_analysis']}")
    print(f"Task metrics: {summary['outputs']['task_metrics']}")
    print(f"Correlations: {summary['outputs']['correlations']}")
    print(f"Delta summary: {summary['outputs']['delta_summary']}")


if __name__ == "__main__":
    main()
