from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class CorrelationAnalysisConfig:
    project_root: Path
    degraded_manifest_path: Path
    reconstruction_manifest_path: Path
    predictions_path: Path
    output_dir: Path


def _relative_to_project(project_root: Path, path: Path) -> str:
    return str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path)


def _pearson(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    return float(x.corr(y, method="pearson"))


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    return float(x.corr(y, method="spearman"))


def _load_required_csv(path: Path, required_columns: set[str], name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    frame = pd.read_csv(path)
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing required columns: {sorted(missing_columns)}")
    return frame


def _task_metrics(frame: pd.DataFrame) -> dict[str, float | int | str]:
    return {
        "condition": str(frame["condition"].iloc[0]),
        "method": str(frame["method"].iloc[0]),
        "split": str(frame["split"].iloc[0]),
        "n_samples": int(len(frame)),
        "accuracy": float(accuracy_score(frame["target"], frame["prediction"])),
        "macro_f1": float(f1_score(frame["target"], frame["prediction"], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(frame["target"], frame["prediction"], average="weighted", zero_division=0)),
    }


def _join_reconstruction_predictions(reconstruction: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    reconstructed_predictions = predictions[predictions["condition"] == "reconstructed"].copy()
    joined = reconstruction.merge(
        reconstructed_predictions,
        on=["GalaxyID", "method"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_prediction"),
    )
    if joined.empty:
        raise ValueError("No reconstructed prediction rows matched reconstruction metrics.")
    joined["correct"] = joined["target"] == joined["prediction"]
    joined["correct_int"] = joined["correct"].astype(int)
    return joined


def _join_degraded_predictions(degraded: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    degraded_predictions = predictions[predictions["condition"] == "degraded"].copy()
    joined = degraded.merge(
        degraded_predictions,
        on=["GalaxyID", "split"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_prediction"),
    )
    if joined.empty:
        raise ValueError("No degraded prediction rows matched degraded metrics.")
    joined["condition"] = "degraded"
    joined["method"] = "degraded"
    joined["correct"] = joined["target"] == joined["prediction"]
    joined["correct_int"] = joined["correct"].astype(int)
    return joined


def _correlation_records(frame: pd.DataFrame, metric_columns: tuple[str, ...]) -> list[dict[str, float | int | str]]:
    records = []
    for (split, method), group in frame.groupby(["split", "method"], sort=True):
        for metric in metric_columns:
            records.append(
                {
                    "split": split,
                    "method": method,
                    "metric": metric,
                    "n_samples": int(len(group)),
                    "pearson_with_correct": _pearson(group[metric], group["correct_int"]),
                    "spearman_with_correct": _spearman(group[metric], group["correct_int"]),
                    "mean_metric_when_correct": float(group.loc[group["correct"], metric].mean()),
                    "mean_metric_when_incorrect": float(group.loc[~group["correct"], metric].mean()),
                }
            )
    return records


def _add_reference_deltas(reconstructed: pd.DataFrame, degraded: pd.DataFrame) -> pd.DataFrame:
    degraded_reference = degraded[["GalaxyID", "psnr", "ssim", "correct_int"]].rename(
        columns={
            "psnr": "degraded_psnr",
            "ssim": "degraded_ssim",
            "correct_int": "degraded_correct_int",
        }
    )
    with_deltas = reconstructed.merge(degraded_reference, on="GalaxyID", how="left", validate="many_to_one")
    with_deltas["psnr_delta_vs_degraded"] = with_deltas["psnr"] - with_deltas["degraded_psnr"]
    with_deltas["ssim_delta_vs_degraded"] = with_deltas["ssim"] - with_deltas["degraded_ssim"]
    with_deltas["correct_delta_vs_degraded"] = with_deltas["correct_int"] - with_deltas["degraded_correct_int"]

    identity_reference = reconstructed[reconstructed["method"] == "identity"][
        ["GalaxyID", "psnr", "ssim", "correct_int"]
    ].rename(
        columns={
            "psnr": "identity_psnr",
            "ssim": "identity_ssim",
            "correct_int": "identity_correct_int",
        }
    )
    with_deltas = with_deltas.merge(identity_reference, on="GalaxyID", how="left", validate="many_to_one")
    with_deltas["psnr_delta_vs_identity"] = with_deltas["psnr"] - with_deltas["identity_psnr"]
    with_deltas["ssim_delta_vs_identity"] = with_deltas["ssim"] - with_deltas["identity_ssim"]
    with_deltas["correct_delta_vs_identity"] = with_deltas["correct_int"] - with_deltas["identity_correct_int"]
    return with_deltas


def run_correlation_analysis(config: CorrelationAnalysisConfig) -> dict[str, object]:
    degraded = _load_required_csv(
        config.degraded_manifest_path,
        {"GalaxyID", "split", "psnr", "ssim"},
        "Degraded manifest",
    )
    reconstruction = _load_required_csv(
        config.reconstruction_manifest_path,
        {"GalaxyID", "method", "psnr", "ssim"},
        "Reconstruction manifest",
    )
    predictions = _load_required_csv(
        config.predictions_path,
        {"GalaxyID", "condition", "method", "split", "target", "prediction"},
        "Prediction file",
    )

    degraded_joined = _join_degraded_predictions(degraded, predictions)
    reconstructed_joined = _join_reconstruction_predictions(reconstruction, predictions)
    reconstructed_with_deltas = _add_reference_deltas(reconstructed_joined, degraded_joined)

    task_metrics = pd.DataFrame(
        [
            _task_metrics(group)
            for _, group in predictions.groupby(["condition", "method", "split"], sort=True)
        ]
    ).sort_values(["split", "condition", "method"])

    correlation_frame = pd.DataFrame(
        _correlation_records(reconstructed_with_deltas, ("psnr", "ssim"))
        + _correlation_records(
            reconstructed_with_deltas,
            ("psnr_delta_vs_degraded", "ssim_delta_vs_degraded", "psnr_delta_vs_identity", "ssim_delta_vs_identity"),
        )
    )

    delta_summary = (
        reconstructed_with_deltas.groupby(["split", "method"], sort=True)[
            [
                "psnr",
                "ssim",
                "psnr_delta_vs_degraded",
                "ssim_delta_vs_degraded",
                "psnr_delta_vs_identity",
                "ssim_delta_vs_identity",
                "correct_delta_vs_degraded",
                "correct_delta_vs_identity",
            ]
        ]
        .mean()
        .reset_index()
    )

    per_image_path = config.output_dir / "per_image_analysis.csv"
    task_metrics_path = config.output_dir / "task_metrics.csv"
    correlations_path = config.output_dir / "correlations.csv"
    delta_summary_path = config.output_dir / "delta_summary.csv"
    summary_path = config.output_dir / "summary.json"

    config.output_dir.mkdir(parents=True, exist_ok=True)
    reconstructed_with_deltas.to_csv(per_image_path, index=False)
    task_metrics.to_csv(task_metrics_path, index=False)
    correlation_frame.to_csv(correlations_path, index=False)
    delta_summary.to_csv(delta_summary_path, index=False)

    strongest_abs = correlation_frame.copy()
    strongest_abs["abs_spearman_with_correct"] = strongest_abs["spearman_with_correct"].abs()
    strongest_abs = strongest_abs.sort_values("abs_spearman_with_correct", ascending=False)

    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "num_degraded_rows": int(len(degraded_joined)),
        "num_reconstructed_rows": int(len(reconstructed_with_deltas)),
        "outputs": {
            "per_image_analysis": _relative_to_project(config.project_root, per_image_path),
            "task_metrics": _relative_to_project(config.project_root, task_metrics_path),
            "correlations": _relative_to_project(config.project_root, correlations_path),
            "delta_summary": _relative_to_project(config.project_root, delta_summary_path),
        },
        "strongest_spearman_correlations": strongest_abs.head(10).to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
