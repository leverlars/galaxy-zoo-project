from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2gray
from skimage.feature import hog
from tqdm.auto import tqdm


CLASS1_TARGET_COLUMNS = ("Class1.1", "Class1.2", "Class1.3")
FEATURE_MODES = ("rgb", "hog", "color_stats", "hog_color")
CLASS1_LABELS = {
    "Class1.1": "smooth",
    "Class1.2": "features_or_disk",
    "Class1.3": "star_or_artifact",
}


@dataclass(frozen=True)
class MLBaselineConfig:
    project_root: Path
    processed_manifest_path: Path
    output_dir: Path
    degraded_manifest_path: Path | None = None
    reconstruction_manifest_path: Path | None = None
    target_columns: tuple[str, ...] = CLASS1_TARGET_COLUMNS
    train_split: str = "train"
    eval_splits: tuple[str, ...] = ("val", "test")
    feature_mode: str = "rgb"
    feature_size: int = 32
    max_iter: int = 500
    seed: int = 42
    train_limit: int | None = None


def _resolve_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def color_stats_features(image: np.ndarray) -> np.ndarray:
    channel_features = []
    for channel in range(image.shape[-1]):
        values = image[:, :, channel].reshape(-1)
        histogram, _ = np.histogram(values, bins=16, range=(0.0, 1.0), density=True)
        channel_features.extend(
            [
                float(values.mean()),
                float(values.std()),
                float(np.quantile(values, 0.1)),
                float(np.quantile(values, 0.5)),
                float(np.quantile(values, 0.9)),
                *histogram.astype(float).tolist(),
            ]
        )
    return np.asarray(channel_features, dtype=np.float32)


def extract_image_features(image: np.ndarray, *, feature_mode: str) -> np.ndarray:
    if feature_mode == "rgb":
        return image.reshape(-1).astype(np.float32)
    if feature_mode == "color_stats":
        return color_stats_features(image)
    if feature_mode in {"hog", "hog_color"}:
        gray = rgb2gray(image)
        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)
        if feature_mode == "hog":
            return hog_features
        return np.concatenate([hog_features, color_stats_features(image)]).astype(np.float32)
    raise ValueError(f"Unknown feature mode: {feature_mode}. Expected one of {FEATURE_MODES}.")


def load_image_features(
    project_root: Path,
    image_paths: Iterable[str | Path],
    *,
    feature_size: int,
    feature_mode: str,
) -> np.ndarray:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature mode: {feature_mode}. Expected one of {FEATURE_MODES}.")

    features = []
    paths = list(image_paths)
    for path_value in tqdm(paths, desc="Loading image features", unit="image"):
        image_path = _resolve_path(project_root, path_value)
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((feature_size, feature_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        features.append(extract_image_features(array, feature_mode=feature_mode))

    if not features:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(features).astype(np.float32)


def class1_targets(manifest: pd.DataFrame, target_columns: tuple[str, ...]) -> pd.Series:
    missing_columns = set(target_columns) - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing target columns: {sorted(missing_columns)}")

    hard_labels = manifest.loc[:, list(target_columns)].idxmax(axis=1)
    return hard_labels.map(CLASS1_LABELS).astype("category")


def _label_frame(processed_manifest: pd.DataFrame, target_columns: tuple[str, ...]) -> pd.DataFrame:
    required_columns = {"GalaxyID", "split", *target_columns}
    missing_columns = required_columns - set(processed_manifest.columns)
    if missing_columns:
        raise ValueError(f"Processed manifest is missing required columns: {sorted(missing_columns)}")
    return processed_manifest[["GalaxyID", "split", *target_columns]].copy()


def _clean_condition_frame(processed_manifest: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"GalaxyID", "split", "processed_path"}
    missing_columns = required_columns - set(processed_manifest.columns)
    if missing_columns:
        raise ValueError(f"Processed manifest is missing required columns: {sorted(missing_columns)}")

    frame = processed_manifest.copy()
    frame["condition"] = "clean"
    frame["method"] = "clean"
    frame["image_path"] = frame["processed_path"]
    return frame


def _degraded_condition_frame(
    degraded_manifest_path: Path | None,
    processed_manifest: pd.DataFrame,
    target_columns: tuple[str, ...],
) -> pd.DataFrame:
    if degraded_manifest_path is None or not degraded_manifest_path.exists():
        return pd.DataFrame()

    degraded_manifest = pd.read_csv(degraded_manifest_path)
    required_columns = {"GalaxyID", "degraded_path"}
    missing_columns = required_columns - set(degraded_manifest.columns)
    if missing_columns:
        raise ValueError(f"Degraded manifest is missing required columns: {sorted(missing_columns)}")

    frame = degraded_manifest[["GalaxyID", "degraded_path"]].merge(
        _label_frame(processed_manifest, target_columns),
        on="GalaxyID",
        how="left",
        validate="many_to_one",
    )
    frame["condition"] = "degraded"
    frame["method"] = "degraded"
    frame["image_path"] = frame["degraded_path"]
    return frame


def _reconstructed_condition_frame(
    reconstruction_manifest_path: Path | None,
    processed_manifest: pd.DataFrame,
    target_columns: tuple[str, ...],
) -> pd.DataFrame:
    if reconstruction_manifest_path is None or not reconstruction_manifest_path.exists():
        return pd.DataFrame()

    reconstruction_manifest = pd.read_csv(reconstruction_manifest_path)
    required_columns = {"GalaxyID", "method", "reconstructed_path"}
    missing_columns = required_columns - set(reconstruction_manifest.columns)
    if missing_columns:
        raise ValueError(f"Reconstruction manifest is missing required columns: {sorted(missing_columns)}")

    frame = reconstruction_manifest[["GalaxyID", "method", "reconstructed_path"]].merge(
        _label_frame(processed_manifest, target_columns),
        on="GalaxyID",
        how="left",
        validate="many_to_one",
    )
    frame["condition"] = "reconstructed"
    frame["image_path"] = frame["reconstructed_path"]
    return frame


def _build_model(config: MLBaselineConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=config.max_iter,
                    class_weight="balanced",
                    random_state=config.seed,
                ),
            ),
        ]
    )


def _evaluate_frame(
    model: Pipeline,
    frame: pd.DataFrame,
    *,
    project_root: Path,
    target_columns: tuple[str, ...],
    feature_size: int,
    feature_mode: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    y_true = class1_targets(frame, target_columns)
    features = load_image_features(
        project_root,
        frame["image_path"],
        feature_size=feature_size,
        feature_mode=feature_mode,
    )
    y_pred = model.predict(features)

    metrics = {
        "condition": str(frame["condition"].iloc[0]),
        "method": str(frame["method"].iloc[0]),
        "split": str(frame["split"].iloc[0]),
        "n_samples": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    predictions = frame[["GalaxyID", "condition", "method", "split", "image_path"]].copy()
    predictions["target"] = y_true.astype(str).to_numpy()
    predictions["prediction"] = y_pred
    return metrics, predictions


def run_ml_baseline(config: MLBaselineConfig) -> dict[str, object]:
    if not config.processed_manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest not found: {config.processed_manifest_path}")

    processed_manifest = pd.read_csv(config.processed_manifest_path)
    clean_frame = _clean_condition_frame(processed_manifest)
    train_frame = clean_frame[clean_frame["split"] == config.train_split].copy()
    if config.train_limit is not None:
        train_frame = train_frame.head(config.train_limit).copy()
    if train_frame.empty:
        raise ValueError(f"No training images found for split: {config.train_split}")

    y_train = class1_targets(train_frame, config.target_columns)
    x_train = load_image_features(
        config.project_root,
        train_frame["image_path"],
        feature_size=config.feature_size,
        feature_mode=config.feature_mode,
    )
    model = _build_model(config)
    model.fit(x_train, y_train)

    condition_frames = [
        clean_frame,
        _degraded_condition_frame(config.degraded_manifest_path, processed_manifest, config.target_columns),
        _reconstructed_condition_frame(config.reconstruction_manifest_path, processed_manifest, config.target_columns),
    ]
    eval_frame = pd.concat([frame for frame in condition_frames if not frame.empty], ignore_index=True)
    eval_frame = eval_frame[eval_frame["split"].isin(config.eval_splits)].copy()
    if eval_frame.empty:
        raise ValueError(f"No evaluation images found for splits: {list(config.eval_splits)}")

    metric_records = []
    prediction_frames = []
    for (_, _, _), group in eval_frame.groupby(["condition", "method", "split"], sort=True):
        metrics, predictions = _evaluate_frame(
            model,
            group,
            project_root=config.project_root,
            target_columns=config.target_columns,
            feature_size=config.feature_size,
            feature_mode=config.feature_mode,
        )
        metric_records.append(metrics)
        prediction_frames.append(predictions)

    metrics_frame = pd.DataFrame(metric_records).sort_values(["split", "condition", "method"])
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "metrics.csv"
    predictions_path = config.output_dir / "predictions.csv"
    model_path = config.output_dir / "model.pkl"
    summary_path = config.output_dir / "summary.json"

    metrics_frame.to_csv(metrics_path, index=False)
    predictions_frame.to_csv(predictions_path, index=False)
    with model_path.open("wb") as file:
        pickle.dump(model, file)

    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "target": {
            "name": "class1_morphology_argmax",
            "columns": list(config.target_columns),
            "labels": CLASS1_LABELS,
        },
        "num_train_images": int(len(train_frame)),
        "train_class_counts": y_train.value_counts().sort_index().astype(int).to_dict(),
        "metrics_path": str(metrics_path.relative_to(config.project_root)),
        "predictions_path": str(predictions_path.relative_to(config.project_root)),
        "model_path": str(model_path.relative_to(config.project_root)),
        "metrics": metrics_frame.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
