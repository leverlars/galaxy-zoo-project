from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import gaussian
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from tqdm.auto import tqdm


@dataclass(frozen=True)
class DegradationConfig:
    project_root: Path
    processed_manifest_path: Path
    output_dir: Path
    blur_sigma: float = 1.2
    downsample_factor: int = 2
    gaussian_noise_std: float = 0.03
    poisson_peak: float | None = None
    limit: int | None = None
    split: str | tuple[str, ...] | None = None
    seed: int = 42
    overwrite: bool = False


def load_rgb_float(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return array


def save_rgb_float(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(array, 0.0, 1.0)
    image = Image.fromarray((clipped * 255).round().astype(np.uint8), mode="RGB")
    image.save(path, quality=95)


def degrade_array(
    clean: np.ndarray,
    *,
    blur_sigma: float,
    downsample_factor: int,
    gaussian_noise_std: float,
    poisson_peak: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if clean.ndim != 3 or clean.shape[-1] != 3:
        raise ValueError(f"Expected RGB image array with shape HxWx3, got {clean.shape}")
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be at least 1.")

    degraded = clean.astype(np.float32, copy=True)

    if blur_sigma > 0:
        degraded = gaussian(
            degraded,
            sigma=blur_sigma,
            channel_axis=-1,
            preserve_range=True,
        ).astype(np.float32)

    if downsample_factor > 1:
        height, width = degraded.shape[:2]
        low_height = max(1, height // downsample_factor)
        low_width = max(1, width // downsample_factor)
        low_res = resize(
            degraded,
            (low_height, low_width, 3),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
        degraded = resize(
            low_res,
            (height, width, 3),
            order=1,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        ).astype(np.float32)

    if poisson_peak is not None and poisson_peak > 0:
        degraded = rng.poisson(np.clip(degraded, 0.0, 1.0) * poisson_peak) / poisson_peak

    if gaussian_noise_std > 0:
        degraded = degraded + rng.normal(0.0, gaussian_noise_std, size=degraded.shape)

    return np.clip(degraded, 0.0, 1.0).astype(np.float32)


def image_quality_metrics(clean: np.ndarray, degraded: np.ndarray) -> dict[str, float]:
    return {
        "psnr": float(peak_signal_noise_ratio(clean, degraded, data_range=1.0)),
        "ssim": float(
            structural_similarity(
                clean,
                degraded,
                channel_axis=-1,
                data_range=1.0,
            )
        ),
    }


def build_degraded_dataset(config: DegradationConfig) -> pd.DataFrame:
    if not config.processed_manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest not found: {config.processed_manifest_path}")

    manifest = pd.read_csv(config.processed_manifest_path)
    if "processed_path" not in manifest.columns or "GalaxyID" not in manifest.columns:
        raise ValueError("Processed manifest must contain GalaxyID and processed_path columns.")

    if config.split is not None:
        if "split" not in manifest.columns:
            raise ValueError("Cannot filter by split because the manifest has no split column.")
        selected_splits = (config.split,) if isinstance(config.split, str) else tuple(config.split)
        manifest = manifest[manifest["split"].isin(selected_splits)].copy()

    if config.limit is not None:
        manifest = manifest.head(config.limit).copy()

    if manifest.empty:
        raise ValueError("No images selected for degradation.")

    records = []
    images_dir = config.output_dir / "images"

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Degrading images", unit="image"):
        galaxy_id = int(row.GalaxyID)
        clean_path = config.project_root / row.processed_path
        degraded_path = images_dir / f"{galaxy_id}.jpg"
        relative_degraded_path = degraded_path.relative_to(config.project_root)

        if degraded_path.exists() and not config.overwrite:
            clean = load_rgb_float(clean_path)
            degraded = load_rgb_float(degraded_path)
        else:
            clean = load_rgb_float(clean_path)
            rng = np.random.default_rng(config.seed + galaxy_id)
            degraded = degrade_array(
                clean,
                blur_sigma=config.blur_sigma,
                downsample_factor=config.downsample_factor,
                gaussian_noise_std=config.gaussian_noise_std,
                poisson_peak=config.poisson_peak,
                rng=rng,
            )
            save_rgb_float(degraded, degraded_path)

        metrics = image_quality_metrics(clean, degraded)
        records.append(
            {
                "GalaxyID": galaxy_id,
                "clean_path": row.processed_path,
                "degraded_path": str(relative_degraded_path),
                "split": getattr(row, "split", None),
                "blur_sigma": config.blur_sigma,
                "downsample_factor": config.downsample_factor,
                "gaussian_noise_std": config.gaussian_noise_std,
                "poisson_peak": config.poisson_peak,
                **metrics,
            }
        )

    degraded_manifest = pd.DataFrame(records)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    degraded_manifest_path = config.output_dir / "manifest.csv"
    degraded_manifest.to_csv(degraded_manifest_path, index=False)

    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "num_degraded_images": len(degraded_manifest),
        "manifest_path": str(degraded_manifest_path.relative_to(config.project_root)),
        "mean_psnr": float(degraded_manifest["psnr"].mean()),
        "mean_ssim": float(degraded_manifest["ssim"].mean()),
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    return degraded_manifest
