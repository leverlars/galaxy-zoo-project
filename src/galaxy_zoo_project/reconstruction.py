from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.filters import gaussian, unsharp_mask
from skimage.restoration import denoise_tv_chambolle, richardson_lucy
from tqdm.auto import tqdm

from galaxy_zoo_project.degradation import image_quality_metrics, load_rgb_float, save_rgb_float


@dataclass(frozen=True)
class ReconstructionConfig:
    project_root: Path
    degraded_manifest_path: Path
    output_dir: Path
    methods: tuple[str, ...] = ("identity", "gaussian_smooth", "tv_denoise", "richardson_lucy")
    limit: int | None = None
    smooth_sigma: float = 0.6
    tv_weight: float = 0.06
    unsharp_radius: float = 1.0
    unsharp_amount: float = 1.2
    rl_sigma: float = 1.2
    rl_iterations: int = 12
    overwrite: bool = False


def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("PSF size must be odd.")
    if sigma <= 0:
        raise ValueError("PSF sigma must be positive.")

    axis = np.arange(size, dtype=np.float32) - size // 2
    xx, yy = np.meshgrid(axis, axis)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def reconstruct_array(image: np.ndarray, method: str, config: ReconstructionConfig) -> np.ndarray:
    if method == "identity":
        reconstructed = image
    elif method == "gaussian_smooth":
        reconstructed = gaussian(
            image,
            sigma=config.smooth_sigma,
            channel_axis=-1,
            preserve_range=True,
        )
    elif method == "tv_denoise":
        reconstructed = denoise_tv_chambolle(
            image,
            weight=config.tv_weight,
            channel_axis=-1,
        )
    elif method == "unsharp":
        reconstructed = unsharp_mask(
            image,
            radius=config.unsharp_radius,
            amount=config.unsharp_amount,
            preserve_range=True,
            channel_axis=-1,
        )
    elif method == "richardson_lucy":
        psf_size = max(5, int(np.ceil(config.rl_sigma * 6)) | 1)
        psf = gaussian_psf(psf_size, config.rl_sigma)
        channels = [
            richardson_lucy(
                image[:, :, channel],
                psf,
                num_iter=config.rl_iterations,
                clip=False,
            )
            for channel in range(image.shape[-1])
        ]
        reconstructed = np.stack(channels, axis=-1)
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")

    return np.clip(reconstructed, 0.0, 1.0).astype(np.float32)


def build_reconstruction_dataset(config: ReconstructionConfig) -> pd.DataFrame:
    if not config.degraded_manifest_path.exists():
        raise FileNotFoundError(f"Degraded manifest not found: {config.degraded_manifest_path}")

    degraded_manifest = pd.read_csv(config.degraded_manifest_path)
    required_columns = {"GalaxyID", "clean_path", "degraded_path"}
    missing_columns = required_columns - set(degraded_manifest.columns)
    if missing_columns:
        raise ValueError(f"Degraded manifest is missing required columns: {sorted(missing_columns)}")

    if config.limit is not None:
        degraded_manifest = degraded_manifest.head(config.limit).copy()

    if degraded_manifest.empty:
        raise ValueError("No degraded images selected for reconstruction.")

    records = []

    for row in tqdm(
        degraded_manifest.itertuples(index=False),
        total=len(degraded_manifest),
        desc="Reconstructing images",
        unit="image",
    ):
        galaxy_id = int(row.GalaxyID)
        clean = load_rgb_float(config.project_root / row.clean_path)
        degraded = load_rgb_float(config.project_root / row.degraded_path)

        for method in config.methods:
            output_path = config.output_dir / method / "images" / f"{galaxy_id}.jpg"
            relative_output_path = output_path.relative_to(config.project_root)

            if output_path.exists() and not config.overwrite:
                reconstructed = load_rgb_float(output_path)
            else:
                reconstructed = reconstruct_array(degraded, method, config)
                save_rgb_float(reconstructed, output_path)

            metrics = image_quality_metrics(clean, reconstructed)
            records.append(
                {
                    "GalaxyID": galaxy_id,
                    "method": method,
                    "clean_path": row.clean_path,
                    "degraded_path": row.degraded_path,
                    "reconstructed_path": str(relative_output_path),
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                }
            )

    reconstruction_manifest = pd.DataFrame(records)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "manifest.csv"
    reconstruction_manifest.to_csv(manifest_path, index=False)

    metric_summary = (
        reconstruction_manifest.groupby("method")[["psnr", "ssim"]]
        .mean()
        .sort_values("ssim", ascending=False)
        .to_dict(orient="index")
    )
    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "num_input_images": len(degraded_manifest),
        "num_reconstructed_images": len(reconstruction_manifest),
        "manifest_path": str(manifest_path.relative_to(config.project_root)),
        "metric_summary": metric_summary,
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    return reconstruction_manifest
