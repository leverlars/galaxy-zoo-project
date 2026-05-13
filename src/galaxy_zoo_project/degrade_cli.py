from __future__ import annotations

import argparse
from pathlib import Path

from galaxy_zoo_project.degradation import DegradationConfig, build_degraded_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create degraded Galaxy Zoo observations.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--processed-manifest-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--blur-sigma", type=float, default=1.2)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.03)
    parser.add_argument("--poisson-peak", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    processed_manifest_path = (
        args.processed_manifest_path
        or project_root / "data" / "processed" / "galaxy_zoo_128" / "manifest.csv"
    )
    output_dir = args.output_dir or project_root / "data" / "degraded" / "galaxy_zoo_128_moderate"

    config = DegradationConfig(
        project_root=project_root,
        processed_manifest_path=processed_manifest_path,
        output_dir=output_dir,
        blur_sigma=args.blur_sigma,
        downsample_factor=args.downsample_factor,
        gaussian_noise_std=args.gaussian_noise_std,
        poisson_peak=args.poisson_peak,
        limit=args.limit,
        split=args.split,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    manifest = build_degraded_dataset(config)
    print(f"Degraded {len(manifest):,} images")
    print(f"Manifest: {(output_dir / 'manifest.csv').relative_to(project_root)}")
    print(f"Images: {(output_dir / 'images').relative_to(project_root)}")
