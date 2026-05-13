from __future__ import annotations

import argparse
from pathlib import Path

from galaxy_zoo_project.reconstruction import ReconstructionConfig, build_reconstruction_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical Galaxy Zoo reconstruction baselines.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--degraded-manifest-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=["identity", "gaussian_smooth", "tv_denoise", "richardson_lucy"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smooth-sigma", type=float, default=0.6)
    parser.add_argument("--tv-weight", type=float, default=0.06)
    parser.add_argument("--unsharp-radius", type=float, default=1.0)
    parser.add_argument("--unsharp-amount", type=float, default=1.2)
    parser.add_argument("--rl-sigma", type=float, default=1.2)
    parser.add_argument("--rl-iterations", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    degraded_manifest_path = (
        args.degraded_manifest_path
        or project_root / "data" / "degraded" / "galaxy_zoo_128_moderate" / "manifest.csv"
    )
    output_dir = args.output_dir or project_root / "data" / "reconstructed" / "galaxy_zoo_128_baselines"

    config = ReconstructionConfig(
        project_root=project_root,
        degraded_manifest_path=degraded_manifest_path,
        output_dir=output_dir,
        methods=tuple(args.methods),
        limit=args.limit,
        smooth_sigma=args.smooth_sigma,
        tv_weight=args.tv_weight,
        unsharp_radius=args.unsharp_radius,
        unsharp_amount=args.unsharp_amount,
        rl_sigma=args.rl_sigma,
        rl_iterations=args.rl_iterations,
        overwrite=args.overwrite,
    )

    manifest = build_reconstruction_dataset(config)
    print(f"Reconstructed {len(manifest):,} method-image pairs")
    print(f"Manifest: {(output_dir / 'manifest.csv').relative_to(project_root)}")
    print(f"Images: {output_dir.relative_to(project_root)}")
