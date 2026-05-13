from __future__ import annotations

import argparse
from pathlib import Path

from galaxy_zoo_project.preprocessing import PreprocessConfig, build_processed_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a processed Galaxy Zoo image dataset.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--raw-image-dir", type=Path, default=None)
    parser.add_argument("--raw-labels-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-size", type=int, default=128)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    raw_image_dir = args.raw_image_dir or project_root / "data" / "raw" / "images_training_rev1"
    raw_labels_path = args.raw_labels_path or project_root / "data" / "raw" / "training_solutions_rev1.csv"
    output_dir = args.output_dir or project_root / "data" / "processed" / f"galaxy_zoo_{args.output_size}"

    config = PreprocessConfig(
        project_root=project_root,
        raw_image_dir=raw_image_dir,
        raw_labels_path=raw_labels_path,
        output_dir=output_dir,
        output_size=args.output_size,
        crop_size=args.crop_size,
        image_format=args.image_format,
        limit=args.limit,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        overwrite=args.overwrite,
    )

    manifest = build_processed_dataset(config)
    print(f"Processed {len(manifest):,} images")
    print(f"Manifest: {(output_dir / 'manifest.csv').relative_to(project_root)}")
    print(f"Images: {(output_dir / 'images').relative_to(project_root)}")
