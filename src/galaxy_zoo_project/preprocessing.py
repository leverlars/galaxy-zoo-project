from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class PreprocessConfig:
    project_root: Path
    raw_image_dir: Path
    raw_labels_path: Path
    output_dir: Path
    output_size: int = 128
    crop_size: int | None = 256
    image_format: str = "jpg"
    limit: int | None = None
    seed: int = 42
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    overwrite: bool = False


def center_crop(image: Image.Image, crop_size: int | None) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    if crop_size is not None:
        side = min(side, crop_size)

    left = (width - side) // 2
    upper = (height - side) // 2
    return image.crop((left, upper, left + side, upper + side))


def preprocess_image(
    input_path: Path,
    output_path: Path,
    *,
    output_size: int,
    crop_size: int | None,
    overwrite: bool,
) -> bool:
    if output_path.exists() and not overwrite:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as image:
        image = image.convert("RGB")
        image = center_crop(image, crop_size)
        image = image.resize((output_size, output_size), Image.Resampling.LANCZOS)
        image.save(output_path, quality=95)

    return True


def discover_images(raw_image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def assign_splits(
    galaxy_ids: Iterable[int],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[int, str]:
    galaxy_ids = list(galaxy_ids)
    if not galaxy_ids:
        return {}

    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to less than 1.")

    train_ids, temp_ids = train_test_split(
        galaxy_ids,
        test_size=val_fraction + test_fraction,
        random_state=seed,
        shuffle=True,
    )

    split_by_id = {galaxy_id: "train" for galaxy_id in train_ids}
    if temp_ids:
        relative_test_fraction = test_fraction / (val_fraction + test_fraction)
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=relative_test_fraction,
            random_state=seed,
            shuffle=True,
        )
        split_by_id.update({galaxy_id: "val" for galaxy_id in val_ids})
        split_by_id.update({galaxy_id: "test" for galaxy_id in test_ids})

    return split_by_id


def build_processed_dataset(config: PreprocessConfig) -> pd.DataFrame:
    images = discover_images(config.raw_image_dir)
    if config.limit is not None:
        images = images[: config.limit]

    if not images:
        raise FileNotFoundError(f"No images found in {config.raw_image_dir}")
    if not config.raw_labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {config.raw_labels_path}")

    labels = pd.read_csv(config.raw_labels_path)
    if "GalaxyID" not in labels.columns:
        raise ValueError(f"Labels file must contain a GalaxyID column: {config.raw_labels_path}")

    labels_by_id = labels.set_index("GalaxyID", drop=False)
    image_records = []
    missing_label_ids = []

    for image_path in images:
        galaxy_id = int(image_path.stem)
        if galaxy_id not in labels_by_id.index:
            missing_label_ids.append(galaxy_id)
            continue

        relative_raw_path = image_path.relative_to(config.project_root)
        processed_path = config.output_dir / "images" / f"{galaxy_id}.{config.image_format}"
        relative_processed_path = processed_path.relative_to(config.project_root)

        image_records.append(
            {
                "GalaxyID": galaxy_id,
                "raw_path": str(relative_raw_path),
                "processed_path": str(relative_processed_path),
            }
        )

    if not image_records:
        raise ValueError("No discovered images had matching labels.")

    manifest = pd.DataFrame(image_records)
    split_by_id = assign_splits(
        manifest["GalaxyID"].tolist(),
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    manifest["split"] = manifest["GalaxyID"].map(split_by_id)

    label_columns = [column for column in labels.columns if column != "GalaxyID"]
    manifest = manifest.merge(
        labels[["GalaxyID", *label_columns]],
        on="GalaxyID",
        how="left",
        validate="one_to_one",
    )

    for row in tqdm(image_records, desc="Preprocessing images", unit="image"):
        preprocess_image(
            config.project_root / row["raw_path"],
            config.project_root / row["processed_path"],
            output_size=config.output_size,
            crop_size=config.crop_size,
            overwrite=config.overwrite,
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "num_raw_images": len(images),
        "num_processed_images": len(manifest),
        "num_missing_labels": len(missing_label_ids),
        "splits": manifest["split"].value_counts().sort_index().to_dict(),
        "manifest_path": str(manifest_path.relative_to(config.project_root)),
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if missing_label_ids:
        missing_path = config.output_dir / "missing_label_ids.txt"
        missing_path.write_text("\n".join(str(galaxy_id) for galaxy_id in missing_label_ids) + "\n")

    return manifest
