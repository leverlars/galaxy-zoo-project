# galaxy-zoo-project

## Preprocess the Galaxy Zoo data

After running the download/extract cells in `notebooks/01_data_exploration.ipynb`, create a clean fixed-size dataset with:

```bash
.venv/bin/python scripts/preprocess_data.py
```

By default this reads:

- `data/raw/images_training_rev1/`
- `data/raw/training_solutions_rev1.csv`

and writes:

- `data/processed/galaxy_zoo_128/images/`
- `data/processed/galaxy_zoo_128/manifest.csv`
- `data/processed/galaxy_zoo_128/summary.json`

Useful options:

```bash
.venv/bin/python scripts/preprocess_data.py --limit 100 --overwrite
.venv/bin/python scripts/preprocess_data.py --output-size 64 --crop-size 256
```

## Simulate degraded observations

Create blurred, downsampled, noisy observations from the processed clean images with:

```bash
.venv/bin/python scripts/degrade_data.py --limit 100 --overwrite
```

By default this reads `data/processed/galaxy_zoo_128/manifest.csv` and writes:

- `data/degraded/galaxy_zoo_128_moderate/images/`
- `data/degraded/galaxy_zoo_128_moderate/manifest.csv`
- `data/degraded/galaxy_zoo_128_moderate/summary.json`

## Run reconstruction baselines

Compare classical reconstruction baselines against the degraded observations with:

```bash
.venv/bin/python scripts/reconstruct_data.py --limit 32 --overwrite
```

By default this reads `data/degraded/galaxy_zoo_128_moderate/manifest.csv` and writes:

- `data/reconstructed/galaxy_zoo_128_baselines/<method>/images/`
- `data/reconstructed/galaxy_zoo_128_baselines/manifest.csv`
- `data/reconstructed/galaxy_zoo_128_baselines/summary.json`
