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

To generate only evaluation splits for downstream clean/degraded/reconstructed comparisons:

```bash
.venv/bin/python scripts/degrade_data.py --splits val test --output-dir data/degraded/galaxy_zoo_128_moderate_eval --overwrite
```

## Run reconstruction baselines

Compare classical reconstruction baselines against the degraded observations with:

```bash
.venv/bin/python scripts/reconstruct_data.py --limit 32 --overwrite
```

By default this reads `data/degraded/galaxy_zoo_128_moderate/manifest.csv` and writes:

- `data/reconstructed/galaxy_zoo_128_baselines/<method>/images/`
- `data/reconstructed/galaxy_zoo_128_baselines/manifest.csv`
- `data/reconstructed/galaxy_zoo_128_baselines/summary.json`

## Train the first ML baseline

Train a lightweight split-aware Class 1 morphology classifier on clean images and evaluate it on clean, degraded, and reconstructed images with:

```bash
.venv/bin/python scripts/train_ml_baseline.py
```

The baseline uses downsampled RGB pixels with scikit-learn logistic regression. By default it reads:

- `data/processed/galaxy_zoo_128/manifest.csv`
- `data/degraded/galaxy_zoo_128_moderate/manifest.csv`
- `data/reconstructed/galaxy_zoo_128_baselines/manifest.csv`

and writes:

- `data/ml_baselines/class1_logistic_rgb32/model.pkl`
- `data/ml_baselines/class1_logistic_rgb32/metrics.csv`
- `data/ml_baselines/class1_logistic_rgb32/predictions.csv`
- `data/ml_baselines/class1_logistic_rgb32/summary.json`

Useful options:

```bash
.venv/bin/python scripts/train_ml_baseline.py --feature-size 48 --max-iter 1000
.venv/bin/python scripts/train_ml_baseline.py --feature-mode hog_color --feature-size 64 --max-iter 1000
.venv/bin/python scripts/train_ml_baseline.py --eval-splits val test
.venv/bin/python scripts/train_ml_baseline.py --degraded-manifest-path data/degraded/galaxy_zoo_128_moderate_eval/manifest.csv --reconstruction-manifest-path data/reconstructed/galaxy_zoo_128_eval_baselines/manifest.csv --output-dir data/ml_baselines/class1_logistic_rgb32_eval
```

## Analyze reconstruction-task correlation

Join per-image reconstruction metrics with classifier predictions and compute task metrics, fidelity deltas, and PSNR/SSIM correlations with prediction correctness:

```bash
.venv/bin/python scripts/analyze_reconstruction_task_correlation.py
```

By default this reads:

- `data/degraded/galaxy_zoo_128_moderate_eval/manifest.csv`
- `data/reconstructed/galaxy_zoo_128_eval_baselines/manifest.csv`
- `data/ml_baselines/class1_logistic_rgb32_eval/predictions.csv`

and writes:

- `data/analysis/reconstruction_task_correlation/per_image_analysis.csv`
- `data/analysis/reconstruction_task_correlation/task_metrics.csv`
- `data/analysis/reconstruction_task_correlation/correlations.csv`
- `data/analysis/reconstruction_task_correlation/delta_summary.csv`
- `data/analysis/reconstruction_task_correlation/summary.json`
