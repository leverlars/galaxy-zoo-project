# Galaxy Zoo Imaging and Morphology Experiments

This project studies how computational imaging degradations affect galaxy morphology prediction, and whether classical reconstruction methods recover image quality in a way that helps machine learning performance.

The central imaging model is:

```text
y = Hx + n
```

where `x` is a clean Galaxy Zoo image, `H` is a blur/downsampling operator, and `n` is additive noise. The project compares image fidelity metrics such as PSNR and SSIM with downstream Class 1 Galaxy Zoo morphology classification metrics.

## Repository Contents

- `src/galaxy_zoo_project/`: reusable Python implementation for preprocessing, degradation, reconstruction, ML baselines, and correlation analysis.
- `scripts/`: command-line entry points for running the full experiment pipeline.
- `notebooks/`: exploratory notebooks used during data inspection and early pipeline development.
- `pyproject.toml` and `uv.lock`: Python environment and dependency definitions.

Generated datasets, models, and analysis outputs are written under `data/` and are intentionally not tracked by git.

## Setup

This project uses `uv` and Python 3.13.

```bash
uv sync
```

Commands below assume the local virtual environment at `.venv/`.

## Data

The experiments use the Galaxy Zoo challenge data:

- `images_training_rev1/`
- `training_solutions_rev1.csv`

Place or extract them under:

```text
data/raw/images_training_rev1/
data/raw/training_solutions_rev1.csv
```

The raw data are not included in the repository.

## Reproduce the Pipeline

### 1. Preprocess Clean Images

```bash
.venv/bin/python scripts/preprocess_data.py --overwrite
```

This creates fixed size clean images and a split aware manifest under:

```text
data/processed/galaxy_zoo_128/
```

### 2. Generate Degraded Observations

For the report-scale validation/test comparison:

```bash
.venv/bin/python scripts/degrade_data.py \
  --splits val test \
  --output-dir data/degraded/galaxy_zoo_128_moderate_eval \
  --overwrite
```

The default degradation applies Gaussian blur, 2x downsampling/upsampling, and Gaussian noise.

### 3. Run Reconstruction Baselines

```bash
.venv/bin/python scripts/reconstruct_data.py \
  --degraded-manifest-path data/degraded/galaxy_zoo_128_moderate_eval/manifest.csv \
  --output-dir data/reconstructed/galaxy_zoo_128_eval_baselines \
  --overwrite
```

Implemented reconstruction baselines:

- identity
- gaussian_smooth
- tv_denoise
- richardson_lucy

### 4. Train and Evaluate ML Baselines

Raw RGB logistic regression baseline:

```bash
.venv/bin/python scripts/train_ml_baseline.py \
  --degraded-manifest-path data/degraded/galaxy_zoo_128_moderate_eval/manifest.csv \
  --reconstruction-manifest-path data/reconstructed/galaxy_zoo_128_eval_baselines/manifest.csv \
  --output-dir data/ml_baselines/class1_logistic_rgb32_eval \
  --max-iter 500
```

HOG plus color-statistics baseline:

```bash
.venv/bin/python scripts/train_ml_baseline.py \
  --feature-mode hog_color \
  --feature-size 64 \
  --degraded-manifest-path data/degraded/galaxy_zoo_128_moderate_eval/manifest.csv \
  --reconstruction-manifest-path data/reconstructed/galaxy_zoo_128_eval_baselines/manifest.csv \
  --output-dir data/ml_baselines/class1_logistic_hog_color64_eval \
  --max-iter 1000
```

### 5. Analyze Fidelity vs Task Performance

For the RGB baseline:

```bash
.venv/bin/python scripts/analyze_reconstruction_task_correlation.py
```

For the HOG plus color baseline:

```bash
.venv/bin/python scripts/analyze_reconstruction_task_correlation.py \
  --predictions-path data/ml_baselines/class1_logistic_hog_color64_eval/predictions.csv \
  --output-dir data/analysis/reconstruction_task_correlation_hog_color64
```

The analysis writes per-image joined data, task metrics, fidelity deltas, correlations, and summary JSON files under `data/analysis/`.