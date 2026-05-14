# Experiment Log

Use this file as the persistent memory of what has been tried and what should happen next.

## Entry template

### YYYY-MM-DD — Short title
- **Owner:**
- **Commit(s):**
- **Objective:**
- **Data/config:**
- **Commands run:**
- **Key metrics:**
- **Artifacts produced:**
- **Interpretation:**
- **Next action:**

---

## 2026-05-14 — Project memory scaffolding initialized
- **Owner:** Codex + repo maintainer
- **Commit(s):** pending at time of entry
- **Objective:** Create persistent context files so new Codex sessions can resume without re-explaining project status.
- **Data/config:** N/A
- **Commands run:** file creation only
- **Key metrics:** N/A
- **Artifacts produced:** `AGENTS.md`, `PROJECT_CONTEXT.md`, `docs/EXPERIMENT_LOG.md`
- **Interpretation:** Baseline memory scaffolding is in place.
- **Next action:** Start ML baseline implementation (manifest-driven dataset + first CNN training/eval scripts).

---

## 2026-05-14 — First manifest-driven ML baseline
- **Owner:** Codex
- **Commit(s):** pending at time of entry
- **Objective:** Add a reproducible downstream ML baseline that connects clean, degraded, and reconstructed image conditions to Galaxy Zoo morphology performance.
- **Data/config:** Trained on clean `train` split from `data/processed/galaxy_zoo_128/manifest.csv`; target is argmax over `Class1.1`, `Class1.2`, `Class1.3`; features are RGB pixels resized to 32x32; classifier is scikit-learn logistic regression with balanced class weights.
- **Commands run:** `.venv/bin/python scripts/train_ml_baseline.py --output-dir data/ml_baselines/class1_logistic_rgb32 --max-iter 500`
- **Key metrics:** Clean test accuracy `0.62`, macro F1 `0.601`; clean val accuracy `0.57`, macro F1 `0.557`. Degraded/reconstructed metrics were produced but are based on very small val/test coverage in the current pilot artifacts.
- **Artifacts produced:** `src/galaxy_zoo_project/ml_baseline.py`, `src/galaxy_zoo_project/ml_baseline_cli.py`, `scripts/train_ml_baseline.py`, `data/ml_baselines/class1_logistic_rgb32/summary.json`, `metrics.csv`, `predictions.csv`, `model.pkl`.
- **Interpretation:** The downstream evaluation loop now exists, but the current degraded and reconstructed datasets are too small for reliable conclusions about reconstruction impact.
- **Next action:** Regenerate degraded and reconstructed artifacts over full val/test splits, then rerun the ML baseline to compare clean vs degraded vs reconstructed performance with adequate sample sizes.

---

## 2026-05-14 — Full val/test degradation and reconstruction impact baseline
- **Owner:** Codex
- **Commit(s):** pending at time of entry
- **Objective:** Replace tiny pilot downstream metrics with full validation/test split coverage for clean vs degraded vs reconstructed comparison.
- **Data/config:** `val` + `test` splits from `data/processed/galaxy_zoo_128/manifest.csv`; moderate degradation with blur sigma `1.2`, downsample factor `2`, Gaussian noise std `0.03`; reconstruction methods `identity`, `gaussian_smooth`, `tv_denoise`, `richardson_lucy`; Class 1 logistic RGB32 ML baseline trained on clean train split.
- **Commands run:** `.venv/bin/python scripts/degrade_data.py --splits val test --output-dir data/degraded/galaxy_zoo_128_moderate_eval --overwrite`; `.venv/bin/python scripts/reconstruct_data.py --degraded-manifest-path data/degraded/galaxy_zoo_128_moderate_eval/manifest.csv --output-dir data/reconstructed/galaxy_zoo_128_eval_baselines --overwrite`; `.venv/bin/python scripts/train_ml_baseline.py --degraded-manifest-path data/degraded/galaxy_zoo_128_moderate_eval/manifest.csv --reconstruction-manifest-path data/reconstructed/galaxy_zoo_128_eval_baselines/manifest.csv --output-dir data/ml_baselines/class1_logistic_rgb32_eval --max-iter 500`.
- **Key metrics:** Degraded eval set: 200 images, mean PSNR `30.513`, mean SSIM `0.596`. Reconstruction mean SSIM: gaussian smooth `0.786`, TV denoise `0.776`, Richardson-Lucy `0.719`, identity `0.681`. ML test accuracy/macro F1: clean `0.62`/`0.601`, degraded `0.60`/`0.580`, gaussian smooth `0.62`/`0.597`, TV denoise `0.62`/`0.597`, Richardson-Lucy `0.60`/`0.583`, identity `0.61`/`0.592`. ML val accuracy/macro F1: clean `0.57`/`0.557`, degraded `0.54`/`0.525`, gaussian smooth `0.54`/`0.525`, TV denoise `0.52`/`0.507`, Richardson-Lucy `0.52`/`0.504`, identity `0.55`/`0.533`.
- **Artifacts produced:** `data/degraded/galaxy_zoo_128_moderate_eval/`, `data/reconstructed/galaxy_zoo_128_eval_baselines/`, `data/ml_baselines/class1_logistic_rgb32_eval/`.
- **Interpretation:** Reconstruction substantially improves PSNR/SSIM for gaussian smoothing and TV denoising, but this simple downstream classifier shows little to no task-performance recovery. This supports the hypothesis that image-fidelity metrics may not directly track ML utility.
- **Next action:** Add correlation analysis between per-image PSNR/SSIM and prediction correctness, then consider a stronger CNN/transfer-learning baseline if dependencies are allowed.

---

## 2026-05-14 — Reconstruction metric vs task correctness correlation
- **Owner:** Codex
- **Commit(s):** pending at time of entry
- **Objective:** Quantify whether per-image PSNR/SSIM and reconstruction metric gains align with downstream Class 1 prediction correctness.
- **Data/config:** Joined `data/degraded/galaxy_zoo_128_moderate_eval/manifest.csv`, `data/reconstructed/galaxy_zoo_128_eval_baselines/manifest.csv`, and `data/ml_baselines/class1_logistic_rgb32_eval/predictions.csv`.
- **Commands run:** `.venv/bin/python scripts/analyze_reconstruction_task_correlation.py`
- **Key metrics:** Analyzed 800 reconstructed image-method rows. Strongest observed Spearman correlation with correctness was `0.281` for test-set TV-denoise `ssim_delta_vs_identity`; other top absolute Spearman correlations were mostly `0.13` to `0.22`. Mean test task improvement versus degraded was small: gaussian smoothing `+0.02`, TV denoise `+0.02`, Richardson-Lucy `0.00`, identity `+0.01` correctness fraction. Mean val task change versus degraded was gaussian smoothing `0.00`, identity `+0.01`, Richardson-Lucy `-0.02`, TV denoise `-0.02`.
- **Artifacts produced:** `src/galaxy_zoo_project/correlation_analysis.py`, `src/galaxy_zoo_project/correlation_analysis_cli.py`, `scripts/analyze_reconstruction_task_correlation.py`, `data/analysis/reconstruction_task_correlation/per_image_analysis.csv`, `task_metrics.csv`, `correlations.csv`, `delta_summary.csv`, `summary.json`.
- **Interpretation:** Fidelity improvements are real, but their association with correctness is weak for this lightweight classifier. This strengthens the project result that PSNR/SSIM improvement alone is not enough evidence of downstream ML recovery.
- **Next action:** Add visualization notebook/table exports for the report, then decide whether to install PyTorch for a stronger CNN baseline or stay lightweight with HOG/color-feature classifiers.

---

## 2026-05-14 — HOG+color feature classifier baseline
- **Owner:** Codex
- **Commit(s):** pending at time of entry
- **Objective:** Improve the downstream classifier without adding heavy deep-learning dependencies, then test whether the reconstruction-vs-task conclusions hold for a stronger hand-crafted feature baseline.
- **Data/config:** Same clean train split and full val/test degraded/reconstructed eval artifacts as previous run. Logistic regression with balanced class weights, `feature_mode=hog_color`, `feature_size=64`, `max_iter=1000`.
- **Commands run:** `.venv/bin/python scripts/train_ml_baseline.py --feature-mode hog_color --feature-size 64 --degraded-manifest-path data/degraded/galaxy_zoo_128_moderate_eval/manifest.csv --reconstruction-manifest-path data/reconstructed/galaxy_zoo_128_eval_baselines/manifest.csv --output-dir data/ml_baselines/class1_logistic_hog_color64_eval --max-iter 1000`; `.venv/bin/python scripts/analyze_reconstruction_task_correlation.py --predictions-path data/ml_baselines/class1_logistic_hog_color64_eval/predictions.csv --output-dir data/analysis/reconstruction_task_correlation_hog_color64`.
- **Key metrics:** Clean test accuracy/macro F1 `0.61`/`0.598`; clean val `0.68`/`0.657`. Degraded test `0.67`/`0.655`; degraded val `0.65`/`0.634`. Best reconstructed test accuracy was Richardson-Lucy `0.68`; best reconstructed val accuracy was Gaussian smoothing `0.66`. Strongest absolute Spearman correlation between fidelity metric and correctness was weak, about `0.208`.
- **Artifacts produced:** Extended `src/galaxy_zoo_project/ml_baseline.py` and `src/galaxy_zoo_project/ml_baseline_cli.py` with `rgb`, `hog`, `color_stats`, and `hog_color` feature modes. Produced `data/ml_baselines/class1_logistic_hog_color64_eval/` and `data/analysis/reconstruction_task_correlation_hog_color64/`.
- **Interpretation:** HOG+color features are stronger than raw RGB on validation and more robust to the current degradation. Reconstruction still does not show a consistent downstream benefit, and fidelity/correctness correlations remain weak. The scientific conclusion is therefore less tied to the initial raw-pixel baseline.
- **Next action:** Add report-ready visualizations/tables comparing raw RGB vs HOG+color, or install PyTorch for a CNN baseline if dependency size is acceptable.
