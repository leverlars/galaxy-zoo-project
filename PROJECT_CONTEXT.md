# Galaxy Zoo Computational Imaging + AI — Project Context

## Project identity
This repository supports a dual-course project spanning:
- **Computational Imaging**
- **Physics Applications of AI**

## Core research question
How do realistic imaging degradations affect galaxy classification/regression performance, and can computational imaging reconstruction methods recover lost information to improve downstream ML performance?

## Scientific framing
The project studies the relationship between:
- image quality metrics (PSNR, SSIM, MSE)
- downstream task metrics (classification accuracy/F1, regression MSE)

under an imaging model of the form:
\[
y = Hx + n
\]
where:
- `x` = clean image
- `H` = degradation operator (blur/downsampling)
- `n` = noise

## Current status
- Project setup: **COMPLETE**
- Data access: **COMPLETE**
- Degradation/reconstruction pipeline: **WORKING**
- Initial reconstruction experiments: **COMPLETE**
- ML experiments: **STARTED**

## Current implemented pipeline
1. Preprocess Galaxy Zoo images into fixed-size clean images and manifest
2. Create degraded observations (blur, downsampling, noise)
3. Run reconstruction baselines and compute PSNR/SSIM

Current reconstruction baselines:
- identity
- gaussian_smooth
- tv_denoise
- richardson_lucy

## Pilot result snapshot (small-scale)
Moderate degradation, 32 images:

| Method          | PSNR  | SSIM  |
|-----------------|-------|-------|
| gaussian_smooth | 34.67 | 0.782 |
| tv_denoise      | 34.46 | 0.778 |
| richardson_lucy | 32.29 | 0.711 |
| identity        | 32.27 | 0.676 |

Interpretation:
- Gaussian smoothing and TV denoising improved image-quality metrics.
- Richardson–Lucy under this regime did not improve as much and may amplify noise.
- Degradation appears more noise-dominated than blur-dominated.

## Working hypotheses
1. Degradation will reduce ML performance versus clean images.
2. Reconstruction may recover some ML performance compared with degraded inputs.
3. Higher PSNR/SSIM may not always correspond to higher ML performance.

## Immediate roadmap
### Priority 1 — ML baseline
- Implement a baseline CNN classifier.
- Optionally add a regression head for vote percentages.
- Evaluate on clean, degraded, and reconstructed conditions.

Current lightweight baseline:
- scikit-learn logistic regression on RGB pixels resized to 32x32
- Class 1 target via argmax over `Class1.1`, `Class1.2`, `Class1.3`
- trained on clean train split
- evaluated on clean, degraded, and reconstructed val/test splits

### Priority 2 — Reconstruction impact study
Main comparison:
- clean → model
- degraded → model
- reconstructed → model

### Priority 3 — Correlation analysis
- Compare per-condition PSNR/SSIM against downstream ML metrics.
- Analyze whether image-fidelity gains translate to task gains.

Current status:
- per-image reconstruction/task correlation analysis is implemented
- outputs are written to `data/analysis/reconstruction_task_correlation/`
- early result: PSNR/SSIM improvements have weak association with logistic-baseline correctness

## Constraints and conventions
- Use `uv` and local `.venv` only.
- Do not commit datasets or `.venv`.
- Keep reusable logic in `src/`.
- Use notebooks for exploration, scripts/modules for reproducible pipelines.
- Prefer modular, reviewable commits.

## Fast resume checklist (for new sessions)
1. Read this file.
2. Read `AGENTS.md`.
3. Read latest entries in `docs/EXPERIMENT_LOG.md`.
4. Confirm immediate next milestone.
5. Execute in 1–3 small commits with concise rationale.
