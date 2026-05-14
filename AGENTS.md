# AGENTS.md

Scope: entire repository.

## Workflow preferences
- At session start, read `PROJECT_CONTEXT.md` and `docs/EXPERIMENT_LOG.md` before proposing changes.
- Keep commits small and modular.
- Prefer reusable Python modules in `src/` over notebook-only code.
- Use `uv` and local `.venv`; do not introduce conda workflows.

## Scientific priorities
- Primary objective: connect image degradations/reconstruction quality with downstream ML task performance.
- Preserve strict train/val/test split consistency across clean, degraded, and reconstructed variants.
- Favor reproducibility over benchmark chasing.

## Safety and repo hygiene
- Never commit datasets, generated large binaries, or `.venv`.
- Avoid destructive file operations unless explicitly requested.

## Resume protocol
1. Read `PROJECT_CONTEXT.md`.
2. Read latest entries in `docs/EXPERIMENT_LOG.md`.
3. Propose next 1–3 concrete tasks and execute after confirmation.
