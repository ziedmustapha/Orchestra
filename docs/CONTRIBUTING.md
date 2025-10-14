# Contributing to Orchestra

Thank you for your interest in contributing!

## Development setup

- Python 3.11 recommended
- Create a virtualenv and install minimal deps:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Optional model-specific deps live under `requirements/`:
  - `requirements_env3.txt` (Qwen / WhisSent)
  - `requirements_env4.txt` (Gemma)
  - `requirements_env5.txt` (Qwen3)

## Running locally

- Start workers + load balancer: `./run_api.sh <gemma> <qwen> <qwen3> [whissent] start`
- Dashboard: `http://127.0.0.1:9001/dashboard`
- Smoke test: `python3 tests/test_instances.py`
- Concurrency test: `python3 tests/test_concurrency.py --models qwen3 --per-model-requests 50 --concurrency-per-model 12`

## Coding standards

- Python 3.11, type hints encouraged
- Keep logs structured (see `logging_config.py`) and avoid printing secrets
- For new models, place code under `models/` and expose:
  - `initialize_<model>()`
  - Inference function (e.g., `chat_<model>(...)` or `infer_<model>(...)`)
- Ensure `main.py` dispatch remains consistent and that payload validation returns helpful errors

## Pull requests

- Create a feature branch from `main` (e.g., `feature/qwen-fixes`)
- Keep PRs focused and small when possible
- Include before/after notes and testing steps
- Update docs when user‑facing behavior changes (README, examples)

## Issue reports

When filing bugs, include:
- Version info and environment (GPU/driver/CUDA if relevant)
- Exact steps to reproduce, expected vs actual
- Relevant logs (scrub secrets) from `logs/`

## Security

Please do not file public issues for vulnerabilities. See `SECURITY.md`.
