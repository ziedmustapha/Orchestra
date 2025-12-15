# Contributing to Orchestra v3

Thank you for your interest in contributing!

## Development setup

- Python 3.10+ recommended (3.11 for best performance)
- Install [uv](https://github.com/astral-sh/uv) for fast package management:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Run the setup script to create virtual environments:
  ```bash
  scripts/setup_envs.sh  # uses uv for fast dependency installation
  ```
- Model-specific requirements:
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
- Orchestra version (v3.x.x) and environment (GPU/driver/CUDA if relevant)
- Exact steps to reproduce, expected vs actual
- Relevant logs (scrub secrets) from `logs/`
- vLLM configuration if performance-related (environment variables)

## Security

Please do not file public issues for vulnerabilities. See `SECURITY.md`.
