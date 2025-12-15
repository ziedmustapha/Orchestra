# Orchestra v3

[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/ziedmustapha/Orchestra)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

> Orchestrate multiple AI models on distributed GPUs with intelligent load balancing, vLLM optimization, and real-time observability.

**What's New in v3**: H100 GPU optimization, vLLM engine configuration, improved batching strategies, and 17+ req/s throughput on demanding models. See [CHANGELOG](docs/CHANGELOG.md).

## Table of Contents

- [Features](#features)
- [Installation & Quick Start](#installation--quick-start)
- [Models & Environments](#models--environments)
- [cURL: Qwen3 text generation](#curl-qwen3-text-generation)
- [cURL: Qwen‑VL (multimodal)](#curl-qwen-vl-multimodal)
- [cURL: WhisSent (ASR + Emotion)](#curl-whissent-asr--emotion)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)
- [Author](#author)
- [Dashboard & Observability](#dashboard--observability)
- [Structured Logging](#structured-logging)
- [Parallelization & GPU Sharing](#parallelization--gpu-sharing)
- [v3 Performance Benchmarks](#v3-performance-benchmarks)
- [Validation & Load Testing](#validation--load-testing)
- [Security & Binding Defaults](#security--binding-defaults)

## Features

### Core Capabilities
- **Multi-Model Support**: Gemma3, Qwen2.5-VL (multimodal), Qwen3 (MoE), WhisSent (ASR + Emotion)
- **Load Balancing**: Intelligent routing across GPU workers with queueing and health checks
- **Real-time Dashboard**: Live metrics, per-worker stats, log viewer

### v3 Performance Optimizations
- **vLLM Engine Tuning**: Environment-driven configuration for max throughput
- **H100 GPU Optimization**: MoE kernel configs, 95% GPU memory utilization
- **Intelligent Batching**: Coalescing delays, chunked prefill, optimized scheduler
- **Async Processing**: Burst-aware request dispatcher for better parallelism

### Infrastructure
- **GPU Management**: CUDA MPS for multi-process sharing, single-model-per-GPU enforcement
- **Authentication**: Optional API key auth for external access
- **Observability**: Structured JSON logs, correlation IDs, log rotation
- **Testing**: Concurrency tests, multi-user simulation, GPU profiling tools

## Installation & Quick Start

Follow these steps to get Orchestra running:

```bash
# 1) Clone the repository
git clone https://github.com/ziedmustapha/Orchestra.git
cd Orchestra

# 2) Install uv (fast Python package manager) if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3) Set up per-model virtual environments and install dependencies
scripts/setup_envs.sh           # creates .venvs/{env3,env4,env5,env-lb}, uses uv for fast installs

# 4) Activate the environment variables
source ./orchestra.env          # exports GEMMA/QWEN/QWEN3/WHISSENT/LOAD_BALANCER_PYTHON_PATH

# 5) Install system prerequisites (for WhisSent and PDFs)
sudo apt-get update && sudo apt-get install -y ffmpeg poppler-utils

# 6) Start the API with 1 Qwen3 worker (adjust numbers as needed)
scripts/run_api.sh 0 0 1 1 start

# 7) Open the dashboard
http://127.0.0.1:9001/dashboard

# 8) Test it quickly
python3 tests/test_concurrency.py --models whissent,qwen3 --per-model-requests 10 --concurrency-per-model 10 --models-concurrently

# 9) Stop everything
scripts/stop_all.sh
```

## Models & Environments

- **env4 → Gemma3** (GEMMA_PYTHON_PATH)
- **env3 → Qwen2.5‑VL + WhisSent** (QWEN_PYTHON_PATH, WHISSENT_PYTHON_PATH)
- **env5 → Qwen3** (QWEN3_PYTHON_PATH)
- **env-lb → Load balancer** (LOAD_BALANCER_PYTHON_PATH)

See Installation & Quick Start above for setup.

## cURL: Qwen3 text generation

The load balancer exposes a single endpoint `POST /infer`. To target Qwen3, set `model_name` to `"qwen3"` and provide a `request_body` with your prompt.

Minimal local example:

```bash
curl -sS -X POST http://127.0.0.1:9001/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "qwen3",
    "request_body": {
      "input": "Write a 50-word summary about GPU scheduling.",
      "max_new_tokens": 256,
      "session_id": "demo-qwen3-001"
    }
  }' | jq .
```

With API key and custom request ID (for external callers):

```bash
LB=http://127.0.0.1:9001
API_KEY="<your-api-key>"  # see docs/API_AUTHENTICATION.md and api_keys.json

curl -sS -X POST "$LB/infer" \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $API_KEY" \
  -H "X-Request-ID: demo-req-12345" \
  -d '{
    "model_name": "qwen3",
    "request_body": {
      "input": "Explain the difference between tensor parallelism and pipeline parallelism.",
      "max_new_tokens": 256,
      "session_id": "demo-qwen3-002"
    }
  }' | jq .
```

What the fields mean:

- `model_name`: must be `"qwen3"` to hit the Qwen3 worker pool.
- `request_body.input`: your prompt text (string).
- `request_body.max_new_tokens` (optional): upper bound on generated tokens. Default is model-specific; we use a safe default if omitted.
- `request_body.session_id` (optional): any string to help you correlate requests.
- `X-API-Key` header (optional): only required for non-local requests if you enable auth in `api_keys.json`.
- `X-Request-ID` header (optional): sets/propagates a correlation ID across logs.

Example success response (shape):

```json
{
  "response": "... model text ...",
  "gpu_id_of_model_worker": "auto",
  "model_name_processed": "qwen3",
  "duration_seconds": 2.34,
  "api_worker_pid": 12345,
  "load_balancer_worker_id": 0,
  "lb_latency_ms": 123.4,
  "lb_queue_at_dispatch": 0
}
```

Notes:

- Qwen3 currently accepts `input` and `max_new_tokens` via the public API. Additional sampling params (e.g., `temperature`) are not exposed at this layer yet.
- For more about auth, see `docs/API_AUTHENTICATION.md`. Local requests (127.0.0.1) typically don’t require an API key unless you enforce it.

## cURL: Qwen‑VL (multimodal)

Send one or more images along with a text prompt. Supported image sources: local absolute path, URL, or base64 (see Qwen section in code for accepted formats).

```bash
curl -sS -X POST http://127.0.0.1:9001/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "qwen",
    "request_body": {
      "input": "Extract the total amount and date from this invoice.",
      "images": ["/absolute/path/to/invoice.png"],
      "max_new_tokens": 512,
      "session_id": "demo-qwen-vl-001"
    }
  }' | jq .
```

## cURL: WhisSent (ASR + Emotion)

WhisSent combines Hugging Face models:
- ASR: `openai/whisper-large-v3`
- Emotion (FR): `Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition`

Each WhisSent worker accepts `model_name = "whissent"` with an audio payload and returns:
- `transcript` and optional `chunks` with timestamps
- `emotion` (top label) and `emotion_scores` (list of labels/scores)

Example `POST /infer` request via the load balancer:
```json
{
  "model_name": "whissent",
  "request_body": {
    "audio": {"path": "/absolute/path/to/audio.wav"},
    "return_timestamps": true,
    "task": "transcribe",
    "language": "fr",
    "emotion_top_k": 3,
    "session_id": "demo-123"
  }
}
```

Audio sources supported:
- `{ "path": "/abs/file.wav" }`
- `{ "url": "https://.../file.mp3" }`
- `{ "data": "<base64>", "format": "wav|mp3|flac|m4a|ogg" }`

Dependencies for WhisSent (env3):
- Python: `torch`, `torchaudio`, `transformers`, `requests`
- Audio I/O: `librosa`, `soundfile` (install via pip)
- System: `ffmpeg` (install via apt/yum if not present)

If you run `tests/test_concurrency.py`, the test excludes `whissent` by default (to avoid requiring an audio file). To include WhisSent, pass the model name and an audio path:
```bash
python3 tests/test_concurrency.py --models whissent --audio-file /absolute/path/to/audio.wav
```

## Installation & Environment Setup

- **Host/port**: set `LOAD_BALANCER_HOST` (default `127.0.0.1`) and `LOAD_BALANCER_PORT` (default `9001`).
- **API keys file**: set `API_KEYS_FILE` or place `api_keys.json` at repo root.
- **Worker counts**: pass counts to `scripts/run_api.sh` or use `GEMMA3_INSTANCES`, `QWEN_INSTANCES`, `QWEN3_INSTANCES`, `WHISSENT_INSTANCES`.

## Development

- **Run**: `./scripts/run_api.sh 2 1 1 start` then open `http://127.0.0.1:9001/dashboard`.
- **Smoke test**: `python3 tests/test_instances.py`.
- **Concurrency**: `python3 tests/test_concurrency.py --models qwen3 --per-model-requests 50 --concurrency-per-model 12`.

## Contributing

Please see `CONTRIBUTING.md` for workflow, coding standards, and how to run tests. By participating, you agree to the `CODE_OF_CONDUCT.md`.

## Security

See `SECURITY.md` for reporting vulnerabilities. Do not include secrets in issues/PRs. Keep `api_keys.json` out of version control.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Zied Mustapha - Data Scientist and AI engineer @ Teldar IA - Switzerland

## Dashboard & Observability

The load balancer now embeds a real‑time dashboard and metrics endpoints.

![Dashboard Screenshot](docs/image_readme.png)

- Dashboard: `http://127.0.0.1:9001/dashboard`
  - Overview: uptime, total requests, success/errors
  - Per model: request, success, error counters
  - Per worker: busy/idle state, queue length, last and average latencies, and color warning if an instance is overcharged (sustained queueing/high latency)
  - Logs viewer: switch between load balancer logs and individual worker logs

- JSON endpoints:
  - `/status` — current worker pools and their live states
  - `/metrics` — aggregated metrics (uptime, request counts, per‑model and per‑worker stats)
  - `/logs?kind=balancer` — last lines of the load balancer log
  - `/logs?kind=worker&worker_id=<ID>` — last lines of a worker log

Notes:
- Logs are written under `./logs/`. The `scripts/run_api.sh` script sets and exports `LOG_DIR` and will attempt to open the dashboard automatically with `xdg-open` when available.
- Overcharge detection is heuristic and can be adjusted in `load_balancer.py` (`LoadBalancer._is_overcharged`).

## Structured Logging

All components now emit structured JSON logs with correlation IDs and log rotation. This makes it easy to trace a single request end-to-end across the load balancer, workers, and any model sub‑processes.

- Files
  - Load balancer: `logs/load_balancer.jsonl`
  - Workers (uvicorn app): `logs/worker_<ID>_<model>.jsonl`
  - Model sub‑processes (when used): `logs/workerproc_<ID>_<model>_<subproc>.jsonl`
  - Legacy redirected stdout logs (from run script) still appear as `.log` files for completeness.

- Correlation ID
  - Incoming requests get an `X-Request-ID` (or you can pass your own). The balancer propagates it to workers.
  - Find all logs for a request by grepping for that ID.

- Rotation
  - Rotation is handled by the application (50MB per file, 10 backups by default). You can tune via env: `LOG_LEVEL`, `LOG_DIR`, `LOG_FILE_PATH`.

- Viewing logs
  - Dashboard: switch between balancer and per‑worker logs in the top‑right selector; or use `/logs` endpoints:
    - `GET /logs?kind=balancer&lines=300`
    - `GET /logs?kind=worker&worker_id=0&lines=200`
  - CLI examples:
    ```bash
    # last 100 lines of balancer logs
    tail -n 100 logs/load_balancer.jsonl | jq .

    # trace a single request ID
    grep '"correlation_id":"<REQ-ID>"' -R logs | cut -d: -f1 | sort -u
    ```

- Fields (example)
  ```json
  {
    "ts": "2025-09-19T10:13:11.123Z",
    "level": "INFO",
    "logger": "inference_api_worker",
    "msg": "API_INFER_SUCCESS",
    "event": "api_infer_success",
    "correlation_id": "7b1d3a8f2c9e4b6ba2e6f01c3a5b7c9d",
    "role": "worker",
    "service": "orchestra",
    "worker_id": "0",
    "model": "gemma3",
    "duration_s": 2.37
  }
  ```

Implementation details:
- `logging_config.py` defines the JSON formatter, correlation filters, and rotating handlers.
- `load_balancer.py` and `main.py` install correlation/ACCESS middleware and propagate `X-Request-ID` to workers.
- Model sub‑processes (`gemma3`, `qwen`, `qwen3`, `whisSent`) log to separate JSONL files and inherit the correlation ID from parent where available.

## Parallelization & GPU Sharing

- **Load balancer async concurrency**: `load_balancer.py` (FastAPI) handles many in‑flight requests concurrently using `aiohttp.ClientSession`. Multiple `POST /infer` calls are processed in parallel at the LB layer.

- **Per‑model worker pools**: The LB routes to free workers first; otherwise it enqueues per worker (`LoadBalancer.get_worker_for_model()` and `LoadBalancer.process_request()`). Each worker processes one request at a time (guarded by `worker.lock` and `worker.is_busy`), while overall parallelism comes from running multiple workers per model.

- **Parallel on the same GPU via NVIDIA MPS**: `run_api.sh` launches N workers and maps them round‑robin to GPUs with `CUDA_VISIBLE_DEVICES`. When N > number of GPUs, multiple workers share the same GPU. The script attempts to start the NVIDIA MPS control daemon (`nvidia-cuda-mps-control`) so kernels from different processes can overlap on a single GPU.

- **Model‑layer parallelism**:
  - Qwen‑VL (`models/qwen.py`) spawns a dedicated vLLM sub‑process per worker, configured with optimized batching for multimodal (max 147K image tokens).
  - Qwen3 (`models/qwen3.py`) uses MoE-optimized kernel configs for H100 GPUs with chunked prefill enabled.
  - Worker FastAPI apps (`main.py`) offload blocking model calls with `asyncio.to_thread()` so the HTTP event loop stays responsive while models run.

### v3 Model Configuration

All model parameters are configurable via environment variables:

```bash
# Qwen3 (text, MoE) - vLLM
export VLLM_MAX_MODEL_LEN=10000
export VLLM_MAX_NUM_BATCHED_TOKENS=16384
export VLLM_MAX_NUM_SEQS=256
export VLLM_ENABLE_CHUNKED_PREFILL=true
export VLLM_COALESCE_MS=5

# Qwen-VL (multimodal) - vLLM
export QWEN_VL_MAX_MODEL_LEN=32768
export QWEN_VL_MAX_NUM_BATCHED_TOKENS=32768
export QWEN_VL_MAX_NUM_SEQS=16
export QWEN_VL_ENABLE_CHUNKED_PREFILL=false  # disabled for multimodal

# Gemma3 (text, dense) - vLLM
export GEMMA_MAX_MODEL_LEN=10000
export GEMMA_MAX_NUM_BATCHED_TOKENS=16384
export GEMMA_MAX_NUM_SEQS=256
export GEMMA_ENABLE_CHUNKED_PREFILL=true

# WhisSent (ASR + Emotion) - Hugging Face pipelines
export WHISSENT_BATCH_SIZE=24
export WHISSENT_CHUNK_LENGTH_S=30.0
export WHISSENT_USE_FLASH_ATTN=true
export WHISSENT_TORCH_COMPILE=false  # experimental
```

### Demo: parallel even on the same GPU

```bash
# Launch more workers than GPUs; workers will share GPUs via MPS
./run_api.sh 4 0 0 start

# Drive concurrency and see distribution across workers
python3 tests/test_concurrency.py --models gemma3 \
  --per-model-requests 100 --concurrency-per-model 16

# Observe live in the dashboard
xdg-open http://127.0.0.1:9001/dashboard || true
```

## v3 Performance Benchmarks

Tested on NVIDIA H100 PCIe 80GB:

| Model | Type | Throughput | Latency (p50) | Concurrency |
|-------|------|-----------|---------------|-------------|
| **Qwen3** (30B MoE) | Text | 17.83 req/s | 2,847ms | 100 |
| **Qwen-VL** (7B Dense) | Multimodal | 6.78 req/s | 9,573ms | 100 |

Key optimizations:
- Chunked prefill for continuous batching
- MoE kernel configs for H100
- 95% GPU memory utilization
- Burst-aware request coalescing

## Validation & Load Testing

- **`tests/test_concurrency.py`**: per‑model concurrency with throughput and latency percentiles (p50/p90/p99) and worker distribution.
  - Run multiple models together with `--models-concurrently`.
- **`tests/simulate_users.py`**: multi‑user simulation with ramp‑up/hold/ramp‑down, configurable model mix, and a JSON report saved under `logs/`.
- **`tests/gpu_profile.py`**: High-frequency GPU utilization monitoring during tests.
- **`tests/vllm_monitor.py`**: Real-time vLLM engine metrics and statistics.

Examples:

```bash
# Concurrency test (Qwen3)
python3 tests/test_concurrency.py --models qwen3 --per-model-requests 100 --concurrency-per-model 100

# Multi‑user simulation for all active models
python3 tests/simulate_users.py --users 50 --hold-sec 60

# GPU profiling during test
python3 tests/gpu_profile.py &
python3 tests/test_concurrency.py --models qwen3 --per-model-requests 50
```

## Security & Binding Defaults

- By default the load balancer binds to `127.0.0.1:9001` (local‑only). All examples use `http://127.0.0.1:9001` and the dashboard at `/dashboard`.
- To expose the API externally, set `LOAD_BALANCER_HOST=0.0.0.0` (and ensure firewall rules allow traffic) and configure API keys in `api_keys.json`. See `docs/API_AUTHENTICATION.md` for details.