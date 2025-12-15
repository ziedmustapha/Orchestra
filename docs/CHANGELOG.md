# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

## [3.0.0] - 2025-12-15

### Major Release: H100 GPU Optimization & vLLM Engine Overhaul

This release brings significant performance improvements through comprehensive vLLM engine optimization, better GPU memory management, and improved batching strategies.

### Added
- **vLLM Engine Configuration**: Environment variable-driven configuration for all vLLM parameters
  - `VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_BATCHED_TOKENS`, `VLLM_MAX_NUM_SEQS`
  - `VLLM_ENABLE_CHUNKED_PREFILL`, `VLLM_ENABLE_PREFIX_CACHING`
  - `VLLM_NUM_SCHEDULER_STEPS` for multi-step scheduling
  - Model-specific variants: `QWEN_VL_*` for Qwen-VL multimodal
- **Optimized Batching Strategy**: Coalescing delay for better request batching under load
  - `VLLM_COALESCE_MS` / `QWEN_VL_COALESCE_MS` for tunable batch collection windows
- **MoE Kernel Configuration**: H100 PCIe optimized kernel config for Mixture-of-Experts models (Qwen3)
- **GPU Profiling Tools**: New test utilities for monitoring GPU utilization
  - `tests/gpu_profile.py` - High-frequency GPU utilization profiler
  - `tests/vllm_monitor.py` - Real-time vLLM engine metrics
  - `tests/vllm_live_stats.py` - Live log tail monitor

### Changed
- **vLLM v0 Engine**: Migrated to vLLM v0 engine for better stability and performance
- **GPU Memory Utilization**: Increased default from 0.90 to 0.95 for maximum GPU usage
- **Single Model Per GPU**: Enforced one instance per model per GPU to avoid sub-optimal resource contention
- **Async Process Improvements**: Better asynchronous request handling with improved dispatcher
- **Chunked Prefill**: Enabled by default for text models (disabled for multimodal due to token budget)
- **Request Dispatcher**: Burst-aware mode with configurable coalescing for better batching

### Fixed
- **KV Cache Concurrency**: Resolved race conditions in KV cache management
- **Model Memory Configs**: Fixed automatic memory config overrides that caused sub-optimal allocation
- **Environment Setup**: Cleaner virtual environment setup with proper dependency isolation
- **Multimodal Token Budget**: Fixed Qwen-VL crash with high concurrency (147K image tokens handled correctly)

### Infrastructure
- **uv Package Manager**: Migrated to [uv](https://github.com/astral-sh/uv) for 10-100x faster dependency installation

### Performance
- **Qwen3 (30B MoE)**: 17.83 req/s throughput on H100 (optimized batching)
- **Qwen-VL (7B Dense)**: 6.78 req/s throughput on H100 (multimodal optimized)
- **Stable at 100+ concurrent requests**: No crashes under heavy load

---

## [2.5.0] - 2025-11-XX

### Added
- WhisSent model integration (Whisper ASR + Emotion recognition)
- Multi-user simulation test (`tests/simulate_users.py`)
- Real-time dashboard with per-worker metrics

### Changed
- Improved load balancer queue management
- Better worker health checks

---

## [2.0.0] - 2025-10-XX

### Added
- Initial public release
- Multi-model support: Gemma3, Qwen2.5-VL, Qwen3
- Load balancing with round-robin distribution
- CUDA MPS support for GPU sharing
- API key authentication
- Structured JSON logging with correlation IDs
- Real-time dashboard

### Infrastructure
- Rename project to Orchestra in README and logs
- Remove hard-coded absolute paths; use env or repo-relative paths
- Add requirements.txt and api_keys.example.json
- Add CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, initial CHANGELOG
- Expand .gitignore; portable run/stop scripts
