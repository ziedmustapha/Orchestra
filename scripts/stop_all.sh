#!/usr/bin/env bash
# Author: Zied Mustapha
# Fast stop/cleanup for all workers, balancer, and CUDA MPS
# Usage: ./stop_all.sh
# Note: May prompt for sudo when stopping NVIDIA MPS control daemon.

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Stop API services via the provided script (ignore errors)
./scripts/run_api.sh 1 1 1 1 stop || true

# Kill MPS server
pkill -9 -f "nvidia-cuda-mps-server" || true

# Stop MPS control (sudo may prompt)
echo quit | sudo nvidia-cuda-mps-control || true

# Kill worker/balancer processes by pattern (best-effort)
pkill -9 -f "uvicorn main:app" || true
pkill -9 -f "load_balancer.py" || true

# Kill vLLM cores if any
pkill -9 -f "VLLM::EngineCore" || true

# Kill MPS server again just in case
pkill -9 -f "nvidia-cuda-mps-server" || true

pkill -9 -f "$ROOT_DIR/.venvs/env3/bin/python" || true
pkill -9 -f "$ROOT_DIR/.venvs/env4/bin/python" || true
pkill -9 -f "$ROOT_DIR/.venvs/env5/bin/python" || true

echo "All services and MPS processes signaled to stop."
