#!/usr/bin/env bash
# Author: Zied Mustapha

# --- Ensure we run from the repository root ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# --- Logs directory ---
# Preserve existing logs; rotation is handled by the app (see logging_config.py)
mkdir -p ./logs

# --- Python Path for src/ modules ---
export PYTHONPATH="$PWD/src"

# --- Configuration for Multiple Python Environments ---
# Define the paths to the python executables for each environment.
# The script will use these if the environment variables are not set.
# IMPORTANT: These are FALLBACKS. Best practice is to set them before running the script.
# Example: export GEMMA_PYTHON_PATH="/home/user/miniconda3/envs/env4/bin/python"
GEMMA_PYTHON_EXEC="${GEMMA_PYTHON_PATH:-$(command -v python3)}"
QWEN_PYTHON_EXEC="${QWEN_PYTHON_PATH:-$(command -v python3)}"
QWEN3_PYTHON_EXEC="${QWEN3_PYTHON_PATH:-$(command -v python3)}"
# Default python for WhisSent (adjust to your env)
WHISSENT_PYTHON_EXEC="${WHISSENT_PYTHON_PATH:-$(command -v python3)}"
# Python for the load balancer (can be either one, or a third one)
LOAD_BALANCER_PYTHON_EXEC="${LOAD_BALANCER_PYTHON_PATH:-$GEMMA_PYTHON_EXEC}"

# --- Script Parameters ---
# Number of model instances (defaults: 1 gemma, 0 qwen, 0 qwen3, 0 whissent)
GEMMA3_INSTANCES=${1:-1}
QWEN_INSTANCES=${2:-0}
QWEN3_INSTANCES=${3:-0}

# Backward-compatible parsing for optional 4th numeric argument as WHISSENT_INSTANCES
if [[ "${4:-}" =~ ^[0-9]+$ ]]; then
    WHISSENT_INSTANCES=${4:-0}
    ACTION=${5:-start}
else
    WHISSENT_INSTANCES=${WHISSENT_INSTANCES:-0}
    ACTION=${4:-start}
fi

# Accept single-argument action shorthand
if [[ "${1:-}" =~ ^(start|stop|restart|status|logs)$ ]]; then
    ACTION="${1}"
fi

# Export instance counts for the load balancer
export GEMMA3_INSTANCES
export QWEN_INSTANCES
export QWEN3_INSTANCES
export WHISSENT_INSTANCES

# --- Main Script Logic (largely unchanged) ---
if [[ "$ACTION" == "start" || "$ACTION" == "restart" ]]; then
    TOTAL_WORKERS=$((GEMMA3_INSTANCES + QWEN_INSTANCES + QWEN3_INSTANCES + WHISSENT_INSTANCES))
    if [ "$TOTAL_WORKERS" -eq 0 ]; then
        echo "No model instances configured. Set GEMMA3_INSTANCES, QWEN_INSTANCES, or QWEN3_INSTANCES > 0."
        exit 1
    fi
    
    # --- Validate: No more than 1 instance of the same model per GPU ---
    # Multiple instances of the same model on one GPU is suboptimal (no benefit, wastes memory)
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    if [ "$NUM_GPUS" -gt 0 ]; then
        VALIDATION_FAILED=0
        if [ "$GEMMA3_INSTANCES" -gt "$NUM_GPUS" ]; then
            echo "Error: GEMMA3_INSTANCES=$GEMMA3_INSTANCES exceeds NUM_GPUS=$NUM_GPUS"
            echo "       Running multiple instances of the same model on one GPU is suboptimal."
            VALIDATION_FAILED=1
        fi
        if [ "$QWEN_INSTANCES" -gt "$NUM_GPUS" ]; then
            echo "Error: QWEN_INSTANCES=$QWEN_INSTANCES exceeds NUM_GPUS=$NUM_GPUS"
            echo "       Running multiple instances of the same model on one GPU is suboptimal."
            VALIDATION_FAILED=1
        fi
        if [ "$QWEN3_INSTANCES" -gt "$NUM_GPUS" ]; then
            echo "Error: QWEN3_INSTANCES=$QWEN3_INSTANCES exceeds NUM_GPUS=$NUM_GPUS"
            echo "       Running multiple instances of the same model on one GPU is suboptimal."
            VALIDATION_FAILED=1
        fi
        if [ "$WHISSENT_INSTANCES" -gt "$NUM_GPUS" ]; then
            echo "Error: WHISSENT_INSTANCES=$WHISSENT_INSTANCES exceeds NUM_GPUS=$NUM_GPUS"
            echo "       Running multiple instances of the same model on one GPU is suboptimal."
            VALIDATION_FAILED=1
        fi
        if [ "$VALIDATION_FAILED" -eq 1 ]; then
            echo ""
            echo "Each model type can have at most 1 instance per GPU."
            echo "You have $NUM_GPUS GPU(s). Adjust your instance counts accordingly."
            exit 1
        fi
    fi
fi

HOST="${LOAD_BALANCER_HOST:-127.0.0.1}"
LOAD_BALANCER_PORT="9001"
WORKER_BASE_PORT="9100"
export WORKER_BASE_PORT
export LOAD_BALANCER_HOST="$HOST"
export LOAD_BALANCER_PORT

LOG_DIR="./logs"
mkdir -p $LOG_DIR
export LOG_DIR
PID_FILE="./api_pids.pid"
declare -a WORKER_PIDS

# (Helper functions check_and_kill_port and check_existing_instance remain the same)
check_and_kill_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Port $port is in use. Killing process(es)..."
        lsof -ti :$port -sTCP:LISTEN | xargs kill -9 2>/dev/null || true; sleep 1
    fi
}
check_existing_instance() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" | head -n 1)" 2>/dev/null; then
        echo "API appears to be already running (PID file: $PID_FILE). Stop it first: $0 stop"
        exit 1
    fi
}

start_service() {
    check_existing_instance

    # --- Pre-run check for Python executables ---
    echo "Checking Python environments..."
    if ! [ -x "$(command -v "$GEMMA_PYTHON_EXEC")" ]; then
        echo "Error: Gemma python executable not found at '$GEMMA_PYTHON_EXEC'."
        echo "Please set GEMMA_PYTHON_PATH correctly."
        exit 1
    fi
    if ! [ -x "$(command -v "$QWEN_PYTHON_EXEC")" ]; then
        echo "Error: Qwen python executable not found at '$QWEN_PYTHON_EXEC'."
        echo "Please set QWEN_PYTHON_PATH correctly."
        exit 1
    fi
    if ! [ -x "$(command -v "$QWEN3_PYTHON_EXEC")" ]; then
        echo "Error: Qwen3 python executable not found at '$QWEN3_PYTHON_EXEC'."
        echo "Please set QWEN3_PYTHON_PATH correctly."
        exit 1
    fi
    if [ "$WHISSENT_INSTANCES" -gt 0 ]; then
        if ! [ -x "$(command -v "$WHISSENT_PYTHON_EXEC")" ]; then
            echo "Error: WhisSent python executable not found at '$WHISSENT_PYTHON_EXEC'."
            echo "Please set WHISSENT_PYTHON_PATH correctly."
            exit 1
        fi
    fi
    echo "Gemma Python: $GEMMA_PYTHON_EXEC"
    echo "Qwen Python:  $QWEN_PYTHON_EXEC"
    echo "Qwen3 Python: $QWEN3_PYTHON_EXEC"
    echo "WhisSent Python: $WHISSENT_PYTHON_EXEC (if used)"
    echo "Load Balancer Python: $LOAD_BALANCER_PYTHON_EXEC"
    
    echo "Checking ports..."
    check_and_kill_port $LOAD_BALANCER_PORT
    for ((i=0; i<$TOTAL_WORKERS; i++)); do
        check_and_kill_port $((WORKER_BASE_PORT + i))
    done

    # GPU detection and MPS setup
    if ! command -v nvidia-smi &> /dev/null; then NUM_GPUS=0; else NUM_GPUS=$(nvidia-smi -L | wc -l); fi
    export NUM_GPUS; echo "Detected $NUM_GPUS GPUs"
    
    # Enable CUDA MPS for better multi-process GPU sharing (only if GPUs are present)
    if [ "$NUM_GPUS" -gt 0 ]; then
        echo "Setting up CUDA MPS for better GPU sharing..."
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true # Stop any existing MPS servers
        sleep 1
        pkill -f nvidia-cuda-mps 2>/dev/null || true # Kill any lingering MPS processes

        for ((i=0; i<$NUM_GPUS; i++)); do
            # Attempt to set to DEFAULT compute mode for MPS. This often requires root.
            if sudo nvidia-smi -i $i -c DEFAULT > /dev/null 2>&1; then
                echo "Set GPU $i to DEFAULT mode for MPS"
            else
                # If sudo fails or command fails, check current mode
                current_mode=$(nvidia-smi -i $i --query-gpu=compute_mode --format=csv,noheader)
                if [ "$current_mode" = "Default" ]; then
                    echo "GPU $i is already in DEFAULT mode."
                elif [ "$current_mode" = "Exclusive_Process" ]; then
                     echo "Warning: GPU $i is in EXCLUSIVE_PROCESS mode. MPS may not work optimally or at all."
                     echo "Consider changing to DEFAULT mode manually (requires root): sudo nvidia-smi -i $i -c DEFAULT"
                else
                    echo "Warning: Could not set GPU $i to DEFAULT mode (may require root or GPU is in an incompatible mode: $current_mode)."
                fi
            fi
        done

        # Start MPS control daemon
        if sudo nvidia-cuda-mps-control -d; then # Try with sudo
            echo "CUDA MPS daemon started successfully"
        else
            # If sudo fails, try without (might work if user has permissions or if already running)
            if nvidia-cuda-mps-control -d; then
                echo "CUDA MPS daemon started successfully (without sudo)"
            else
                echo "Warning: Could not start CUDA MPS daemon (may require root or specific permissions)."
                echo "Continuing without MPS active control, but workers will still respect CUDA_VISIBLE_DEVICES."
            fi
        fi
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps # Standard directory
    else
        echo "No GPUs detected or nvidia-smi not found. Skipping MPS setup."
    fi

    # --- Calculate models per GPU for dynamic memory allocation ---
    # Count how many models will be loaded on each GPU
    if [ "$NUM_GPUS" -gt 0 ]; then
        declare -a MODELS_PER_GPU
        for ((g=0; g<$NUM_GPUS; g++)); do
            MODELS_PER_GPU[$g]=0
        done
        for ((i=0; i<$TOTAL_WORKERS; i++)); do
            gpu_id=$((i % NUM_GPUS))
            MODELS_PER_GPU[$gpu_id]=$((MODELS_PER_GPU[$gpu_id] + 1))
        done
        echo "Models per GPU distribution:"
        for ((g=0; g<$NUM_GPUS; g++)); do
            echo "  GPU $g: ${MODELS_PER_GPU[$g]} model(s)"
        done
    fi

    echo "Starting $TOTAL_WORKERS total worker processes..."
    for ((i=0; i<$TOTAL_WORKERS; i++)); do
        worker_port=$((WORKER_BASE_PORT + i))
        if [ "$NUM_GPUS" -gt 0 ]; then
            gpu_id=$((i % NUM_GPUS))
            models_on_this_gpu=${MODELS_PER_GPU[$gpu_id]}
        else
            gpu_id=""
            models_on_this_gpu=1
        fi
        
        if [ $i -lt $GEMMA3_INSTANCES ]; then
            MODEL_TO_LOAD="gemma3"
            PYTHON_EXEC_FOR_WORKER=$GEMMA_PYTHON_EXEC
        elif [ $i -lt $((GEMMA3_INSTANCES + QWEN_INSTANCES)) ]; then
            MODEL_TO_LOAD="qwen"
            PYTHON_EXEC_FOR_WORKER=$QWEN_PYTHON_EXEC
        elif [ $i -lt $((GEMMA3_INSTANCES + QWEN_INSTANCES + QWEN3_INSTANCES)) ]; then
            MODEL_TO_LOAD="qwen3"
            PYTHON_EXEC_FOR_WORKER=$QWEN3_PYTHON_EXEC
        else
            MODEL_TO_LOAD="whissent"
            PYTHON_EXEC_FOR_WORKER=$WHISSENT_PYTHON_EXEC
        fi

        echo "Starting worker $i ($MODEL_TO_LOAD) on port $worker_port (GPU $gpu_id, sharing with $models_on_this_gpu model(s)) using ${PYTHON_EXEC_FOR_WORKER}..."
        
        if [ "$NUM_GPUS" -gt 0 ]; then
            export CUDA_VISIBLE_DEVICES=$gpu_id
        else
            unset CUDA_VISIBLE_DEVICES
        fi
        export WORKER_ID=$i
        export MODEL_TO_LOAD=$MODEL_TO_LOAD
        export MODELS_ON_GPU=$models_on_this_gpu  # For dynamic gpu_memory_utilization
        # Pass through GPU_MEMORY_OVERRIDE if set (for testing different utilization levels)
        if [ -n "$GPU_MEMORY_OVERRIDE" ]; then
            export GPU_MEMORY_OVERRIDE=$GPU_MEMORY_OVERRIDE
        fi
        
        # *** KEY CHANGE HERE ***
        # We now use the specific Python executable for the worker
        nohup $PYTHON_EXEC_FOR_WORKER -m uvicorn main:app --host $HOST --port $worker_port --workers 1 > "$LOG_DIR/worker_${i}_${MODEL_TO_LOAD}.log" 2>&1 &
        WORKER_PIDS[$i]=$!
        
        echo "Worker $i ($MODEL_TO_LOAD) started with PID ${WORKER_PIDS[$i]}"
        sleep 45
    done

    echo "All workers launched. Waiting..."
    sleep 15

    echo "Starting load balancer on port $LOAD_BALANCER_PORT..."
    # *** KEY CHANGE HERE ***
    nohup $LOAD_BALANCER_PYTHON_EXEC src/load_balancer.py > $LOG_DIR/load_balancer.log 2>&1 &
    LOAD_BALANCER_PID=$!

    echo $LOAD_BALANCER_PID > "$PID_FILE"
    for pid in "${WORKER_PIDS[@]}"; do echo $pid >> "$PID_FILE"; done

    echo ""
    echo "=========================================="
    echo "Multi-Model API started successfully!"
    echo "Load balancer: http://${HOST}:${LOAD_BALANCER_PORT}"
    echo "Dashboard:     http://${HOST}:${LOAD_BALANCER_PORT}/dashboard"
    echo "  - Gemma3 Instances:   $GEMMA3_INSTANCES (Python: $GEMMA_PYTHON_EXEC)"
    echo "  - Qwen Instances:     $QWEN_INSTANCES (Python: $QWEN_PYTHON_EXEC)"
    echo "  - Qwen3 Instances:    $QWEN3_INSTANCES (Python: $QWEN3_PYTHON_EXEC)"
    echo "  - WhisSent Instances: $WHISSENT_INSTANCES (Python: $WHISSENT_PYTHON_EXEC)"
    echo "Stop service: ./scripts/run_api.sh stop"
    echo "=========================================="

    # Try to open the dashboard if xdg-open is available (best-effort)
    if command -v xdg-open >/dev/null 2>&1; then
        (sleep 2; xdg-open "http://${HOST}:${LOAD_BALANCER_PORT}/dashboard" >/dev/null 2>&1 || true) &
    fi
}

# (stop_service, check_status, tail_logs functions remain the same)
stop_service() {
    if [ ! -f "$PID_FILE" ]; then echo "PID file not found."; return; fi
    echo "Stopping Multi-Model API service..."
    kill -9 $(cat "$PID_FILE") 2>/dev/null || true
    
    # Stop CUDA MPS if it was started
    if [ "$NUM_GPUS" -gt 0 ]; then
        echo "Stopping CUDA MPS daemon..."
        echo quit | sudo nvidia-cuda-mps-control 2>/dev/null || true
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    fi
    
    rm -f "$PID_FILE"
    echo "Service stopped."
}
check_status() {
    # (This function does not need changes)
    if [ ! -f "$PID_FILE" ]; then echo "API is not running (no PID file)."; return; fi; echo "Checking status..."; for pid in $(cat "$PID_FILE"); do if kill -0 $pid 2>/dev/null; then echo "PID $pid: RUNNING"; else echo "PID $pid: STOPPED"; fi; done
}
tail_logs() {
    tail -f $LOG_DIR/*.log
}


# Main script logic - action parsed above (supports optional 4th numeric argument)
case "$ACTION" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        sleep 2
        start_service
        ;;
    status)
        check_status
        ;;
    logs)
        tail_logs
        ;;
    *)
        echo "Usage: [ENV_VARS] $0 <gemma_instances> <qwen_instances> <qwen3_instances> [whissent_instances] [ACTION]"
        echo "------------------------------------------------------------------"
        echo "ENV_VARS (Optional):"
        echo "  GEMMA_PYTHON_PATH: Full path to the Python executable for Gemma (e.g., /path/to/env4/bin/python)"
        echo "  QWEN_PYTHON_PATH:  Full path to the Python executable for Qwen (e.g., /path/to/env3/bin/python)"
        echo "  QWEN3_PYTHON_PATH: Full path to the Python executable for Qwen3 (e.g., /path/to/env5/bin/python)"
        echo "  WHISSENT_PYTHON_PATH: Full path to the Python executable for WhisSent (e.g., /path/to/env3/bin/python)"
        echo ""
        echo "Arguments:"
        echo "  gemma_instances: Number of Gemma3 worker instances (default: 1)"
        echo "  qwen_instances:  Number of Qwen worker instances (default: 0)"
        echo "  qwen3_instances: Number of Qwen3 worker instances (default: 0)"
        echo "  whissent_instances: Number of WhisSent worker instances (default: 0). This argument is optional; if omitted, set WHISSENT_INSTANCES env var instead."
        echo ""
        echo "Examples:"
        echo "  # Start 2 Gemma3, 1 Qwen, 1 Qwen3, 1 WhisSent:"
        echo "  ./scripts/run_api.sh 2 1 1 1 start"
        echo "  "
        echo "  # Start 1 Gemma3, 0 Qwen, 2 Qwen3, (env var) 1 WhisSent:"
        echo "  WHISSENT_INSTANCES=1 ./scripts/run_api.sh 1 0 2 start"
        echo ""
        echo "Then open the dashboard at: http://${HOST}:${LOAD_BALANCER_PORT}/dashboard"
        exit 1
        ;;
esac