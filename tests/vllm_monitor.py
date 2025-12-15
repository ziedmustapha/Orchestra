#!/usr/bin/env python3
"""
vLLM Deep Monitoring Tool for Orchestra
Author: Zied Mustapha

Provides real-time insights into vLLM engine performance:
- GPU utilization (memory, compute)
- vLLM internal metrics (batch size, KV cache, queue depth)
- Token throughput (prefill + decode)
- Request latency distribution

Usage:
  python3 tests/vllm_monitor.py                    # Monitor localhost:9001
  python3 tests/vllm_monitor.py --interval 0.5     # Faster refresh
  python3 tests/vllm_monitor.py --duration 60      # Run for 60 seconds
  python3 tests/vllm_monitor.py --csv metrics.csv  # Export to CSV
"""

import argparse
import asyncio
import csv
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:
    print("Install aiohttp: pip install aiohttp", file=sys.stderr)
    sys.exit(1)

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
    print("Warning: pynvml not installed - GPU metrics unavailable (pip install pynvml)", file=sys.stderr)


@dataclass
class GPUMetrics:
    gpu_id: int
    name: str
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    gpu_util_percent: float
    temperature_c: int
    power_draw_w: float
    power_limit_w: float


@dataclass
class VLLMMetrics:
    timestamp: float
    running_requests: int = 0
    pending_requests: int = 0
    swapped_requests: int = 0
    gpu_kv_cache_percent: float = 0.0
    cpu_kv_cache_percent: float = 0.0
    prompt_throughput_tps: float = 0.0
    generation_throughput_tps: float = 0.0
    avg_batch_size: float = 0.0
    max_batch_size: int = 0


@dataclass
class MonitorSnapshot:
    timestamp: datetime
    gpu_metrics: List[GPUMetrics] = field(default_factory=list)
    vllm_metrics: Optional[VLLMMetrics] = None
    orchestra_status: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def init_nvml():
    """Initialize NVML for GPU monitoring."""
    if not HAS_NVML:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception as e:
        print(f"NVML init failed: {e}", file=sys.stderr)
        return False


def shutdown_nvml():
    """Shutdown NVML."""
    if HAS_NVML:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def get_gpu_metrics() -> List[GPUMetrics]:
    """Collect metrics from all GPUs."""
    if not HAS_NVML:
        return []
    
    metrics = []
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except Exception:
                power = 0.0
                power_limit = 0.0
            
            metrics.append(GPUMetrics(
                gpu_id=i,
                name=name,
                memory_used_gb=mem_info.used / (1024**3),
                memory_total_gb=mem_info.total / (1024**3),
                memory_percent=(mem_info.used / mem_info.total) * 100,
                gpu_util_percent=util.gpu,
                temperature_c=temp,
                power_draw_w=power,
                power_limit_w=power_limit,
            ))
    except Exception as e:
        print(f"GPU metrics error: {e}", file=sys.stderr)
    
    return metrics


async def fetch_orchestra_status(session: aiohttp.ClientSession, url: str) -> Optional[Dict]:
    """Fetch Orchestra load balancer status."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass
    return None


async def fetch_vllm_metrics_from_logs(log_path: str) -> Optional[VLLMMetrics]:
    """Parse vLLM metrics from worker log file (last metrics line)."""
    try:
        if not os.path.exists(log_path):
            return None
        
        # Read last N lines looking for metrics
        with open(log_path, 'rb') as f:
            f.seek(0, 2)  # End
            size = f.tell()
            f.seek(max(0, size - 10000))  # Last 10KB
            lines = f.read().decode('utf-8', errors='ignore').splitlines()
        
        # Find most recent metrics line
        for line in reversed(lines):
            if 'Running:' in line and 'Pending:' in line:
                # Parse: "Running: 28 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.2%"
                metrics = VLLMMetrics(timestamp=time.time())
                
                import re
                running_match = re.search(r'Running:\s*(\d+)', line)
                pending_match = re.search(r'Pending:\s*(\d+)', line)
                swapped_match = re.search(r'Swapped:\s*(\d+)', line)
                gpu_kv_match = re.search(r'GPU KV cache usage:\s*([\d.]+)%', line)
                cpu_kv_match = re.search(r'CPU KV cache usage:\s*([\d.]+)%', line)
                prompt_tput_match = re.search(r'Avg prompt throughput:\s*([\d.]+)', line)
                gen_tput_match = re.search(r'Avg generation throughput:\s*([\d.]+)', line)
                
                if running_match:
                    metrics.running_requests = int(running_match.group(1))
                    metrics.max_batch_size = max(metrics.max_batch_size, metrics.running_requests)
                if pending_match:
                    metrics.pending_requests = int(pending_match.group(1))
                if swapped_match:
                    metrics.swapped_requests = int(swapped_match.group(1))
                if gpu_kv_match:
                    metrics.gpu_kv_cache_percent = float(gpu_kv_match.group(1))
                if cpu_kv_match:
                    metrics.cpu_kv_cache_percent = float(cpu_kv_match.group(1))
                if prompt_tput_match:
                    metrics.prompt_throughput_tps = float(prompt_tput_match.group(1))
                if gen_tput_match:
                    metrics.generation_throughput_tps = float(gen_tput_match.group(1))
                
                return metrics
        
    except Exception as e:
        pass
    return None


def find_vllm_log_files(logs_dir: str = "logs") -> List[str]:
    """Find vLLM worker log files."""
    log_files = []
    if os.path.exists(logs_dir):
        for f in os.listdir(logs_dir):
            if f.startswith("worker_") and (f.endswith(".log") or f.endswith(".jsonl")):
                log_files.append(os.path.join(logs_dir, f))
    return log_files


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_bar(value: float, max_val: float = 100, width: int = 30, char: str = "█") -> str:
    """Create a visual progress bar."""
    filled = int((value / max_val) * width) if max_val > 0 else 0
    empty = width - filled
    return f"[{char * filled}{'░' * empty}]"


def print_snapshot(snapshot: MonitorSnapshot, show_header: bool = True):
    """Print a formatted snapshot to terminal."""
    if show_header:
        clear_screen()
        print("=" * 80)
        print("  vLLM Deep Monitor - Orchestra")
        print("=" * 80)
    
    print(f"\n⏱️  Timestamp: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    
    if snapshot.error:
        print(f"\n❌ Error: {snapshot.error}")
        return
    
    # GPU Metrics
    if snapshot.gpu_metrics:
        print("\n" + "─" * 40)
        print("🖥️  GPU STATUS")
        print("─" * 40)
        for gpu in snapshot.gpu_metrics:
            mem_bar = format_bar(gpu.memory_percent, 100)
            util_bar = format_bar(gpu.gpu_util_percent, 100)
            power_pct = (gpu.power_draw_w / gpu.power_limit_w * 100) if gpu.power_limit_w > 0 else 0
            power_bar = format_bar(power_pct, 100)
            
            print(f"\n  GPU {gpu.gpu_id}: {gpu.name}")
            print(f"    Memory:  {mem_bar} {gpu.memory_used_gb:.1f}/{gpu.memory_total_gb:.1f} GB ({gpu.memory_percent:.1f}%)")
            print(f"    Compute: {util_bar} {gpu.gpu_util_percent}%")
            print(f"    Power:   {power_bar} {gpu.power_draw_w:.0f}/{gpu.power_limit_w:.0f} W")
            print(f"    Temp:    {gpu.temperature_c}°C")
    
    # vLLM Metrics
    if snapshot.vllm_metrics:
        v = snapshot.vllm_metrics
        print("\n" + "─" * 40)
        print("⚡ vLLM ENGINE METRICS")
        print("─" * 40)
        
        # Batch size visualization
        batch_bar = format_bar(v.running_requests, 256, char="▓")
        kv_bar = format_bar(v.gpu_kv_cache_percent, 100)
        
        print(f"\n  📊 BATCHING")
        print(f"    Running:   {batch_bar} {v.running_requests} reqs")
        print(f"    Pending:   {v.pending_requests} reqs")
        print(f"    Swapped:   {v.swapped_requests} reqs")
        print(f"    Max Seen:  {v.max_batch_size} reqs")
        
        print(f"\n  💾 KV CACHE")
        print(f"    GPU:       {kv_bar} {v.gpu_kv_cache_percent:.1f}%")
        print(f"    CPU:       {v.cpu_kv_cache_percent:.1f}%")
        
        print(f"\n  📈 THROUGHPUT")
        print(f"    Prefill:   {v.prompt_throughput_tps:.1f} tokens/s")
        print(f"    Decode:    {v.generation_throughput_tps:.1f} tokens/s")
        print(f"    Total:     {v.prompt_throughput_tps + v.generation_throughput_tps:.1f} tokens/s")
    
    # Orchestra Status
    if snapshot.orchestra_status:
        status = snapshot.orchestra_status
        print("\n" + "─" * 40)
        print("🎼 ORCHESTRA STATUS")
        print("─" * 40)
        
        pools = status.get("model_pools", {})
        for model, workers in pools.items():
            worker_count = len(workers) if isinstance(workers, list) else workers
            print(f"    {model}: {worker_count} worker(s)")
    
    print("\n" + "─" * 40)
    print("  Press Ctrl+C to stop")
    print("─" * 40)


class MetricsHistory:
    """Track metrics over time for analysis."""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.samples: List[MonitorSnapshot] = []
        self.max_batch_seen = 0
        self.max_kv_cache_seen = 0.0
        self.total_throughput_samples: List[float] = []
    
    def add(self, snapshot: MonitorSnapshot):
        self.samples.append(snapshot)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)
        
        if snapshot.vllm_metrics:
            v = snapshot.vllm_metrics
            self.max_batch_seen = max(self.max_batch_seen, v.running_requests)
            self.max_kv_cache_seen = max(self.max_kv_cache_seen, v.gpu_kv_cache_percent)
            self.total_throughput_samples.append(
                v.prompt_throughput_tps + v.generation_throughput_tps
            )
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {}
        
        return {
            "samples_collected": len(self.samples),
            "duration_seconds": (self.samples[-1].timestamp - self.samples[0].timestamp).total_seconds() if len(self.samples) > 1 else 0,
            "max_batch_size_seen": self.max_batch_seen,
            "max_kv_cache_percent_seen": self.max_kv_cache_seen,
            "avg_throughput_tps": sum(self.total_throughput_samples) / len(self.total_throughput_samples) if self.total_throughput_samples else 0,
            "max_throughput_tps": max(self.total_throughput_samples) if self.total_throughput_samples else 0,
        }


def export_to_csv(history: MetricsHistory, filepath: str):
    """Export metrics history to CSV."""
    if not history.samples:
        print("No data to export")
        return
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "gpu_memory_percent", "gpu_util_percent", "gpu_power_w",
            "running_requests", "pending_requests", "gpu_kv_cache_percent",
            "prompt_throughput_tps", "generation_throughput_tps"
        ])
        
        for snap in history.samples:
            gpu_mem = snap.gpu_metrics[0].memory_percent if snap.gpu_metrics else 0
            gpu_util = snap.gpu_metrics[0].gpu_util_percent if snap.gpu_metrics else 0
            gpu_power = snap.gpu_metrics[0].power_draw_w if snap.gpu_metrics else 0
            
            running = snap.vllm_metrics.running_requests if snap.vllm_metrics else 0
            pending = snap.vllm_metrics.pending_requests if snap.vllm_metrics else 0
            kv_cache = snap.vllm_metrics.gpu_kv_cache_percent if snap.vllm_metrics else 0
            prompt_tput = snap.vllm_metrics.prompt_throughput_tps if snap.vllm_metrics else 0
            gen_tput = snap.vllm_metrics.generation_throughput_tps if snap.vllm_metrics else 0
            
            writer.writerow([
                snap.timestamp.isoformat(),
                f"{gpu_mem:.2f}", f"{gpu_util:.1f}", f"{gpu_power:.1f}",
                running, pending, f"{kv_cache:.2f}",
                f"{prompt_tput:.2f}", f"{gen_tput:.2f}"
            ])
    
    print(f"Exported {len(history.samples)} samples to {filepath}")


async def monitor_loop(
    status_url: str,
    logs_dir: str,
    interval: float,
    duration: Optional[float],
    csv_path: Optional[str],
    quiet: bool = False
):
    """Main monitoring loop."""
    nvml_ok = init_nvml()
    history = MetricsHistory()
    start_time = time.time()
    running = True
    
    def handle_signal(sig, frame):
        nonlocal running
        running = False
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    log_files = find_vllm_log_files(logs_dir)
    
    async with aiohttp.ClientSession() as session:
        iteration = 0
        while running:
            iteration += 1
            
            # Check duration limit
            if duration and (time.time() - start_time) >= duration:
                break
            
            snapshot = MonitorSnapshot(timestamp=datetime.now())
            
            # Collect GPU metrics
            if nvml_ok:
                snapshot.gpu_metrics = get_gpu_metrics()
            
            # Collect Orchestra status
            snapshot.orchestra_status = await fetch_orchestra_status(session, status_url)
            
            # Collect vLLM metrics from logs
            for log_file in log_files:
                vllm_metrics = await fetch_vllm_metrics_from_logs(log_file)
                if vllm_metrics:
                    if snapshot.vllm_metrics:
                        # Merge if multiple workers
                        snapshot.vllm_metrics.running_requests += vllm_metrics.running_requests
                        snapshot.vllm_metrics.pending_requests += vllm_metrics.pending_requests
                    else:
                        snapshot.vllm_metrics = vllm_metrics
            
            history.add(snapshot)
            
            if not quiet:
                print_snapshot(snapshot)
            else:
                # Quiet mode: just print key metrics
                v = snapshot.vllm_metrics
                if v:
                    print(f"[{snapshot.timestamp.strftime('%H:%M:%S')}] "
                          f"batch={v.running_requests} pending={v.pending_requests} "
                          f"kv={v.gpu_kv_cache_percent:.1f}% "
                          f"tput={v.prompt_throughput_tps + v.generation_throughput_tps:.1f}tok/s")
            
            await asyncio.sleep(interval)
    
    shutdown_nvml()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 MONITORING SESSION SUMMARY")
    print("=" * 60)
    summary = history.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    # Export to CSV if requested
    if csv_path:
        export_to_csv(history, csv_path)
    
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Deep vLLM monitoring for Orchestra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tests/vllm_monitor.py                      # Basic monitoring
  python3 tests/vllm_monitor.py --interval 0.5       # 500ms refresh
  python3 tests/vllm_monitor.py --duration 60        # Run for 60s
  python3 tests/vllm_monitor.py --csv report.csv     # Export data
  python3 tests/vllm_monitor.py --quiet              # Minimal output
        """
    )
    parser.add_argument("--status-url", default="http://127.0.0.1:9001/status",
                        help="Orchestra status endpoint")
    parser.add_argument("--logs-dir", default="logs",
                        help="Directory containing vLLM worker logs")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Polling interval in seconds (default: 1.0)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after N seconds (default: run until Ctrl+C)")
    parser.add_argument("--csv", dest="csv_path", default=None,
                        help="Export metrics to CSV file")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output (one line per sample)")
    
    args = parser.parse_args()
    
    print("Starting vLLM Deep Monitor...")
    print(f"  Status URL: {args.status_url}")
    print(f"  Logs dir: {args.logs_dir}")
    print(f"  Interval: {args.interval}s")
    if args.duration:
        print(f"  Duration: {args.duration}s")
    print()
    
    asyncio.run(monitor_loop(
        status_url=args.status_url,
        logs_dir=args.logs_dir,
        interval=args.interval,
        duration=args.duration,
        csv_path=args.csv_path,
        quiet=args.quiet
    ))


if __name__ == "__main__":
    main()
