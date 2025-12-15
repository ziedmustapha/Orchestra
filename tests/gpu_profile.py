#!/usr/bin/env python3
"""
GPU Profiling Tool for Orchestra vLLM Load Tests
Author: Zied Mustapha

Captures high-frequency GPU metrics during load tests to prove H100 utilization.
Run this alongside test_concurrency.py to see real GPU behavior.

Usage:
  # Terminal 1: Start profiler
  python3 tests/gpu_profile.py --duration 30 --output gpu_metrics.csv

  # Terminal 2: Run load test
  python3 tests/test_concurrency.py --models qwen3 --per-model-requests 100 --concurrency-per-model 100
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


@dataclass
class GPUSample:
    timestamp: float
    gpu_id: int
    sm_util: int          # SM (compute) utilization %
    mem_util: int         # Memory controller utilization %
    memory_used_mb: int
    memory_total_mb: int
    power_w: float
    temp_c: int
    pcie_tx_mb: float     # PCIe TX throughput
    pcie_rx_mb: float     # PCIe RX throughput


class GPUProfiler:
    def __init__(self, gpu_ids: Optional[List[int]] = None):
        self.gpu_ids = gpu_ids
        self.samples: List[GPUSample] = []
        self.running = False
        
        if HAS_NVML:
            pynvml.nvmlInit()
            if self.gpu_ids is None:
                self.gpu_ids = list(range(pynvml.nvmlDeviceGetCount()))
        else:
            print("Warning: pynvml not available, using nvidia-smi fallback", file=sys.stderr)
            self.gpu_ids = self.gpu_ids or [0]
    
    def sample_nvml(self) -> List[GPUSample]:
        """High-frequency sampling via NVML."""
        samples = []
        ts = time.time()
        
        for gpu_id in self.gpu_ids:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except:
                    power = 0.0
                
                try:
                    pcie = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                    pcie_tx = pcie / (1024 * 1024)  # to MB/s
                    pcie = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                    pcie_rx = pcie / (1024 * 1024)
                except:
                    pcie_tx = pcie_rx = 0.0
                
                samples.append(GPUSample(
                    timestamp=ts,
                    gpu_id=gpu_id,
                    sm_util=util.gpu,
                    mem_util=util.memory,
                    memory_used_mb=mem.used // (1024 * 1024),
                    memory_total_mb=mem.total // (1024 * 1024),
                    power_w=power,
                    temp_c=temp,
                    pcie_tx_mb=pcie_tx,
                    pcie_rx_mb=pcie_rx,
                ))
            except Exception as e:
                print(f"Error sampling GPU {gpu_id}: {e}", file=sys.stderr)
        
        return samples
    
    def sample_nvidia_smi(self) -> List[GPUSample]:
        """Fallback sampling via nvidia-smi."""
        samples = []
        ts = time.time()
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpu_id = int(parts[0])
                    if self.gpu_ids and gpu_id not in self.gpu_ids:
                        continue
                    
                    samples.append(GPUSample(
                        timestamp=ts,
                        gpu_id=gpu_id,
                        sm_util=int(parts[1]) if parts[1] != '[N/A]' else 0,
                        mem_util=int(parts[2]) if parts[2] != '[N/A]' else 0,
                        memory_used_mb=int(parts[3]) if parts[3] != '[N/A]' else 0,
                        memory_total_mb=int(parts[4]) if parts[4] != '[N/A]' else 0,
                        power_w=float(parts[5]) if parts[5] != '[N/A]' else 0,
                        temp_c=int(parts[6]) if parts[6] != '[N/A]' else 0,
                        pcie_tx_mb=0,
                        pcie_rx_mb=0,
                    ))
        except Exception as e:
            print(f"nvidia-smi error: {e}", file=sys.stderr)
        
        return samples
    
    def sample(self) -> List[GPUSample]:
        if HAS_NVML:
            return self.sample_nvml()
        return self.sample_nvidia_smi()
    
    def run(self, duration: float, interval: float = 0.1):
        """Run profiling for specified duration."""
        self.running = True
        start = time.time()
        sample_count = 0
        
        print(f"Profiling GPUs {self.gpu_ids} for {duration}s at {1/interval:.0f} Hz...")
        
        while self.running and (time.time() - start) < duration:
            new_samples = self.sample()
            self.samples.extend(new_samples)
            sample_count += len(new_samples)
            
            # Progress indicator every second
            elapsed = time.time() - start
            if int(elapsed) > int(elapsed - interval):
                avg_sm = sum(s.sm_util for s in new_samples) / len(new_samples) if new_samples else 0
                avg_power = sum(s.power_w for s in new_samples) / len(new_samples) if new_samples else 0
                print(f"\r[{elapsed:5.1f}s] samples={sample_count} SM={avg_sm:3.0f}% power={avg_power:5.0f}W", end="", flush=True)
            
            time.sleep(interval)
        
        print(f"\nCollected {len(self.samples)} samples")
    
    def stop(self):
        self.running = False
    
    def get_summary(self) -> dict:
        """Calculate summary statistics."""
        if not self.samples:
            return {}
        
        sm_utils = [s.sm_util for s in self.samples]
        mem_utils = [s.mem_util for s in self.samples]
        powers = [s.power_w for s in self.samples if s.power_w > 0]
        
        return {
            "total_samples": len(self.samples),
            "duration_s": self.samples[-1].timestamp - self.samples[0].timestamp if len(self.samples) > 1 else 0,
            "sm_util_avg": sum(sm_utils) / len(sm_utils),
            "sm_util_max": max(sm_utils),
            "sm_util_p95": sorted(sm_utils)[int(len(sm_utils) * 0.95)] if sm_utils else 0,
            "mem_util_avg": sum(mem_utils) / len(mem_utils),
            "power_avg_w": sum(powers) / len(powers) if powers else 0,
            "power_max_w": max(powers) if powers else 0,
            "memory_used_max_mb": max(s.memory_used_mb for s in self.samples),
        }
    
    def export_csv(self, filepath: str):
        """Export samples to CSV."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "gpu_id", "sm_util", "mem_util", 
                "memory_used_mb", "memory_total_mb", "power_w", "temp_c",
                "pcie_tx_mb", "pcie_rx_mb"
            ])
            for s in self.samples:
                writer.writerow([
                    f"{s.timestamp:.3f}", s.gpu_id, s.sm_util, s.mem_util,
                    s.memory_used_mb, s.memory_total_mb, f"{s.power_w:.1f}", s.temp_c,
                    f"{s.pcie_tx_mb:.1f}", f"{s.pcie_rx_mb:.1f}"
                ])
        print(f"Exported {len(self.samples)} samples to {filepath}")
    
    def cleanup(self):
        if HAS_NVML:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def print_summary(summary: dict):
    """Pretty print the summary."""
    print("\n" + "=" * 60)
    print("GPU PROFILING SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 COMPUTE UTILIZATION (SM)")
    print(f"   Average:  {summary.get('sm_util_avg', 0):5.1f}%")
    print(f"   Peak:     {summary.get('sm_util_max', 0):5.1f}%")
    print(f"   P95:      {summary.get('sm_util_p95', 0):5.1f}%")
    
    print(f"\n⚡ POWER")
    print(f"   Average:  {summary.get('power_avg_w', 0):5.0f} W")
    print(f"   Peak:     {summary.get('power_max_w', 0):5.0f} W")
    
    print(f"\n💾 MEMORY")
    print(f"   Peak Used: {summary.get('memory_used_max_mb', 0):,} MB")
    print(f"   Mem Util:  {summary.get('mem_util_avg', 0):5.1f}%")
    
    # Assessment
    sm_avg = summary.get('sm_util_avg', 0)
    print(f"\n📈 ASSESSMENT")
    if sm_avg >= 80:
        print(f"   ✅ EXCELLENT - GPU compute is well utilized ({sm_avg:.0f}% avg)")
    elif sm_avg >= 50:
        print(f"   ⚠️  MODERATE - GPU could be better utilized ({sm_avg:.0f}% avg)")
    elif sm_avg >= 20:
        print(f"   ⚠️  LOW - Significant GPU headroom available ({sm_avg:.0f}% avg)")
    else:
        print(f"   ❌ UNDERUTILIZED - GPU is mostly idle ({sm_avg:.0f}% avg)")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="GPU profiler for vLLM load tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tests/gpu_profile.py --duration 30
  python3 tests/gpu_profile.py --duration 60 --interval 0.05 --output metrics.csv
  python3 tests/gpu_profile.py --gpu 0 --duration 30
        """
    )
    parser.add_argument("--duration", "-d", type=float, default=30,
                        help="Profiling duration in seconds (default: 30)")
    parser.add_argument("--interval", "-i", type=float, default=0.1,
                        help="Sampling interval in seconds (default: 0.1)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV file path")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=None,
                        help="GPU IDs to profile (default: all)")
    
    args = parser.parse_args()
    
    profiler = GPUProfiler(gpu_ids=args.gpu)
    
    def signal_handler(sig, frame):
        print("\nStopping profiler...")
        profiler.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"Starting GPU profiler...")
    print(f"  Duration: {args.duration}s")
    print(f"  Interval: {args.interval}s ({1/args.interval:.0f} samples/sec)")
    print(f"  GPUs: {profiler.gpu_ids}")
    print()
    
    profiler.run(duration=args.duration, interval=args.interval)
    
    summary = profiler.get_summary()
    print_summary(summary)
    
    if args.output:
        profiler.export_csv(args.output)
    
    profiler.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
