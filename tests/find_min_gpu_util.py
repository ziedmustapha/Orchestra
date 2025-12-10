#!/usr/bin/env python3
"""
Find minimum gpu_memory_utilization for each model that maintains peak throughput.
This helps optimize multi-model GPU sharing.

Usage:
    python3 tests/find_min_gpu_util.py --model qwen3
    python3 tests/find_min_gpu_util.py --model qwen3 --concurrency 20
    python3 tests/find_min_gpu_util.py --model qwen3 --utils "0.9,0.7,0.5,0.4,0.3"
"""

import argparse
import subprocess
import re
import sys
import time
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


ORCHESTRA_DIR = "/home/zied/Orchestra/Orchestra"


@dataclass
class TestResult:
    gpu_util: float
    throughput: float
    avg_latency_ms: float
    p99_latency_ms: float
    success: bool
    error: Optional[str] = None


def stop_service() -> bool:
    """Stop the Orchestra service."""
    print("  Stopping service...")
    try:
        result = subprocess.run(
            ["scripts/stop_all.sh"],
            cwd=ORCHESTRA_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        time.sleep(5)  # Wait for cleanup
        return True
    except Exception as e:
        print(f"  Warning: Error stopping service: {e}")
        return False


def start_service_with_util(model: str, gpu_util: float) -> bool:
    """Start the service with a specific GPU memory utilization."""
    print(f"  Starting service with gpu_memory_utilization={gpu_util}...")
    
    # Determine which models to start based on model name
    model_args = {
        "gemma3": "1 0 0 0",
        "qwen": "0 1 0 0",
        "qwen3": "0 0 1 0",
        "whissent": "0 0 0 1",
    }
    
    if model not in model_args:
        print(f"  Unknown model: {model}")
        return False
    
    # Set environment and start
    env = os.environ.copy()
    env["GPU_MEMORY_OVERRIDE"] = str(gpu_util)
    
    try:
        # Source orchestra.env and run
        cmd = f"source orchestra.env && GPU_MEMORY_OVERRIDE={gpu_util} scripts/run_api.sh {model_args[model]} start"
        result = subprocess.run(
            ["bash", "-c", cmd],
            cwd=ORCHESTRA_DIR,
            capture_output=True,
            text=True,
            timeout=180,  # 3 min for model loading
            env=env
        )
        
        if result.returncode != 0:
            print(f"  Service start failed: {result.stderr}")
            return False
        
        # Wait for service to be ready
        print("  Waiting for service to be ready...")
        time.sleep(10)
        
        # Check if service is responding
        for attempt in range(30):
            try:
                check = subprocess.run(
                    ["curl", "-s", "http://127.0.0.1:9001/status"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # Check for valid JSON response with model_pools or worker_id
                if "model_pools" in check.stdout or "worker_id" in check.stdout:
                    print("  Service is ready!")
                    return True
            except:
                pass
            time.sleep(2)
        
        print("  Service did not become ready in time")
        return False
        
    except subprocess.TimeoutExpired:
        print("  Service start timed out")
        return False
    except Exception as e:
        print(f"  Error starting service: {e}")
        return False


def run_throughput_test(model: str, concurrency: int, requests: int) -> Optional[TestResult]:
    """Run a throughput test and return results."""
    print(f"  Running test: concurrency={concurrency}, requests={requests}...")
    
    cmd = [
        "python3", "tests/test_concurrency.py",
        "--models", model,
        "--per-model-requests", str(requests),
        "--concurrency-per-model", str(concurrency),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=ORCHESTRA_DIR
        )
        output = result.stdout + result.stderr
        
        # Check for failures
        fail_match = re.search(r'Failure:\s*(\d+)', output)
        if fail_match and int(fail_match.group(1)) > 0:
            return TestResult(
                gpu_util=0, throughput=0, avg_latency_ms=0, p99_latency_ms=0,
                success=False, error=f"Test had {fail_match.group(1)} failures"
            )
        
        # Parse throughput
        thr_match = re.search(r'Throughput:\s*([\d.]+)\s*req/s', output)
        if not thr_match:
            return TestResult(
                gpu_util=0, throughput=0, avg_latency_ms=0, p99_latency_ms=0,
                success=False, error="Failed to parse throughput"
            )
        throughput = float(thr_match.group(1))
        
        # Parse latencies
        lat_match = re.search(r'Latency.*avg=([\d.]+).*p99=([\d.]+)', output)
        avg_lat = float(lat_match.group(1)) if lat_match else 0
        p99_lat = float(lat_match.group(2)) if lat_match else 0
        
        return TestResult(
            gpu_util=0,  # Will be set by caller
            throughput=throughput,
            avg_latency_ms=avg_lat,
            p99_latency_ms=p99_lat,
            success=True
        )
        
    except subprocess.TimeoutExpired:
        return TestResult(
            gpu_util=0, throughput=0, avg_latency_ms=0, p99_latency_ms=0,
            success=False, error="Test timed out"
        )
    except Exception as e:
        return TestResult(
            gpu_util=0, throughput=0, avg_latency_ms=0, p99_latency_ms=0,
            success=False, error=str(e)
        )


def find_min_util(
    model: str,
    util_levels: List[float],
    concurrency: int,
    requests: int,
    throughput_threshold: float = 0.95  # Allow 5% drop from peak
) -> Tuple[List[TestResult], float]:
    """
    Find the minimum gpu_memory_utilization that maintains throughput.
    
    Returns: (list of results, recommended minimum utilization)
    """
    results = []
    peak_throughput = 0.0
    min_viable_util = util_levels[0]  # Start with highest
    
    print(f"\n{'='*70}")
    print(f"Finding minimum gpu_memory_utilization for: {model}")
    print(f"Concurrency: {concurrency}, Requests per test: {requests}")
    print(f"Testing utilization levels: {util_levels}")
    print(f"Throughput threshold: {throughput_threshold*100:.0f}% of peak")
    print(f"{'='*70}\n")
    
    for i, util in enumerate(util_levels):
        print(f"\n[{i+1}/{len(util_levels)}] Testing gpu_memory_utilization={util}")
        print("-" * 50)
        
        # Stop any running service
        stop_service()
        
        # Start with new utilization
        if not start_service_with_util(model, util):
            result = TestResult(
                gpu_util=util, throughput=0, avg_latency_ms=0, p99_latency_ms=0,
                success=False, error="Service failed to start (likely OOM)"
            )
            results.append(result)
            print(f"  ✗ Failed to start - likely out of memory")
            continue
        
        # Run the test
        test_result = run_throughput_test(model, concurrency, requests)
        
        if test_result is None or not test_result.success:
            error = test_result.error if test_result else "Unknown error"
            result = TestResult(
                gpu_util=util, throughput=0, avg_latency_ms=0, p99_latency_ms=0,
                success=False, error=error
            )
            results.append(result)
            print(f"  ✗ Test failed: {error}")
            continue
        
        test_result.gpu_util = util
        results.append(test_result)
        
        print(f"  ✓ Throughput: {test_result.throughput:.2f} req/s, "
              f"Avg Latency: {test_result.avg_latency_ms:.0f}ms")
        
        # Track peak and minimum viable
        if test_result.throughput > peak_throughput:
            peak_throughput = test_result.throughput
        
        # Check if this utilization maintains acceptable throughput
        if peak_throughput > 0 and test_result.throughput >= peak_throughput * throughput_threshold:
            min_viable_util = util
    
    # Stop service after testing
    stop_service()
    
    return results, min_viable_util


def print_results(results: List[TestResult], min_util: float, model: str):
    """Print analysis of results."""
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'GPU Util':>10} | {'Throughput':>12} | {'Avg Latency':>12} | {'Status':>15}")
    print("-" * 70)
    
    peak_thr = max((r.throughput for r in results if r.success), default=0)
    
    for r in results:
        if r.success:
            status = "✓ OK"
            if r.throughput < peak_thr * 0.95:
                status = "⚠ Degraded"
            print(f"{r.gpu_util:>10.2f} | {r.throughput:>10.2f} req/s | {r.avg_latency_ms:>10.0f}ms | {status:>15}")
        else:
            print(f"{r.gpu_util:>10.2f} | {'N/A':>12} | {'N/A':>12} | {'✗ ' + (r.error or 'Failed')[:12]:>15}")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    print(f"\n📊 Peak Throughput: {peak_thr:.2f} req/s")
    print(f"🎯 Minimum viable gpu_memory_utilization for {model}: {min_util:.2f}")
    
    # Calculate memory saved
    if min_util < 0.9:
        saved = 0.9 - min_util
        print(f"💾 GPU memory saved vs 0.90: {saved*100:.0f}% (available for other models)")
    
    # Suggest multi-model config
    print(f"\n💡 For multi-model setup:")
    print(f"   Set {model.upper()}_GPU_UTIL={min_util:.2f} in your config")
    print(f"   Remaining capacity: {0.95 - min_util:.2f} for other models")


def main():
    parser = argparse.ArgumentParser(
        description="Find minimum gpu_memory_utilization that maintains peak throughput"
    )
    parser.add_argument("--model", required=True, 
                        choices=["gemma3", "qwen", "qwen3", "whissent"],
                        help="Model to test")
    parser.add_argument("--utils", default="0.9,0.7,0.5,0.4,0.35,0.3",
                        help="Comma-separated utilization levels to test (high to low)")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Concurrency level for testing (default: 20)")
    parser.add_argument("--requests", type=int, default=None,
                        help="Requests per test (default: same as concurrency)")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Throughput threshold as fraction of peak (default: 0.95)")
    
    args = parser.parse_args()
    
    util_levels = [float(x.strip()) for x in args.utils.split(",")]
    util_levels.sort(reverse=True)  # Ensure high to low
    
    requests = args.requests if args.requests else args.concurrency
    
    try:
        results, min_util = find_min_util(
            model=args.model,
            util_levels=util_levels,
            concurrency=args.concurrency,
            requests=requests,
            throughput_threshold=args.threshold
        )
        
        if results:
            print_results(results, min_util, args.model)
        else:
            print("No successful tests.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted. Stopping service...")
        stop_service()
        sys.exit(1)


if __name__ == "__main__":
    main()
