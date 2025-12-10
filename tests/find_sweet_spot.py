#!/usr/bin/env python3
"""
Find the optimal concurrency sweet spot for a model.
Identifies whether the system is KV-bound or compute-bound.
"""

import argparse
import subprocess
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestResult:
    concurrency: int
    throughput: float
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    success_rate: float


def run_concurrency_test(model: str, concurrency: int, requests: int) -> Optional[TestResult]:
    """Run a single concurrency test and parse results."""
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
            timeout=600,  # 10 minute timeout
            cwd="/home/zied/Orchestra/Orchestra"
        )
        output = result.stdout + result.stderr
        
        # Parse throughput
        thr_match = re.search(r'Throughput:\s*([\d.]+)\s*req/s', output)
        if not thr_match:
            print(f"  Failed to parse throughput from output")
            return None
        throughput = float(thr_match.group(1))
        
        # Parse latencies
        lat_match = re.search(r'Latency.*avg=([\d.]+).*p50=([\d.]+).*p99=([\d.]+)', output)
        if not lat_match:
            print(f"  Failed to parse latency from output")
            return None
        avg_lat = float(lat_match.group(1))
        p50_lat = float(lat_match.group(2))
        p99_lat = float(lat_match.group(3))
        
        # Parse success rate
        success_match = re.search(r'Success:\s*(\d+)', output)
        total_match = re.search(r'Total:\s*(\d+)', output)
        if success_match and total_match:
            success_rate = int(success_match.group(1)) / int(total_match.group(1))
        else:
            success_rate = 1.0
        
        return TestResult(
            concurrency=concurrency,
            throughput=throughput,
            avg_latency_ms=avg_lat,
            p50_latency_ms=p50_lat,
            p99_latency_ms=p99_lat,
            success_rate=success_rate
        )
        
    except subprocess.TimeoutExpired:
        print(f"  Test timed out at concurrency {concurrency}")
        return None
    except Exception as e:
        print(f"  Error running test: {e}")
        return None


def find_sweet_spot(model: str, concurrency_levels: List[int], requests_per_level: int) -> List[TestResult]:
    """Run tests at multiple concurrency levels and find the sweet spot."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Finding sweet spot for model: {model}")
    print(f"Testing concurrency levels: {concurrency_levels}")
    print(f"Requests per test: {requests_per_level}")
    print(f"{'='*60}\n")
    
    for i, conc in enumerate(concurrency_levels):
        print(f"[{i+1}/{len(concurrency_levels)}] Testing concurrency={conc}...")
        
        result = run_concurrency_test(model, conc, requests_per_level)
        
        if result:
            results.append(result)
            print(f"  ✓ Throughput: {result.throughput:.2f} req/s, "
                  f"Avg Latency: {result.avg_latency_ms:.0f}ms, "
                  f"P99: {result.p99_latency_ms:.0f}ms")
        else:
            print(f"  ✗ Test failed")
        
        # Small delay between tests
        if i < len(concurrency_levels) - 1:
            time.sleep(2)
    
    return results


def analyze_results(results: List[TestResult]) -> None:
    """Analyze results and identify the sweet spot."""
    if len(results) < 2:
        print("\nNot enough data points to analyze.")
        return
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Conc':>6} | {'Thr (req/s)':>12} | {'Avg Lat':>10} | {'P99 Lat':>10} | {'Lat/Conc':>10}")
    print("-" * 60)
    
    for r in results:
        lat_per_conc = r.avg_latency_ms / r.concurrency
        print(f"{r.concurrency:>6} | {r.throughput:>12.2f} | {r.avg_latency_ms:>9.0f}ms | {r.p99_latency_ms:>9.0f}ms | {lat_per_conc:>9.0f}ms")
    
    # Find peak throughput
    peak_result = max(results, key=lambda r: r.throughput)
    peak_idx = results.index(peak_result)
    
    # Detect plateau (throughput stops increasing significantly)
    plateau_threshold = 0.05  # 5% improvement threshold
    sweet_spot = results[0]
    
    for i in range(1, len(results)):
        improvement = (results[i].throughput - results[i-1].throughput) / results[i-1].throughput
        if improvement > plateau_threshold:
            sweet_spot = results[i]
        else:
            # Throughput plateaued, previous point is sweet spot
            break
    
    # Check if we're KV-bound or compute-bound
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\n📊 Peak Throughput: {peak_result.throughput:.2f} req/s at concurrency {peak_result.concurrency}")
    print(f"🎯 Sweet Spot: concurrency {sweet_spot.concurrency} ({sweet_spot.throughput:.2f} req/s)")
    
    # Determine regime
    first = results[0]
    last = results[-1]
    throughput_growth = (last.throughput - first.throughput) / first.throughput * 100
    
    print(f"\n📈 Throughput growth from c={first.concurrency} to c={last.concurrency}: {throughput_growth:+.1f}%")
    
    if throughput_growth > 20:
        regime = "KV-BOUND"
        advice = "More KV cache would help. Consider increasing gpu_memory_utilization or reducing max_model_len."
    elif throughput_growth > 5:
        regime = "TRANSITIONING"
        advice = "Approaching compute-bound. Current config is reasonable."
    else:
        regime = "COMPUTE-BOUND"
        advice = "GPU is the bottleneck. More KV cache won't help much. Consider quantization or faster GPU."
    
    print(f"\n🔍 Current Regime: {regime}")
    print(f"💡 Advice: {advice}")
    
    # Latency analysis
    print(f"\n📉 Latency Scaling:")
    print(f"   At c=1 equivalent: {first.avg_latency_ms:.0f}ms")
    print(f"   At c={last.concurrency}: {last.avg_latency_ms:.0f}ms (per request)")
    print(f"   Effective per-request at c={last.concurrency}: {last.avg_latency_ms/last.concurrency:.0f}ms")
    
    # Efficiency metric
    efficiency = (last.avg_latency_ms / last.concurrency) / first.avg_latency_ms * 100
    print(f"   Batching efficiency: {efficiency:.0f}% of single-request latency")


def main():
    parser = argparse.ArgumentParser(description="Find optimal concurrency sweet spot")
    parser.add_argument("--model", default="qwen3", help="Model to test (default: qwen3)")
    parser.add_argument("--levels", default="1,2,5,10,20,30,50", 
                        help="Comma-separated concurrency levels to test")
    parser.add_argument("--requests", type=int, default=None,
                        help="Requests per test (default: same as concurrency)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with fewer levels: 1,5,10,20")
    
    args = parser.parse_args()
    
    if args.quick:
        concurrency_levels = [1, 5, 10, 20]
    else:
        concurrency_levels = [int(x.strip()) for x in args.levels.split(",")]
    
    results = []
    for conc in concurrency_levels:
        requests = args.requests if args.requests else conc
        print(f"[{concurrency_levels.index(conc)+1}/{len(concurrency_levels)}] Testing concurrency={conc}...")
        
        result = run_concurrency_test(args.model, conc, requests)
        
        if result:
            results.append(result)
            print(f"  ✓ Throughput: {result.throughput:.2f} req/s, "
                  f"Avg Latency: {result.avg_latency_ms:.0f}ms, "
                  f"P99: {result.p99_latency_ms:.0f}ms")
        else:
            print(f"  ✗ Test failed")
        
        time.sleep(2)
    
    if results:
        analyze_results(results)
    else:
        print("No successful tests to analyze.")
        sys.exit(1)


if __name__ == "__main__":
    main()
