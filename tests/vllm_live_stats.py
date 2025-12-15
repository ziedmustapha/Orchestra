#!/usr/bin/env python3
"""
vLLM Live Stats - Real-time log tail monitor for Orchestra
Author: Zied Mustapha

Tails vLLM worker logs and displays real-time batch/throughput stats.
Much lighter than the full monitor - just shows the key metrics.

Usage:
  python3 tests/vllm_live_stats.py                     # Auto-detect logs
  python3 tests/vllm_live_stats.py logs/worker_0_qwen3.log
  python3 tests/vllm_live_stats.py --all               # Show all log lines
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def colorize_batch(batch_size: int) -> str:
    """Color batch size based on utilization."""
    if batch_size >= 100:
        return f"{GREEN}{BOLD}{batch_size}{RESET}"
    elif batch_size >= 50:
        return f"{GREEN}{batch_size}{RESET}"
    elif batch_size >= 20:
        return f"{YELLOW}{batch_size}{RESET}"
    else:
        return f"{RED}{batch_size}{RESET}"


def colorize_kv(kv_percent: float) -> str:
    """Color KV cache usage."""
    if kv_percent >= 50:
        return f"{GREEN}{kv_percent:.1f}%{RESET}"
    elif kv_percent >= 20:
        return f"{YELLOW}{kv_percent:.1f}%{RESET}"
    else:
        return f"{RED}{kv_percent:.1f}%{RESET}"


def parse_metrics_line(line: str) -> dict:
    """Extract metrics from vLLM log line."""
    metrics = {}
    
    # Pattern: Running: 28 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.2%
    running_match = re.search(r'Running:\s*(\d+)', line)
    pending_match = re.search(r'Pending:\s*(\d+)', line)
    swapped_match = re.search(r'Swapped:\s*(\d+)', line)
    gpu_kv_match = re.search(r'GPU KV cache usage:\s*([\d.]+)%', line)
    prompt_tput_match = re.search(r'Avg prompt throughput:\s*([\d.]+)', line)
    gen_tput_match = re.search(r'Avg generation throughput:\s*([\d.]+)', line)
    
    if running_match:
        metrics['running'] = int(running_match.group(1))
    if pending_match:
        metrics['pending'] = int(pending_match.group(1))
    if swapped_match:
        metrics['swapped'] = int(swapped_match.group(1))
    if gpu_kv_match:
        metrics['kv_cache'] = float(gpu_kv_match.group(1))
    if prompt_tput_match:
        metrics['prompt_tput'] = float(prompt_tput_match.group(1))
    if gen_tput_match:
        metrics['gen_tput'] = float(gen_tput_match.group(1))
    
    return metrics


def find_log_file() -> str:
    """Find the most recent vLLM worker log."""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        print(f"No logs directory found", file=sys.stderr)
        sys.exit(1)
    
    candidates = []
    for f in os.listdir(logs_dir):
        if "qwen3" in f.lower() and (f.endswith(".log") or f.endswith(".jsonl")):
            path = os.path.join(logs_dir, f)
            candidates.append((os.path.getmtime(path), path))
    
    if not candidates:
        print("No qwen3 worker logs found in logs/", file=sys.stderr)
        sys.exit(1)
    
    candidates.sort(reverse=True)
    return candidates[0][1]


def main():
    parser = argparse.ArgumentParser(description="Real-time vLLM stats from logs")
    parser.add_argument("logfile", nargs="?", default=None, help="Log file to tail")
    parser.add_argument("--all", "-a", action="store_true", help="Show all log lines")
    args = parser.parse_args()
    
    logfile = args.logfile or find_log_file()
    print(f"{CYAN}📊 vLLM Live Stats - Tailing: {logfile}{RESET}")
    print(f"{CYAN}   Press Ctrl+C to stop{RESET}\n")
    
    # Track max values
    max_batch = 0
    max_kv = 0.0
    sample_count = 0
    
    # Use tail -F to follow the log
    proc = subprocess.Popen(
        ["tail", "-F", "-n", "100", logfile],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )
    
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a metrics line
            if 'Running:' in line and 'Pending:' in line:
                metrics = parse_metrics_line(line)
                if metrics:
                    sample_count += 1
                    running = metrics.get('running', 0)
                    pending = metrics.get('pending', 0)
                    swapped = metrics.get('swapped', 0)
                    kv = metrics.get('kv_cache', 0)
                    prompt_tput = metrics.get('prompt_tput', 0)
                    gen_tput = metrics.get('gen_tput', 0)
                    
                    max_batch = max(max_batch, running)
                    max_kv = max(max_kv, kv)
                    
                    ts = datetime.now().strftime('%H:%M:%S')
                    print(
                        f"[{ts}] "
                        f"batch={colorize_batch(running):>12} "
                        f"pend={pending:<3} "
                        f"swap={swapped:<3} "
                        f"kv={colorize_kv(kv):>12} "
                        f"tput={prompt_tput + gen_tput:>7.1f} tok/s "
                        f"| max_batch={BOLD}{max_batch}{RESET} max_kv={max_kv:.1f}%"
                    )
            elif args.all:
                # Show other important lines
                if any(k in line.lower() for k in ['error', 'warning', 'initialized', 'loaded', 'config']):
                    print(f"{YELLOW}[LOG] {line}{RESET}")
    
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        print(f"\n{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
        print(f"{BOLD}Session Summary:{RESET}")
        print(f"  Samples: {sample_count}")
        print(f"  Max batch size seen: {GREEN}{BOLD}{max_batch}{RESET}")
        print(f"  Max KV cache usage: {max_kv:.1f}%")
        print(f"{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")


if __name__ == "__main__":
    main()
