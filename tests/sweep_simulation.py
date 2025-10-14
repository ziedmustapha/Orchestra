#!/usr/bin/env python3
"""
Run a sweep of the simulate_users.py command for multiple user counts and
aggregate all JSON reports into a single file.

Author: Zied Mustapha

Default command template (only --users changes between runs):
  /usr/bin/python3 tests/simulate_users.py \
    --users <N> \
    --ramp-up-sec 60 --hold-sec 300 --ramp-down-sec 60 \
    --think-ms-min 15000 --think-ms-max 60000 \
    --think-dist lognormal

Usage examples:
  python3 tests/sweep_simulation.py
  python3 tests/sweep_simulation.py --users 1,10,20 --out logs/my_combined.json
  python3 tests/sweep_simulation.py --python /usr/bin/python3 --api-key <KEY>

Notes:
- This runs sequentially and may take a long time (hold-sec per run!).
- We parse the stdout of each run to extract the per-run report path
  (line: "Report saved to: <path>").
- If that line is missing, we try to find the most recent logs/sim_report_*.json
  modified after the run started.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_USERS = [1, 10, 20, 50, 100, 250, 500, 1000]
DEFAULT_PY = "/usr/bin/python3"
DEFAULT_SIM = "tests/simulate_users.py"
DEFAULT_OUT = f"logs/sweep_results_{int(time.time())}.json"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep runner for simulate_users.py")
    p.add_argument("--python", default=DEFAULT_PY, help="Python interpreter path to invoke simulate_users.py")
    p.add_argument("--script", default=DEFAULT_SIM, help="Path to tests/simulate_users.py")
    p.add_argument(
        "--users",
        default=",".join(str(u) for u in DEFAULT_USERS),
        help="Comma-separated list of user counts, e.g. '1,10,20,50,100,250,500,1000'",
    )
    p.add_argument("--ramp-up-sec", type=float, default=60.0)
    p.add_argument("--hold-sec", type=float, default=300.0)
    p.add_argument("--ramp-down-sec", type=float, default=60.0)
    p.add_argument("--think-ms-min", type=int, default=15000)
    p.add_argument("--think-ms-max", type=int, default=60000)
    p.add_argument("--think-dist", choices=["uniform", "exponential", "lognormal"], default="lognormal")
    p.add_argument("--tokens-jitter-pct", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=100)
    # Optional pass-throughs
    p.add_argument("--api-key", default=None)
    p.add_argument("--weights", default=None)
    p.add_argument("--user-weight-jitter", type=float, default=None)
    p.add_argument("--max-backoff-sec", type=float, default=None)
    p.add_argument("--qwen-image", default=None)
    p.add_argument("--whissent-audio", default=None)
    p.add_argument("--out", default=DEFAULT_OUT, help="Combined JSON output path under logs/")
    return p.parse_args(argv)


def build_cmd(args: argparse.Namespace, users: int) -> List[str]:
    cmd: List[str] = [
        args.python,
        args.script,
        "--users", str(users),
        "--ramp-up-sec", str(args.ramp_up_sec),
        "--hold-sec", str(args.hold_sec),
        "--ramp-down-sec", str(args.ramp_down_sec),
        "--think-ms-min", str(args.think_ms_min),
        "--think-ms-max", str(args.think_ms_max),
        "--think-dist", str(args.think_dist),
        "--max-tokens", str(args.max_tokens),
        "--tokens-jitter-pct", str(args.tokens_jitter_pct),
    ]
    # Pass-through optional flags if provided
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    if args.weights:
        cmd += ["--weights", args.weights]
    if args.user_weight_jitter is not None:
        cmd += ["--user-weight-jitter", str(args.user_weight_jitter)]
    if args.max_backoff_sec is not None:
        cmd += ["--max-backoff-sec", str(args.max_backoff_sec)]
    if args.qwen_image:
        cmd += ["--qwen-image", args.qwen_image]
    if args.whissent_audio:
        cmd += ["--whissent-audio", args.whissent_audio]
    return cmd


def extract_report_path(output: str) -> Optional[str]:
    # Look for a line like: Report saved to: logs/sim_report_1758617405.json
    m = re.search(r"Report saved to:\s*(.+)", output)
    if m:
        path = m.group(1).strip()
        # Remove surrounding quotes if any
        return path.strip("'\"")
    return None


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load report {path}: {e}", file=sys.stderr)
        return None


def run_one(args: argparse.Namespace, users: int) -> Dict[str, Any]:
    cmd = build_cmd(args, users)
    print("\n=== Starting run ===")
    print("Command:", " ".join(cmd))
    started_at = time.time()
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"users": users, "error": f"failed to start: {e}"}

    out = completed.stdout or ""
    err = completed.stderr or ""
    combined = out + "\n" + err
    report_path = extract_report_path(combined)

    # Fallback: pick most recent sim_report_*.json modified after start
    if not report_path:
        try:
            logs_dir = Path("logs")
            candidates = list(logs_dir.glob("sim_report_*.json"))
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for p in candidates:
                if p.stat().st_mtime >= started_at - 2:  # allow slight clock skew
                    report_path = str(p)
                    break
        except Exception:
            report_path = None

    run_entry: Dict[str, Any] = {
        "users": users,
        "returncode": completed.returncode,
        "stdout": out[-4000:],  # keep only tail to reduce size
        "stderr": err[-4000:],
        "report_path": report_path,
    }
    if report_path:
        rep = load_json(report_path)
        if rep is not None:
            run_entry["report"] = rep
    return run_entry


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        users_list = [int(x.strip()) for x in str(args.users).split(",") if x.strip()]
    except Exception:
        print("Invalid --users list. Use comma-separated integers.", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combined: Dict[str, Any] = {
        "command_template": {
            "python": args.python,
            "script": args.script,
            "fixed_flags": {
                "ramp_up_sec": args.ramp_up_sec,
                "hold_sec": args.hold_sec,
                "ramp_down_sec": args.ramp_down_sec,
                "think_ms_min": args.think_ms_min,
                "think_ms_max": args.think_ms_max,
                "think_dist": args.think_dist,
            },
        },
        "users_list": users_list,
        "runs": [],
        "started_at": int(time.time()),
    }

    try:
        for u in users_list:
            entry = run_one(args, u)
            combined["runs"].append(entry)
            # Write intermediate results for safety
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
            print(f"Saved intermediate combined results to: {out_path}")
    except KeyboardInterrupt:
        print("Interrupted. Writing partial results...")
    finally:
        combined["finished_at"] = int(time.time())
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"Combined results saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
