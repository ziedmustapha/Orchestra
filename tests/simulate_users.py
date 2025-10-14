#!/usr/bin/env python3
"""
Multi-user simulation for the ORCHESTRA load balancer.
- Spawns virtual users with ramp-up -> hold -> ramp-down.
- Each user sends requests to POST /infer across available models.
- Scenarios: Gemma3 text, Qwen multimodal (image), Qwen3 text, WhisSent audio+emotion.
- Auto-discovers active models from GET /status; adapts traffic mix.
- Random think-time between iterations.
- Aggregates per-model metrics and writes a JSON report to ./logs/.

Examples:
  python3 tests/simulate_users.py --users 30 --ramp-up-sec 10 --hold-sec 60 --ramp-down-sec 10
  python3 tests/simulate_users.py --users 50 --weights gemma3:0.5,qwen:0.2,qwen3:0.2,whissent:0.1 --api-key <KEY>
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import string
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except Exception:
    print("This simulation requires aiohttp. Please install it (e.g., pip install aiohttp).", file=sys.stderr)
    raise

DEFAULT_INFER_URL = "http://127.0.0.1:9001/infer"
DEFAULT_STATUS_URL = "http://127.0.0.1:9001/status"
DEFAULT_IMAGE = "tests/image.png"
DEFAULT_AUDIO_CANDIDATES = [
    "tests/audio.mp3",
]

FALLBACK_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/az2n0sAAAAASUVORK5CYII="
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
ACTIVE_MODELS = ("qwen", "qwen3", "whissent")

# A small set of varied prompts to diversify requests for text models
PROMPTS = [
    "Write a concise (~50 words) summary about GPU scheduling strategies.",
    "Explain the difference between tensor parallelism and pipeline parallelism in simple terms.",
    "Given N requests and K GPUs, propose ways to improve inference throughput.",
    "Briefly describe consistency distillation and why it matters.",
]


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Sample:
    ok: bool
    status: int
    duration_ms: float
    model: str
    worker_id: Optional[int]
    err: Optional[str] = None


@dataclass
class ModelStats:
    name: str
    durations_ok: List[float] = field(default_factory=list)
    successes: int = 0
    failures: int = 0
    worker_counts: Dict[int, int] = field(default_factory=dict)

    def record(self, s: Sample) -> None:
        if s.ok:
            self.successes += 1
            self.durations_ok.append(s.duration_ms)
            if s.worker_id is not None:
                self.worker_counts[s.worker_id] = self.worker_counts.get(s.worker_id, 0) + 1
        else:
            self.failures += 1

    def pct(self, p: float) -> float:
        d = self.durations_ok
        if not d:
            return 0.0
        d = sorted(d)
        k = (len(d) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(d) - 1)
        if f == c:
            return d[f]
        return d[f] + (d[c] - d[f]) * (k - f)

    def summary(self, elapsed_s: float) -> Dict[str, Any]:
        d = self.durations_ok
        total = self.successes + self.failures
        avg = (sum(d) / len(d)) if d else 0.0
        rps = (self.successes / elapsed_s) if elapsed_s > 0 else 0.0
        return {
            "model": self.name,
            "total": total,
            "success": self.successes,
            "failure": self.failures,
            "throughput_rps": round(rps, 2),
            "avg_ms": round(avg, 1),
            "p50_ms": round(self.pct(50), 1),
            "p90_ms": round(self.pct(90), 1),
            "p99_ms": round(self.pct(99), 1),
            "workers_used": sorted(list(self.worker_counts.keys())),
            "worker_distribution": {int(k): int(v) for k, v in sorted(self.worker_counts.items())},
        }


class Aggregator:
    def __init__(self, active_models: List[str]):
        self.t0 = time.perf_counter()
        self.models: Dict[str, ModelStats] = {m: ModelStats(name=m) for m in active_models}
        self._lock = asyncio.Lock()

    async def record(self, s: Sample) -> None:
        async with self._lock:
            if s.model not in self.models:
                self.models[s.model] = ModelStats(name=s.model)
            self.models[s.model].record(s)

    def build_report(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        elapsed = time.perf_counter() - self.t0
        summaries = [ms.summary(elapsed) for ms in self.models.values()]
        overall = {
            "total": sum(ms.successes + ms.failures for ms in self.models.values()),
            "success": sum(ms.successes for ms in self.models.values()),
            "failure": sum(ms.failures for ms in self.models.values()),
            "elapsed_s": round(elapsed, 2),
        }
        return {"meta": meta, "overall": overall, "per_model": summaries}


def parse_weights(s: Optional[str]) -> Dict[str, float]:
    w = {m: 0.0 for m in ACTIVE_MODELS}
    if not s:
        return w
    for item in s.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        k, v = item.split(":", 1)
        if k in w:
            try:
                w[k] = float(v)
            except ValueError:
                pass
    return w


def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    t = sum(v for v in w.values() if v > 0)
    if t <= 0:
        return {k: 0.0 for k in w}
    return {k: (v / t) if v > 0 else 0.0 for k, v in w.items()}


async def fetch_status(url: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            return await resp.json()


def encode_image_b64(path_or_dir: Optional[str]) -> str:
    if not path_or_dir:
        if os.path.exists(DEFAULT_IMAGE):
            try:
                with open(DEFAULT_IMAGE, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return FALLBACK_PNG_BASE64
        return FALLBACK_PNG_BASE64
    p = Path(path_or_dir)
    try:
        if p.is_dir():
            cands = [f for f in p.glob("**/*") if f.suffix.lower() in IMAGE_EXTS]
            if not cands:
                return FALLBACK_PNG_BASE64
            choice = random.choice(cands)
        else:
            choice = p
        with open(choice, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return FALLBACK_PNG_BASE64


def pick_audio_path(path_or_dir: Optional[str]) -> Optional[str]:
    if path_or_dir:
        p = Path(path_or_dir)
        if p.is_dir():
            cands = [f for f in p.glob("**/*") if f.suffix.lower() in AUDIO_EXTS]
            if cands:
                return str(random.choice(cands))
        elif p.exists():
            return str(p)
    for c in DEFAULT_AUDIO_CANDIDATES:
        if os.path.exists(c):
            return c
    if os.path.exists("tests/audio.mp3"):
        return "tests/audio.mp3"
    return None


def build_payload(model: str, session_id: str, max_tokens: int, image_b64: Optional[str], audio_path: Optional[str]) -> Dict[str, Any]:
    if model == "whissent":
        if not audio_path:
            raise ValueError("WhisSent requires an audio file")
        rb = {
            "audio": {"path": audio_path},
            "return_timestamps": True,
            "task": "transcribe",
            "language": "fr",
            "emotion_top_k": 3,
            "session_id": session_id,
        }
        return {"model_name": model, "request_body": rb}
    rb: Dict[str, Any] = {
        "input": random.choice(PROMPTS) + f" (session={session_id})",
        "max_new_tokens": max(1, int(max_tokens)),
        "session_id": session_id,
    }
    if model == "qwen":
        img_b64 = image_b64 or FALLBACK_PNG_BASE64
        rb["images"] = [{"type": "image", "data": img_b64, "format": "png"}]
    return {"model_name": model, "request_body": rb}


async def post_infer(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], api_key: Optional[str], request_id: Optional[str] = None) -> Tuple[int, Dict[str, Any] | str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    if request_id:
        headers["X-Request-ID"] = request_id
    try:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=3000)) as resp:
            status = resp.status
            try:
                data = await resp.json()
            except Exception:
                data = await resp.text()
            return status, data
    except Exception as e:
        return 0, str(e)


class VirtualUser:
    def __init__(
        self,
        user_id: int,
        api_url: str,
        api_key: Optional[str],
        model_weights: Dict[str, float],
        active_models: List[str],
        max_tokens: int,
        think_ms_min: int,
        think_ms_max: int,
        think_dist: str,
        tokens_jitter_pct: float,
        user_weight_jitter: float,
        max_backoff_sec: float,
        qwen_image_src: Optional[str],
        whissent_audio_src: Optional[str],
        aggregator: Aggregator,
    ):
        self.user_id = user_id
        self.api_url = api_url
        self.api_key = api_key
        # Base weights provided by controller
        base_weights = {m: model_weights.get(m, 0.0) for m in active_models}
        # Per-user preference jitter (Gaussian, clamp to >= 0)
        if user_weight_jitter > 0:
            jittered = {}
            for m, w in base_weights.items():
                factor = max(0.0, 1.0 + random.gauss(0.0, user_weight_jitter))
                jittered[m] = max(0.0, w * factor)
            # Renormalize; if all zero, fallback to uniform
            s = sum(jittered.values())
            if s <= 0:
                jittered = {m: 1.0 for m in active_models}
                s = float(len(active_models))
            self.model_weights = {m: (v / s) for m, v in jittered.items()}
        else:
            self.model_weights = base_weights
        self.active_models = active_models
        self.max_tokens = max_tokens
        self.think_ms_min = think_ms_min
        self.think_ms_max = max(think_ms_max, think_ms_min)
        self.think_dist = think_dist
        self.tokens_jitter_pct = max(0.0, min(1.0, tokens_jitter_pct))
        self.max_backoff_sec = max(0.0, max_backoff_sec)
        self.qwen_image_src = qwen_image_src
        self.whissent_audio_src = whissent_audio_src
        self.aggregator = aggregator
        self.session_id = f"user-{user_id}-{int(time.time()*1000)}"
        self._stop = asyncio.Event()
        self._image_b64_cache: Optional[str] = None
        if "qwen" in self.active_models:
            self._image_b64_cache = encode_image_b64(self.qwen_image_src)
        self._req_count = 0
        self._consecutive_errors = 0

    def stop(self):
        self._stop.set()

    def _sample_think_ms(self) -> int:
        lo, hi = self.think_ms_min, self.think_ms_max
        if self.think_dist == "uniform" or hi <= lo:
            return random.randint(lo, hi)
        mean = (lo + hi) / 2.0
        if self.think_dist == "exponential":
            # Exponential with mean around (lo+hi)/2, clamped to [lo, hi]
            val = int(random.expovariate(1.0 / max(1.0, mean)))
            return max(lo, min(hi, val))
        if self.think_dist == "lognormal":
            # Choose a sigma that yields a modest tail; center around 'mean'
            sigma = 0.9
            mu = math.log(max(1.0, mean)) - 0.5 * sigma * sigma
            val = int(random.lognormvariate(mu, sigma))
            return max(lo, min(hi, val))
        # Fallback
        return random.randint(lo, hi)

    def _sample_tokens(self) -> int:
        base = max(1, int(self.max_tokens))
        jitter = self.tokens_jitter_pct
        if jitter <= 0:
            return base
        factor = 1.0 + random.uniform(-jitter, jitter)
        # Clamp tokens to a reasonable range
        t = int(base * factor)
        return max(16, min(max(32, base * (1 + int(jitter > 0))), max(base, t)))

    async def run(self):
        weights = [self.model_weights.get(m, 0.0) for m in self.active_models]
        if sum(weights) <= 0:
            weights = [1.0 / len(self.active_models)] * len(self.active_models)
        image_b64 = self._image_b64_cache
        async with aiohttp.ClientSession() as session:
            while not self._stop.is_set():
                model = random.choices(self.active_models, weights=weights, k=1)[0]
                audio_path = pick_audio_path(self.whissent_audio_src) if model == "whissent" else None
                req_tokens = self._sample_tokens()
                payload = build_payload(model, self.session_id, req_tokens, image_b64, audio_path)
                # Build a stable request ID to correlate across logs in the dashboard
                self._req_count += 1
                req_id = f"{self.session_id}-{self._req_count}"
                t0 = time.perf_counter()
                status, data = await post_infer(session, self.api_url, payload, self.api_key, request_id=req_id)
                dt = (time.perf_counter() - t0) * 1000.0
                if isinstance(data, dict) and 200 <= status < 300:
                    worker_id = data.get("load_balancer_worker_id")
                    mproc = data.get("model_name_processed") or model
                    await self.aggregator.record(Sample(True, status, dt, mproc, worker_id))
                    self._consecutive_errors = 0
                else:
                    err_str = data if isinstance(data, str) else json.dumps(data)
                    await self.aggregator.record(Sample(False, status, dt, model, None, err_str[:300]))
                    # On errors, apply exponential backoff before next iteration
                    self._consecutive_errors += 1
                    backoff = min(self.max_backoff_sec, 0.5 * (2 ** min(self._consecutive_errors, 6)))
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                    except asyncio.TimeoutError:
                        pass
                to_sleep = random.randint(self.think_ms_min, self.think_ms_max) / 1000.0
                # Use the configured think-time distribution
                to_sleep = self._sample_think_ms() / 1000.0
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=to_sleep)
                except asyncio.TimeoutError:
                    pass


async def run_simulation(args) -> Dict[str, Any]:
    status = await fetch_status(args.status_url)
    model_pools = status.get("model_pools", {})
    active_models = [m for m in ACTIVE_MODELS if model_pools.get(m)]
    if not active_models:
        print("No active models discovered in /status. Exiting.")
        return {"error": "no_active_models"}

    user_weights = parse_weights(args.weights)
    if not args.weights:
        # Default mix: make WhisSent much rarer than other models
        # Rationale: audio inference is heavier and often less frequent in typical workloads
        for m in active_models:
            if m == "whissent":
                user_weights[m] = 0.05  # rare by default (~1-2% if 3 other models active)
            else:
                user_weights[m] = 1.0
    for m in ACTIVE_MODELS:
        if m not in active_models:
            user_weights[m] = 0.0
    weights = normalize_weights(user_weights)

    meta = {
        "started_at": now_ts(),
        "api_url": args.api_url,
        "status_url": args.status_url,
        "users": int(args.users),
        "ramp_up_sec": float(args.ramp_up_sec),
        "hold_sec": float(args.hold_sec),
        "ramp_down_sec": float(args.ramp_down_sec),
        "think_time_ms": [int(args.think_ms_min), int(args.think_ms_max)],
        "think_dist": str(args.think_dist),
        "max_tokens": int(args.max_tokens),
        "tokens_jitter_pct": float(args.tokens_jitter_pct),
        "user_weight_jitter": float(args.user_weight_jitter),
        "active_models": active_models,
        "weights": weights,
    }

    agg = Aggregator(active_models)
    users: List[VirtualUser] = []
    for i in range(args.users):
        users.append(
            VirtualUser(
                user_id=i,
                api_url=args.api_url,
                api_key=args.api_key,
                model_weights=weights,
                active_models=active_models,
                max_tokens=args.max_tokens,
                think_ms_min=args.think_ms_min,
                think_ms_max=args.think_ms_max,
                think_dist=args.think_dist,
                tokens_jitter_pct=args.tokens_jitter_pct,
                user_weight_jitter=args.user_weight_jitter,
                max_backoff_sec=args.max_backoff_sec,
                qwen_image_src=args.qwen_image,
                whissent_audio_src=args.whissent_audio,
                aggregator=agg,
            )
        )

    tasks: List[asyncio.Task] = []
    interval = (args.ramp_up_sec / args.users) if args.users > 0 else 0
    for u in users:
        tasks.append(asyncio.create_task(u.run()))
        if interval > 0:
            await asyncio.sleep(interval)

    if args.hold_sec > 0:
        await asyncio.sleep(args.hold_sec)

    # Ramp-down
    down_interval = (args.ramp_down_sec / len(users)) if users else 0
    for u in users:
        u.stop()
        if down_interval > 0:
            await asyncio.sleep(down_interval)

    await asyncio.gather(*tasks, return_exceptions=True)
    report = agg.build_report(meta)
    report["meta"]["finished_at"] = now_ts()
    return report


def save_report(report: Dict[str, Any], path: Optional[str]) -> str:
    out_dir = Path("./logs"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(path) if path else out_dir / f"sim_report_{int(time.time())}.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save report: {e}")
    return str(out_path)


def print_summary(report: Dict[str, Any]):
    meta = report.get("meta", {})
    overall = report.get("overall", {})
    per_model = report.get("per_model", [])
    print("\n=== Simulation Summary ===")
    print(f"Users={meta.get('users')} | RampUp={meta.get('ramp_up_sec')}s | Hold={meta.get('hold_sec')}s | RampDown={meta.get('ramp_down_sec')}s")
    am = ", ".join(meta.get("active_models", []))
    print(f"Active models: {am}")
    print("Weights:")
    for k, v in meta.get("weights", {}).items():
        print(f"  - {k}: {v:.2f}")
    print(f"Overall: total={overall.get('total')} success={overall.get('success')} failure={overall.get('failure')} elapsed_s={overall.get('elapsed_s')}")
    for s in sorted(per_model, key=lambda x: x.get("model", "")):
        dist = " ".join(f"{wid}:{cnt}" for wid, cnt in s.get("worker_distribution", {}).items())
        print(
            f"\n[{s['model']}] Total={s['total']} OK={s['success']} Fail={s['failure']} Thr={s['throughput_rps']:.2f} req/s\n"
            f"  Lat(ms): avg={s['avg_ms']:.1f} p50={s['p50_ms']:.1f} p90={s['p90_ms']:.1f} p99={s['p99_ms']:.1f}\n"
            f"  Workers used: {s['workers_used']}\n  Dist: {dist}"
        )


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Multi-user simulation for ORCHESTRA LB")
    p.add_argument("--api-url", default=DEFAULT_INFER_URL)
    p.add_argument("--status-url", default=DEFAULT_STATUS_URL)
    p.add_argument("--api-key", default=None)
    p.add_argument("--users", type=int, default=20)
    p.add_argument("--ramp-up-sec", type=float, default=10.0)
    p.add_argument("--hold-sec", type=float, default=120.0)
    p.add_argument("--ramp-down-sec", type=float, default=10.0)
    p.add_argument("--think-ms-min", type=int, default=15000)
    p.add_argument("--think-ms-max", type=int, default=60000)
    p.add_argument("--think-dist", choices=["uniform", "exponential", "lognormal"], default="uniform", help="Distribution for think-time sampling")
    p.add_argument("--weights", default=None, help="e.g. 'Whissent rare and the rest equals (just base weights, then real one calculated using distribution)'")
    p.add_argument("--max-tokens", type=int, default=1000)
    p.add_argument("--tokens-jitter-pct", type=float, default=0.5, help="Per-request jitter fraction for max_new_tokens (0..1)")
    p.add_argument("--user-weight-jitter", type=float, default=0.1, help="Stddev for per-user Gaussian jitter of model weights (0..1)")
    p.add_argument("--max-backoff-sec", type=float, default=5.0, help="Max exponential backoff on errors")
    p.add_argument("--qwen-image", default=None)
    p.add_argument("--whissent-audio", default=None)
    p.add_argument("--report-path", default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(run_simulation(args))
    if not report or "error" in report:
        return 2
    out_path = save_report(report, args.report_path)
    print_summary(report)
    print(f"\nReport saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
