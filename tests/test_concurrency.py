#!/usr/bin/env python3
"""
Comprehensive concurrency and load test for the Multi-Model Load Balancer.

Features
- Auto-discovers worker counts per model from /status
- Runs concurrent requests per model (gemma3, qwen, qwen3)
- Measures success rate, throughput, and latency percentiles (p50/p90/p99)
- Checks load distribution across workers (via load_balancer_worker_id)
- Supports Qwen multimodal with a built-in tiny PNG when no image is provided

Usage examples
  python3 tests/test_concurrency.py
  python3 tests/test_concurrency.py --api-url http://127.0.0.1:9001/infer --api-key <KEY>
  python3 tests/test_concurrency.py --per-model-requests 10 --concurrency-per-model 4
  python3 tests/test_concurrency.py --models gemma3,qwen3
  # Run multiple models at the same time (fire all loads concurrently)
  python3 tests/test_concurrency.py --models whissent,qwen,qwen3 --per-model-requests 50 --concurrency-per-model 50 --models-concurrently

Exit code
- 0 on success when not asserting
- Non-zero if --assert and criteria are not met
"""

import argparse
import asyncio
import base64
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

try:
    import aiohttp
except Exception as e:
    print("This test requires aiohttp. Please install it (e.g., pip install aiohttp).", file=sys.stderr)
    raise

DEFAULT_INFER_URL = "http://127.0.0.1:9001/infer"
DEFAULT_STATUS_URL = "http://127.0.0.1:9001/status"
DEFAULT_IMAGE = "tests/image.png"

# A tiny 1x1 PNG (transparent) as base64, used as fallback for Qwen multimodal tests.
# Source: data URI of a 1x1 PNG.
FALLBACK_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/az2n0sAAAAASUVORK5CYII="
)

@dataclass
class SampleResult:
    ok: bool
    status: int
    duration: float
    worker_id: Optional[int]
    model_processed: Optional[str]
    error: Optional[str] = None
    # Optional captured response content
    response_json: Optional[Dict[str, Any]] = None
    response_text: Optional[str] = None

@dataclass
class ModelRunSummary:
    model_name: str
    total: int
    success: int
    failure: int
    throughput_rps: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    avg_ms: float
    workers_used: List[int]
    worker_request_counts: Dict[int, int]


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    k = (len(values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def encode_image_b64(image_path: Optional[str]) -> str:
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return FALLBACK_PNG_BASE64


def make_payload(model_name: str, text: str, max_new_tokens: int, session_prefix: str,
                 image_b64: Optional[str] = None,
                 whissent_audio_path: Optional[str] = None) -> Dict[str, Any]:
    rb: Dict[str, Any] = {
        "input": text,
        "max_new_tokens": max_new_tokens,
        "session_id": f"{session_prefix}-{int(time.time()*1000)}"
    }

    if model_name == "qwen":
        # For Qwen, we require an image for multimodal; use fallback PNG if not provided
        image_b64 = image_b64 or FALLBACK_PNG_BASE64
        rb["images"] = [{"type": "image", "data": image_b64, "format": "png"}]

    if model_name == "whissent":
        # Determine audio path: CLI argument or default testing path
        candidate = whissent_audio_path
        if not candidate or not os.path.exists(candidate):
            candidate = "tests/audio.mp3"
            if not os.path.exists(candidate):
                # Try relative fallback when running from repo root
                if os.path.exists("tests/audio.mp3"):
                    candidate = "tests/audio.mp3"
                else:
                    raise ValueError("WhisSent test requires an audio file. Provide --audio-file or ensure tests/audio.mp3 exists.")
        # Build WhisSent-specific request body
        rb = {
            "audio": {"path": candidate},
            "return_timestamps": True,
            "task": "transcribe",
            "language": "fr",
            "emotion_top_k": 3,
            "session_id": f"{session_prefix}-{int(time.time()*1000)}"
        }

    # Gemma3 can be text-only here to keep the test lighter
    return {"model_name": model_name, "request_body": rb}


async def post_json(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], api_key: Optional[str], timeout: int = 300) -> Tuple[int, Dict[str, Any] | str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout), headers=headers) as resp:
            status = resp.status
            try:
                data = await resp.json()
            except Exception:
                data = await resp.text()
            return status, data
    except Exception as e:
        return 0, str(e)


async def run_one_infer(session: aiohttp.ClientSession, api_url: str, payload: Dict[str, Any], api_key: Optional[str]) -> SampleResult:
    t0 = time.perf_counter()
    status, data = await post_json(session, api_url, payload, api_key)
    dt = (time.perf_counter() - t0) * 1000.0  # ms

    if isinstance(data, dict):
        if 200 <= status < 300:
            return SampleResult(
                ok=True,
                status=status,
                duration=dt,
                worker_id=data.get("load_balancer_worker_id"),
                model_processed=data.get("model_name_processed"),
                error=None,
                response_json=data,
            )
        else:
            return SampleResult(
                ok=False,
                status=status,
                duration=dt,
                worker_id=data.get("load_balancer_worker_id"),
                model_processed=data.get("model_name_processed"),
                error=json.dumps(data),
                response_json=data,
            )
    else:
        # Non-JSON error text
        return SampleResult(
            ok=False,
            status=status,
            duration=dt,
            worker_id=None,
            model_processed=None,
            error=str(data)[:500],
            response_text=str(data)[:10000],
        )


async def fetch_status(status_url: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(status_url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            return await resp.json()


async def run_model_load(
    model_name: str,
    api_url: str,
    api_key: Optional[str],
    total_requests: int,
    concurrency: int,
    max_new_tokens: int,
    qwen_image_path: Optional[str],
    session_prefix: str,
    whissent_audio_path: Optional[str] = None,
) -> Tuple[ModelRunSummary, List[SampleResult]]:
    sem = asyncio.Semaphore(max(1, concurrency))
    results: List[SampleResult] = []

    image_b64 = encode_image_b64(qwen_image_path) if model_name == "qwen" else None

    async with aiohttp.ClientSession() as session:
        async def worker_task(idx: int):
            async with sem:
                text = f"Load test request {idx} for {model_name}. Please reply with an acknowledgment of 100 words."
                payload = make_payload(
                    model_name,
                    text,
                    max_new_tokens,
                    session_prefix,
                    image_b64=image_b64,
                    whissent_audio_path=whissent_audio_path,
                )
                res = await run_one_infer(session, api_url, payload, api_key)
                results.append(res)

        t_start = time.perf_counter()
        await asyncio.gather(*(worker_task(i) for i in range(total_requests)))
        t_elapsed = time.perf_counter() - t_start

    durations = sorted([r.duration for r in results if r.ok])
    success = sum(1 for r in results if r.ok)
    failure = len(results) - success
    throughput = success / t_elapsed if t_elapsed > 0 else 0.0
    workers_used = sorted(list({r.worker_id for r in results if r.worker_id is not None}))
    worker_counts_counter = Counter([r.worker_id for r in results if r.worker_id is not None])
    worker_request_counts: Dict[int, int] = {int(k): int(v) for k, v in worker_counts_counter.items()}

    summary = ModelRunSummary(
        model_name=model_name,
        total=len(results),
        success=success,
        failure=failure,
        throughput_rps=throughput,
        p50_ms=percentile(durations, 50.0) if durations else 0.0,
        p90_ms=percentile(durations, 90.0) if durations else 0.0,
        p99_ms=percentile(durations, 99.0) if durations else 0.0,
        avg_ms=(sum(durations) / len(durations)) if durations else 0.0,
        workers_used=workers_used,
        worker_request_counts=worker_request_counts,
    )
    return summary, results


async def run_plan_concurrently(
    plan: List[Tuple[str, int, int, int]],
    api_url: str,
    api_key: Optional[str],
    max_new_tokens: int,
    qwen_image_path: Optional[str],
    audio_file: Optional[str],
) -> List[Tuple[str, int, ModelRunSummary, List[SampleResult]]]:
    """Fire all per-model loads concurrently and return results in plan order.

    Returns a list of tuples (model_name, expected_workers, summary, results).
    """
    tasks = []
    for model_name, workers, total, conc in plan:
        tasks.append(
            run_model_load(
                model_name=model_name,
                api_url=api_url,
                api_key=api_key,
                total_requests=total,
                concurrency=conc,
                max_new_tokens=max_new_tokens,
                qwen_image_path=qwen_image_path,
                session_prefix="concurrency",
                whissent_audio_path=audio_file,
            )
        )

    outputs = await asyncio.gather(*tasks)
    # outputs is List[(summary, results)]; zip with plan to include model/workers
    merged: List[Tuple[str, int, ModelRunSummary, List[SampleResult]]] = []
    for (model_name, workers, _t, _c), (summary, results) in zip(plan, outputs):
        merged.append((model_name, workers, summary, results))
    return merged


def print_summary(summary: ModelRunSummary, expected_workers: int | None):
    print(f"\n=== Model: {summary.model_name} ===")
    print(f"Total: {summary.total} | Success: {summary.success} | Failure: {summary.failure}")
    print(f"Throughput: {summary.throughput_rps:.2f} req/s")
    print(f"Latency (ms): avg={summary.avg_ms:.1f} p50={summary.p50_ms:.1f} p90={summary.p90_ms:.1f} p99={summary.p99_ms:.1f}")
    if expected_workers is not None:
        print(f"Workers used: {summary.workers_used} (expected up to {expected_workers})")
    else:
        print(f"Workers used: {summary.workers_used}")
    if summary.worker_request_counts:
        dist = " ".join(f"{wid}:{count}" for wid, count in sorted(summary.worker_request_counts.items()))
        print(f"Worker distribution: {dist}")


def derive_plan_from_status(status: Dict[str, Any], models_filter: Optional[List[str]], per_model_requests: Optional[int], concurrency_per_model: Optional[int]) -> List[Tuple[str, int, int, int]]:
    """
    Returns a plan: list of tuples (model_name, workers_count, total_requests, concurrency)
    - If per_model_requests/concurrency_per_model provided, use them directly.
    - Otherwise derive defaults: total_requests = max(4, 2 * workers), concurrency = max(2, workers)
    """
    model_pools = status.get("model_pools", {})
    plan: List[Tuple[str, int, int, int]] = []

    for model_name, pool in model_pools.items():
        if models_filter and model_name not in models_filter:
            continue
        workers = len(pool)
        if workers <= 0:
            continue
        total = per_model_requests if per_model_requests is not None else max(4, 2 * workers)
        conc = concurrency_per_model if concurrency_per_model is not None else max(2, workers)
        plan.append((model_name, workers, total, conc))
    return plan


def main():
    parser = argparse.ArgumentParser(description="Concurrency and load test for Multi-Model LB")
    parser.add_argument("--api-url", default=DEFAULT_INFER_URL, help="LB infer URL")
    parser.add_argument("--status-url", default=DEFAULT_STATUS_URL, help="LB status URL")
    parser.add_argument("--api-key", default=None, help="Optional API key")
    parser.add_argument("--qwen-image", default=DEFAULT_IMAGE, help="Image path for Qwen (optional)")
    parser.add_argument("--audio-file", default="tests/audio.mp3", help="Audio file path for WhisSent (default: tests/audio.mp3)")
    parser.add_argument("--max-tokens", type=int, default=64, help="max_new_tokens to request")
    parser.add_argument("--per-model-requests", type=int, help="Total requests per model")
    parser.add_argument("--concurrency-per-model", type=int, help="Concurrency per model")
    parser.add_argument("--models", default=None, help="Comma-separated subset of models to test (e.g., gemma3,qwen3)")
    parser.add_argument("--assert", dest="do_assert", action="store_true", help="Fail if success<100% or poor distribution")
    parser.add_argument("--min-worker-usage", type=int, default=1, help="Minimum distinct workers expected per model when asserting")
    parser.add_argument("--print-responses", action="store_true", help="Print the JSON/text response of each request")
    parser.add_argument("--models-concurrently", action="store_true", help="Run all selected models at the same time (concurrently)")
    args = parser.parse_args()

    models_filter = [m.strip() for m in args.models.split(",")] if args.models else None

    print(f"Fetching status from: {args.status_url}")
    status = asyncio.run(fetch_status(args.status_url))

    # If no models filter provided, exclude 'whissent' by default to avoid requiring audio setup
    effective_models_filter = models_filter
    if effective_models_filter is None:
        model_pools = status.get("model_pools", {})
        effective_models_filter = [m for m in model_pools.keys() if m != "whissent"]

    plan = derive_plan_from_status(
        status,
        models_filter=effective_models_filter,
        per_model_requests=args.per_model_requests,
        concurrency_per_model=args.concurrency_per_model,
    )

    if not plan:
        print("No workers discovered for requested models. Exiting.")
        return 1

    print("Planned runs:")
    for model_name, workers, total, conc in plan:
        print(f"- {model_name}: workers={workers}, total_requests={total}, concurrency={conc}")

    summaries: List[ModelRunSummary] = []
    any_failures = False

    if args.models_concurrently and len(plan) > 1:
        print("\nStarting concurrent loads for all selected models...")
        merged = asyncio.run(
            run_plan_concurrently(
                plan,
                api_url=args.api_url,
                api_key=args.api_key,
                max_new_tokens=args.max_tokens,
                qwen_image_path=args.qwen_image,
                audio_file=args.audio_file,
            )
        )
        for (model_name, workers, summary, results) in merged:
            summaries.append(summary)
            print_summary(summary, expected_workers=workers)
            if args.print_responses:
                for idx, r in enumerate(results, 1):
                    print(f"\nResponse {idx}/{len(results)} [model={model_name}, status={r.status}, worker={r.worker_id}]:")
                    if r.response_json is not None:
                        if model_name == "whissent":
                            subset = {k: r.response_json.get(k) for k in ["transcript", "emotion", "emotion_scores", "chunks", "gpu_id_of_model_worker"]}
                            print(json.dumps(subset, indent=2, ensure_ascii=False))
                        else:
                            print(json.dumps(r.response_json, indent=2, ensure_ascii=False))
                    elif r.response_text:
                        print(r.response_text)
                    elif r.error:
                        print(r.error)
            if summary.failure > 0:
                any_failures = True
            if args.do_assert and len(summary.workers_used) < max(args.min_worker_usage, 1):
                print(f"Assertion: insufficient worker distribution: used={summary.workers_used}")
                any_failures = True
    else:
        for model_name, workers, total, conc in plan:
            print(f"\nStarting load: model={model_name}, total={total}, concurrency={conc}")
            summary, results = asyncio.run(
                run_model_load(
                    model_name=model_name,
                    api_url=args.api_url,
                    api_key=args.api_key,
                    total_requests=total,
                    concurrency=conc,
                    max_new_tokens=args.max_tokens,
                    qwen_image_path=args.qwen_image,
                    session_prefix="concurrency",
                    whissent_audio_path=args.audio_file,
                )
            )
            summaries.append(summary)
            print_summary(summary, expected_workers=workers)
            if args.print_responses:
                for idx, r in enumerate(results, 1):
                    print(f"\nResponse {idx}/{len(results)} [model={model_name}, status={r.status}, worker={r.worker_id}]:")
                    if r.response_json is not None:
                        # For WhisSent, print the most relevant fields first
                        if model_name == "whissent":
                            subset = {k: r.response_json.get(k) for k in ["transcript", "emotion", "emotion_scores", "chunks", "gpu_id_of_model_worker"]}
                            print(json.dumps(subset, indent=2, ensure_ascii=False))
                        else:
                            print(json.dumps(r.response_json, indent=2, ensure_ascii=False))
                    elif r.response_text:
                        print(r.response_text)
                    elif r.error:
                        print(r.error)
            if summary.failure > 0:
                any_failures = True
            if args.do_assert and len(summary.workers_used) < max(args.min_worker_usage, 1):
                print(f"Assertion: insufficient worker distribution: used={summary.workers_used}")
                any_failures = True

    print("\n=== Overall Summary ===")
    for s in summaries:
        print(f"{s.model_name}: OK={s.success}/{s.total}, thr={s.throughput_rps:.2f} req/s, p50={s.p50_ms:.0f}ms, p90={s.p90_ms:.0f}ms, p99={s.p99_ms:.0f}ms, workers={s.workers_used}")

    if args.do_assert and any_failures:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
