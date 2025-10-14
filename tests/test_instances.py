#!/usr/bin/env python3
"""
Small smoke test for each model instance via the load balancer.
- gemma3: text-only
- qwen: text + image
- qwen3: text-only
Author: Zied Mustapha
Usage examples:
  python3 test_instances.py
  python3 test_instances.py --api-url http://127.0.0.1:9001/infer --image /path/to/image.png
  python3 test_instances.py --api-key YOUR_KEY
"""

import argparse
import base64
import json
import os
import sys
import time
import uuid
from urllib import request, error

DEFAULT_API_URL = "http://127.0.0.1:9001/infer"
DEFAULT_IMAGE = "tests/image.png"


def post_json(url: str, payload: dict, api_key: str | None = None, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = request.Request(url=url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {body}") from None
    except Exception as e:
        raise RuntimeError(str(e)) from None


def make_payload(model_name: str, text: str, max_new_tokens: int, session_prefix: str,
                 image_path: str | None = None) -> dict:
    rb = {
        "input": text,
        "max_new_tokens": max_new_tokens,
        "session_id": f"{session_prefix}-{uuid.uuid4().hex[:8]}"
    }

    # Per user requirements: only attach image for qwen
    if model_name == "qwen" and image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        # assume PNG by default
        rb["images"] = [{"type": "image", "data": b64, "format": "png"}]

    return {"model_name": model_name, "request_body": rb}


def run_one(api_url: str, model_name: str, text: str, max_new_tokens: int, api_key: str | None,
            image_path: str | None = None) -> dict:
    payload = make_payload(model_name, text, max_new_tokens, session_prefix="smoke", image_path=image_path)
    t0 = time.time()
    result = post_json(api_url, payload, api_key=api_key)
    dt = time.time() - t0
    # Enrich with timing if not present
    result.setdefault("duration_seconds_client", round(dt, 2))
    # Compute an estimated tokens/s from the response text length
    resp_text = result.get("response", "")
    if isinstance(resp_text, str):
        approx_tokens = len(resp_text.split())
    else:
        approx_tokens = 0
    duration = result.get("duration_seconds") or result.get("duration_seconds_client") or dt
    try:
        tokens_per_sec = (approx_tokens / duration) if duration > 0 else 0.0
    except Exception:
        tokens_per_sec = 0.0
    # Attach estimates
    result.setdefault("tokens_generated_estimate", approx_tokens)
    result.setdefault("tokens_per_second_estimate", round(tokens_per_sec, 2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Smoke test each model via the load balancer")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Inference endpoint URL (LB)")
    parser.add_argument("--api-key", help="Optional API key for the load balancer")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to image for Qwen test")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per request")
    parser.add_argument("--print-full", action="store_true", help="Print full model response text")
    args = parser.parse_args()

    print(f"Using API URL: {args.api_url}")

    textTemp = """
        This paragraph is a dialogue between two people.
        Format it correctly as a conversation between person 1 and person 2.
        "
        Allô? Oui, bonjour. Je voulais juste m'assurer que vous existiez vraiment. Mais apparemment, oui, vous existez. Et qui êtes-vous? Je suis... C'est vous qui vous occupez des assurances Arnavi, c'est ça? Alors, oui, je suis Madame Desjardins. Je... Je... Je peux savoir qui vous êtes? Je suis la maman d'une jeune fille que vous avez arnaquée. Donc du coup, je voulais juste vous dire que dès demain, tout sera bloqué, tout sera mis en place pour bloquer tous les prélèvements. Et voilà. Je voulais juste vous prévenir à l'avance. Parce que moi, je suis honnête. Il n'y a aucune arnaque, Madame. Si, si, si, on a regardé sur le site. Il n'y a que des avis négatifs. Arnaque, arnaque, arnaque, arnaque. Donc, demain, je m'occupe de tout. C'est une jeune infirmière et du coup, vous l'avez bien eue, mais ce n'est pas grave. Je vous dis bonsoir, Madame. Je voulais juste vous tenir au courant. Au revoir. Madame.
        "
    """

    cases = [
        ("gemma3", "Say 'hello' and include the name of your model.", None),
        ("qwen", "Describe this image briefly.", args.image),
        ("qwen3",textTemp, None),
    ]

    all_ok = True

    for model_name, text, img in cases:
        print(f"\n=== Testing {model_name} ===")
        try:
            res = run_one(
                api_url=args.api_url,
                model_name=model_name,
                text=text,
                max_new_tokens=args.max_tokens,
                api_key=args.api_key,
                image_path=img,
            )
            # Friendly summary
            lb_id = res.get("load_balancer_worker_id")
            worker_gpu = res.get("gpu_id_of_model_worker")
            model_processed = res.get("model_name_processed")
            dur = res.get("duration_seconds") or res.get("duration_seconds_client")
            tps = res.get("tokens_per_second_estimate")
            preview = res.get("response", "")
            if not args.print_full and isinstance(preview, str):
                preview = (preview[:500] + "...") if len(preview) > 500 else preview

            print(f"Status: OK | LB Worker: {lb_id} | GPU: {worker_gpu} | Model: {model_processed} | Duration: {dur}s | Speed: {tps} tok/s")
            print(f"Response: {preview}")
        except Exception as e:
            all_ok = False
            print(f"Status: ERROR | {e}")

    print("\nSmoke test complete.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
