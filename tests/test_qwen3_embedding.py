#!/usr/bin/env python3
"""
Test script for Qwen3-Embedding (4B) model via Orchestra API
Author: Zied Mustapha

Usage:
    python tests/test_qwen3_embedding.py
    python tests/test_qwen3_embedding.py --host 127.0.0.1 --port 9001
"""

import argparse
import requests
import time
import json
import numpy as np
from typing import List, Optional


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_single_embedding(base_url: str, text: str, instruction: Optional[str] = None) -> dict:
    """Test single text embedding."""
    print(f"\n{'='*60}")
    print("TEST: Single Text Embedding")
    print(f"{'='*60}")
    print(f"Text: {text[:80]}..." if len(text) > 80 else f"Text: {text}")
    if instruction:
        print(f"Instruction: {instruction}")
    
    payload = {
        "model_name": "qwen3_embedding",
        "request_body": {
            "text": text,
        }
    }
    if instruction:
        payload["request_body"]["instruction"] = instruction
    
    start = time.time()
    response = requests.post(f"{base_url}/infer", json=payload)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        embedding = result.get("embedding", [])
        print(f"✓ Success! Embedding dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print(f"  Response time: {elapsed*1000:.1f}ms")
        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  Detail: {response.text}")
        return {}


def test_batch_embedding(base_url: str, texts: List[str], instruction: Optional[str] = None) -> dict:
    """Test batch text embedding."""
    print(f"\n{'='*60}")
    print("TEST: Batch Text Embedding")
    print(f"{'='*60}")
    print(f"Number of texts: {len(texts)}")
    for i, t in enumerate(texts[:3]):
        print(f"  [{i}] {t[:50]}..." if len(t) > 50 else f"  [{i}] {t}")
    if len(texts) > 3:
        print(f"  ... and {len(texts) - 3} more")
    if instruction:
        print(f"Instruction: {instruction}")
    
    payload = {
        "model_name": "qwen3_embedding",
        "request_body": {
            "texts": texts,
        }
    }
    if instruction:
        payload["request_body"]["instruction"] = instruction
    
    start = time.time()
    response = requests.post(f"{base_url}/infer", json=payload)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        embeddings = result.get("embeddings", [])
        print(f"✓ Success! Got {len(embeddings)} embeddings")
        print(f"  Embedding dimension: {result.get('embedding_dim', 'N/A')}")
        print(f"  Response time: {elapsed*1000:.1f}ms ({elapsed*1000/len(texts):.1f}ms/text)")
        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  Detail: {response.text}")
        return {}


def test_mrl_dimension(base_url: str, text: str, dim: int) -> dict:
    """Test MRL (Matryoshka Representation Learning) with custom dimension."""
    print(f"\n{'='*60}")
    print(f"TEST: MRL - Custom Embedding Dimension ({dim})")
    print(f"{'='*60}")
    
    payload = {
        "model_name": "qwen3_embedding",
        "request_body": {
            "text": text,
            "embedding_dim": dim,
        }
    }
    
    start = time.time()
    response = requests.post(f"{base_url}/infer", json=payload)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        embedding = result.get("embedding", [])
        print(f"✓ Success! Requested dim: {dim}, Got dim: {len(embedding)}")
        print(f"  Response time: {elapsed*1000:.1f}ms")
        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  Detail: {response.text}")
        return {}


def test_semantic_similarity(base_url: str):
    """Test semantic similarity between related and unrelated texts."""
    print(f"\n{'='*60}")
    print("TEST: Semantic Similarity")
    print(f"{'='*60}")
    
    # Define test texts
    query = "What is the capital of France?"
    doc_related = "Paris is the capital and largest city of France."
    doc_unrelated = "The Python programming language was created by Guido van Rossum."
    
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    
    print(f"Query: {query}")
    print(f"Doc A (related): {doc_related}")
    print(f"Doc B (unrelated): {doc_unrelated}")
    
    # Get embeddings
    payload = {
        "model_name": "qwen3_embedding",
        "request_body": {
            "texts": [query, doc_related, doc_unrelated],
            "instruction": instruction,
        }
    }
    
    start = time.time()
    response = requests.post(f"{base_url}/infer", json=payload)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        embeddings = result.get("embeddings", [])
        
        if len(embeddings) == 3:
            sim_related = cosine_similarity(embeddings[0], embeddings[1])
            sim_unrelated = cosine_similarity(embeddings[0], embeddings[2])
            
            print(f"\n✓ Results:")
            print(f"  Query ↔ Related doc:   {sim_related:.4f}")
            print(f"  Query ↔ Unrelated doc: {sim_unrelated:.4f}")
            print(f"  Difference: {sim_related - sim_unrelated:.4f}")
            
            if sim_related > sim_unrelated:
                print(f"  ✓ PASS: Related document has higher similarity!")
            else:
                print(f"  ✗ FAIL: Expected related doc to have higher similarity")
            
            print(f"  Response time: {elapsed*1000:.1f}ms")
        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  Detail: {response.text}")
        return {}


def test_multilingual(base_url: str):
    """Test multilingual embeddings."""
    print(f"\n{'='*60}")
    print("TEST: Multilingual Embeddings")
    print(f"{'='*60}")
    
    texts = [
        "Hello, how are you?",           # English
        "Bonjour, comment allez-vous?",  # French
        "Hola, ¿cómo estás?",            # Spanish
        "你好，你好吗？",                  # Chinese
        "مرحبا، كيف حالك؟",               # Arabic
    ]
    
    payload = {
        "model_name": "qwen3_embedding",
        "request_body": {
            "texts": texts,
        }
    }
    
    start = time.time()
    response = requests.post(f"{base_url}/infer", json=payload)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        embeddings = result.get("embeddings", [])
        
        print(f"\n✓ Got embeddings for {len(embeddings)} languages")
        print(f"\nCross-lingual similarity matrix (same greeting in different languages):")
        
        # Compute similarity matrix
        labels = ["EN", "FR", "ES", "ZH", "AR"]
        print(f"     {' '.join([f'{l:>6}' for l in labels])}")
        for i, label in enumerate(labels):
            sims = [cosine_similarity(embeddings[i], embeddings[j]) for j in range(len(embeddings))]
            print(f"  {label}: {' '.join([f'{s:>6.3f}' for s in sims])}")
        
        print(f"\n  Response time: {elapsed*1000:.1f}ms")
        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  Detail: {response.text}")
        return {}


def check_service_health(base_url: str) -> bool:
    """Check if the service is healthy."""
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            pools = status.get("model_pools", {})
            emb_workers = pools.get("qwen3_embedding", [])
            if emb_workers:
                print(f"✓ Service is running with {len(emb_workers)} qwen3_embedding worker(s)")
                return True
            else:
                print("✗ No qwen3_embedding workers found")
                return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to {base_url}")
        return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    return False


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-Embedding model")
    parser.add_argument("--host", default="127.0.0.1", help="API host")
    parser.add_argument("--port", default=9001, type=int, help="API port")
    parser.add_argument("--test", choices=["all", "single", "batch", "mrl", "semantic", "multilingual"],
                        default="all", help="Which test to run")
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print(f"\n{'#'*60}")
    print(f"  Qwen3-Embedding (4B) Test Suite")
    print(f"  API: {base_url}")
    print(f"{'#'*60}")
    
    # Health check
    print("\nChecking service health...")
    if not check_service_health(base_url):
        print("\nPlease ensure the service is running with qwen3_embedding model:")
        print("  ./scripts/run_api.sh 0 0 0 0 1 start")
        return 1
    
    # Run tests
    if args.test in ["all", "single"]:
        test_single_embedding(
            base_url, 
            "Machine learning is a subset of artificial intelligence.",
            instruction="Represent this sentence for retrieval"
        )
    
    if args.test in ["all", "batch"]:
        test_batch_embedding(
            base_url,
            [
                "The quick brown fox jumps over the lazy dog.",
                "A journey of a thousand miles begins with a single step.",
                "To be or not to be, that is the question.",
                "All that glitters is not gold.",
                "Knowledge is power.",
            ]
        )
    
    if args.test in ["all", "mrl"]:
        test_mrl_dimension(base_url, "Testing custom embedding dimensions", dim=512)
        test_mrl_dimension(base_url, "Testing custom embedding dimensions", dim=1024)
    
    if args.test in ["all", "semantic"]:
        test_semantic_similarity(base_url)
    
    if args.test in ["all", "multilingual"]:
        test_multilingual(base_url)
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())
