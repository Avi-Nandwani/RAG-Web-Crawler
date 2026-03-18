import argparse
import json
import sys
from typing import Any, Dict

import requests


def _print_step(title: str, payload: Dict[str, Any], response: requests.Response):
    print(f"\n=== {title} ===")
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except Exception:
        print(response.text)


def run_demo(base_url: str, start_url: str, question: str, max_pages: int, max_depth: int) -> int:
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    try:
        health = session.get(f"{base_url}/health", timeout=20)
        _print_step("Health", {}, health)
        health.raise_for_status()

        crawl_payload = {
            "start_url": start_url,
            "max_pages": max_pages,
            "max_depth": max_depth,
            "crawl_delay_ms": 300,
        }
        crawl = session.post(f"{base_url}/crawl", json=crawl_payload, timeout=120)
        _print_step("Crawl", crawl_payload, crawl)
        crawl.raise_for_status()

        index_payload = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "min_chunk_size": 100,
        }
        index = session.post(f"{base_url}/index", json=index_payload, timeout=180)
        _print_step("Index", index_payload, index)
        index.raise_for_status()

        ask_payload = {
            "question": question,
            "top_k": 5,
            "similarity_threshold": 0.3,
        }
        ask = session.post(f"{base_url}/ask", json=ask_payload, timeout=180)
        _print_step("Ask", ask_payload, ask)
        ask.raise_for_status()

        print("\nDemo pipeline completed successfully.")
        return 0
    except requests.RequestException as exc:
        print(f"\nDemo failed: {exc}")
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 8 demo pipeline against the local API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of the running API server.")
    parser.add_argument("--start-url", default="https://example.com", help="Starting URL for crawl.")
    parser.add_argument(
        "--question",
        default="What is the main purpose of this website?",
        help="Question to ask after indexing.",
    )
    parser.add_argument("--max-pages", type=int, default=10, help="Maximum pages to crawl.")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum crawl depth.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_demo(
            base_url=args.base_url.rstrip("/"),
            start_url=args.start_url,
            question=args.question,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
        )
    )
