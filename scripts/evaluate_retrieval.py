import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from src.rag.retriever import Retriever
from src.utils.helpers import normalize_url


def _load_qa_pairs(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("QA file must contain a list of items")
    return data


def _is_relevant(result, expected_urls: set[str], expected_keywords: List[str]) -> bool:
    if expected_urls:
        return normalize_url(result.url) in expected_urls
    if expected_keywords:
        text = (result.text or "").lower()
        return any(keyword in text for keyword in expected_keywords)
    return False


def _evaluate_question(
    retriever: Retriever,
    item: Dict[str, Any],
    top_k: int,
    threshold: float | None,
    enforce_threshold: bool,
) -> Dict[str, Any]:
    question = item.get("question", "").strip()
    if not question:
        raise ValueError("Each QA item must include a question")

    expected_urls = [normalize_url(url) for url in item.get("expected_urls", []) if url]
    expected_keywords = [kw.lower() for kw in item.get("expected_keywords", []) if kw]
    if not expected_urls and not expected_keywords:
        expected_note = "no expectations provided"
    elif expected_urls:
        expected_note = "url-based"
    else:
        expected_note = "keyword-based"

    retrieved = retriever.retrieve(
        query=question,
        top_k=top_k,
        similarity_threshold=threshold,
        enforce_threshold=enforce_threshold,
    )

    relevant = [r for r in retrieved if _is_relevant(r, set(expected_urls), expected_keywords)]

    total_retrieved = len(retrieved)
    total_relevant = len(expected_urls) if expected_urls else (1 if expected_keywords else 0)

    precision = (len(relevant) / total_retrieved) if total_retrieved else 0.0
    recall = (len(relevant) / total_relevant) if total_relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    mrr = 0.0
    for idx, result in enumerate(retrieved, start=1):
        if _is_relevant(result, set(expected_urls), expected_keywords):
            mrr = 1.0 / idx
            break

    return {
        "id": item.get("id"),
        "question": question,
        "expected_urls": expected_urls,
        "expected_keywords": expected_keywords,
        "expected_note": expected_note,
        "retrieved_count": total_retrieved,
        "relevant_count": len(relevant),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mrr": round(mrr, 4),
        "top_urls": [r.url for r in retrieved[:3]],
    }


def _summary(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "macro_mrr": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
        }

    precisions = [m["precision"] for m in metrics]
    recalls = [m["recall"] for m in metrics]
    f1s = [m["f1"] for m in metrics]
    mrrs = [m["mrr"] for m in metrics]

    total_retrieved = sum(m["retrieved_count"] for m in metrics)
    total_relevant = sum(
        len(m["expected_urls"]) if m["expected_urls"] else (1 if m["expected_keywords"] else 0)
        for m in metrics
    )
    total_relevant_retrieved = sum(m["relevant_count"] for m in metrics)

    micro_precision = (total_relevant_retrieved / total_retrieved) if total_retrieved else 0.0
    micro_recall = (total_relevant_retrieved / total_relevant) if total_relevant else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) else 0.0

    return {
        "macro_precision": round(statistics.mean(precisions), 4),
        "macro_recall": round(statistics.mean(recalls), 4),
        "macro_f1": round(statistics.mean(f1s), 4),
        "macro_mrr": round(statistics.mean(mrrs), 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "total_retrieved": total_retrieved,
        "total_relevant": total_relevant,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval precision/recall against QA pairs.")
    parser.add_argument(
        "--qa-file",
        default="data/eval/qa_pairs.json",
        help="Path to QA pairs JSON file.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k for retrieval.")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Override similarity threshold.",
    )
    parser.add_argument(
        "--no-threshold",
        action="store_true",
        help="Disable threshold enforcement when evaluating.",
    )
    parser.add_argument(
        "--output",
        default="data/eval/retrieval_report.json",
        help="Path to write evaluation results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qa_path = Path(args.qa_file)
    if not qa_path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_path}")

    qa_pairs = _load_qa_pairs(qa_path)
    retriever = Retriever()

    started = time.perf_counter()
    metrics = [
        _evaluate_question(
            retriever,
            item,
            top_k=args.top_k,
            threshold=args.similarity_threshold,
            enforce_threshold=not args.no_threshold,
        )
        for item in qa_pairs
    ]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    report = {
        "run_ms": elapsed_ms,
        "qa_file": str(qa_path),
        "top_k": args.top_k,
        "similarity_threshold": args.similarity_threshold,
        "enforce_threshold": not args.no_threshold,
        "summary": _summary(metrics),
        "items": metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    print(f"\nReport written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
