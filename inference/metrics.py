"""共享的根因排序评测指标。"""

from typing import Any, Dict, Optional, Sequence


def calculate_metrics(
    results: Sequence[Dict[str, Any]],
    elapsed_seconds: float,
    rank_key: str = "rank",
) -> Dict[str, Any]:
    labeled = [item for item in results if item.get("groundtruth")]
    ranks = [item.get(rank_key) for item in labeled]
    total = len(labeled)

    def top_k(k: int) -> Optional[float]:
        if not total:
            return None
        return sum(rank is not None and rank <= k for rank in ranks) / total

    return {
        "evaluated_count": total,
        "top1": top_k(1),
        "top3": top_k(3),
        "top5": top_k(5),
        "mrr": (
            sum(1.0 / rank if rank else 0.0 for rank in ranks) / total
            if total else None
        ),
        "elapsed_seconds": elapsed_seconds,
        "average_seconds_per_case": elapsed_seconds / total if total else None,
        "unparsed_count": sum(
            item.get("parse_strategy") == "unparsed" for item in labeled
        ),
    }
