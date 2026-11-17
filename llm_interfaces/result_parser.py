"""从最终大模型响应中解析候选根因排序。"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Sequence, Tuple


def candidate_categories(candidates: Any) -> List[str]:
    categories: List[str] = []
    for item in candidates or []:
        if isinstance(item, dict):
            value = item.get("category", "")
        else:
            value = item
        category = str(value).strip() if value is not None else ""
        if category and category not in categories:
            categories.append(category)
    return categories


def _ranking_from_value(value: Any, candidate_set: set[str]) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []

    ranking: List[str] = []
    for item in value:
        if isinstance(item, str):
            category = item.strip()
        elif isinstance(item, dict):
            category = str(item.get("category", "")).strip()
        else:
            continue
        if category in candidate_set and category not in ranking:
            ranking.append(category)
    return ranking


def _structured_rankings(
    response_text: str,
    candidates: Sequence[str],
) -> List[List[str]]:
    candidate_set = set(candidates)
    normalized = (
        response_text
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    found: List[List[str]] = []
    decoder = json.JSONDecoder()

    for match in re.finditer(r"\[", normalized):
        try:
            value, _end = decoder.raw_decode(normalized[match.start():])
        except json.JSONDecodeError:
            continue
        ranking = _ranking_from_value(value, candidate_set)
        if ranking and ranking not in found:
            found.append(ranking)

    for block in re.findall(r"\[[^\[\]]*\]", normalized, flags=re.S):
        try:
            value = ast.literal_eval(block)
        except (ValueError, SyntaxError):
            continue
        ranking = _ranking_from_value(value, candidate_set)
        if ranking and ranking not in found:
            found.append(ranking)
    return found


def extract_ranked_categories(
    response_text: str,
    candidates: Sequence[str],
) -> Tuple[List[str], str]:
    rankings = _structured_rankings(response_text, candidates)
    if rankings:
        ranking = max(
            enumerate(rankings),
            key=lambda item: (len(item[1]), item[0]),
        )[1]
        return ranking, "structured_array"

    final_text = response_text.rsplit("</think>", 1)[-1][-6000:]
    positions = []
    for category in candidates:
        position = final_text.find(category)
        if position >= 0:
            positions.append((position, category))
    ranking = [category for _position, category in sorted(positions)]
    if ranking:
        return ranking, "candidate_text_fallback"
    return [], "unparsed"


def _rank(groundtruth: str, prediction: Sequence[str]):
    if groundtruth and groundtruth in prediction:
        return prediction.index(groundtruth) + 1
    return None


def prediction_fields(ranking: List[str], groundtruth: str, strategy: str) -> Dict[str, Any]:
    top_five = ranking[:5]
    return {
        "predicted_list": top_five,
        "rank": _rank(groundtruth, top_five),
        "predicted_list_v1": ranking,
        "rank_v1": _rank(groundtruth, ranking),
        "parse_strategy": strategy,
    }


def build_prediction_fields(
    response_text: str,
    root_cause_candidates: Any,
    groundtruth: str,
) -> Dict[str, Any]:
    candidates = candidate_categories(root_cause_candidates)
    full_prediction, strategy = extract_ranked_categories(
        response_text, candidates,
    )
    return prediction_fields(full_prediction, groundtruth, strategy)
