# -*- coding: utf-8 -*-

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def candidate_category(candidate: Any) -> str:
    if isinstance(candidate, dict):
        value = (
            candidate.get("category")
            or candidate.get("name")
            or candidate.get("root_cause")
        )
        return str(value).strip() if value is not None else ""
    return str(candidate).strip() if candidate is not None else ""


def normalize_candidates(candidates: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for candidate in candidates or []:
        category = candidate_category(candidate)
        if category and category not in result:
            result.append(category)
    return result


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    decoder = json.JSONDecoder()
    found: List[Dict[str, Any]] = []
    for position, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            found.append(value)
    if not found:
        return None
    for value in reversed(found):
        if any(key in value for key in (
            "ranked_candidates",
            "candidate_assessments",
            "anomaly_findings",
            "key_alerts",
        )):
            return value
    return found[-1]


def parse_document(text: str) -> Dict[str, Any]:
    parsed = extract_json_object(text)
    return parsed if parsed is not None else {
        "parse_strategy": "raw_text",
        "raw_text": text or "",
    }


def normalize_joint_scoring(
    document: Dict[str, Any],
    candidates: Sequence[str],
) -> Tuple[List[Dict[str, Any]], str]:
    items = []
    for key in ("ranked_candidates", "ranking", "results"):
        if isinstance(document.get(key), list):
            items = document[key]
            break
    has_model_ranking = bool(items)

    candidate_set = set(candidates)
    ranked: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if isinstance(item, dict):
            category = candidate_category(item)
            value = item.get("score", item.get("probability"))
            evidence = item.get("evidence", item.get("reason", []))
        else:
            category = candidate_category(item)
            value = None
            evidence = []
        if category not in candidate_set or category in seen:
            continue
        try:
            score = float(value) if value is not None else None
        except (TypeError, ValueError):
            score = None
        ranked.append({
            "category": category,
            "score": score,
            "evidence": evidence,
        })
        seen.add(category)

    for category in candidates:
        if category not in seen:
            ranked.append({
                "category": category,
                "score": None,
                "evidence": [],
            })

    if not ranked:
        return [], "unparsed"

    if all(item["score"] is None for item in ranked):
        for item in ranked:
            item["score"] = 1.0 / len(ranked)
        strategy = "ranking_only" if has_model_ranking else "candidate_order_fallback"
        return ranked, strategy

    for item in ranked:
        if item["score"] is None or item["score"] < 0:
            item["score"] = 0.0
    total = sum(item["score"] for item in ranked)
    if total <= 0:
        for item in ranked:
            item["score"] = 1.0 / len(ranked)
    else:
        for item in ranked:
            item["score"] /= total
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked, "scored_ranking"
