# -*- coding: utf-8 -*-
"""Shared parsing policy for saved Competition/Cooperation responses."""

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _ranking_from_value(value: Any, candidate_set: set) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []

    ranking = []
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


def extract_structured_rankings(
    response_text: str,
    candidates: Sequence[str],
) -> List[List[str]]:
    """Extract all JSON/Python-style arrays containing candidate categories."""
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

    # Models sometimes use single quotes, which are valid Python but not JSON.
    for block in re.findall(r"\[[^\[\]]*\]", normalized, flags=re.S):
        try:
            value = ast.literal_eval(block)
        except (ValueError, SyntaxError):
            continue
        ranking = _ranking_from_value(value, candidate_set)
        if ranking and ranking not in found:
            found.append(ranking)
    return found


def extract_text_ranking(
    response_text: str,
    candidates: Sequence[str],
) -> List[str]:
    """Build a last-resort ranking from candidate order in the final answer."""
    final_text = response_text.rsplit("</think>", 1)[-1][-6000:]
    positions = []
    for category in candidates:
        position = final_text.find(category)
        if position >= 0:
            positions.append((position, category))
    return [category for _position, category in sorted(positions)]


def _best_ranking(rankings: List[List[str]]) -> List[str]:
    if not rankings:
        return []
    # Prefer the most complete answer; for equal lengths, prefer the last one.
    return max(enumerate(rankings), key=lambda item: (len(item[1]), item[0]))[1]


def extract_ranked_categories(
    response_text: str,
    candidates: Sequence[str],
    allow_text_fallback: bool = True,
) -> Tuple[List[str], str]:
    """Apply the repository-wide candidate-ranking parsing policy."""
    rankings = extract_structured_rankings(response_text, candidates)
    if rankings:
        return _best_ranking(rankings), "structured_array"

    if allow_text_fallback:
        ranking = extract_text_ranking(response_text, candidates)
        if ranking:
            return ranking, "candidate_text_fallback"
    return [], "unparsed"


def _read_response(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _attempt(
    path: Path,
    source: str,
    round_id: Optional[int],
    candidates: Sequence[str],
    allow_text: bool,
) -> Tuple[List[str], Dict[str, Any]]:
    text = _read_response(path)
    record = {
        "source": source,
        "round": round_id,
        "path": str(path),
        "exists": text is not None,
        "strategy": None,
        "ranking_count": 0,
    }
    if text is None:
        return [], record

    rankings = extract_structured_rankings(text, candidates)
    record["ranking_count"] = len(rankings)
    if rankings:
        record["strategy"] = "structured_array"
        return _best_ranking(rankings), record

    if allow_text:
        ranking = extract_text_ranking(text, candidates)
        if ranking:
            record["strategy"] = "candidate_text_fallback"
            return ranking, record
    return [], record


def _round_number(path: Path) -> int:
    match = re.fullmatch(r"round(\d+)", path.name)
    return int(match.group(1)) if match else -1


def parse_competition_case(
    output_dir: Path,
    alarm_type: str,
    case_idx: int,
    candidates: Sequence[str],
) -> Tuple[List[str], str, List[Dict[str, Any]]]:
    """Parse only the final Competition Verifier output."""
    # The historical directory name says Reasoner, but this file is written by
    # generate_rca_analysis_verifier_batch.
    verifier_path = (
        output_dir
        / f"{alarm_type}_analysis_Reasoner_{case_idx}"
        / "raw_responses.txt"
    )
    attempts = []

    ranking, record = _attempt(
        verifier_path,
        "competition_reasoner",
        None,
        candidates,
        allow_text=False,
    )
    attempts.append(record)
    if ranking:
        return ranking, "competition_reasoner:structured_array", attempts

    ranking, record = _attempt(
        verifier_path,
        "competition_reasoner",
        None,
        candidates,
        allow_text=True,
    )
    attempts.append(record)
    if ranking:
        return ranking, "competition_reasoner:candidate_text_fallback", attempts
    return [], "unparsed", attempts


def parse_cooperation_case(
    output_dir: Path,
    case_idx: int,
    candidates: Sequence[str],
) -> Tuple[List[str], str, List[Dict[str, Any]]]:
    """Parse only the Meta output from the last Cooperation round."""
    case_dir = output_dir / str(case_idx)
    rounds = sorted(
        [path for path in case_dir.glob("round*") if _round_number(path) >= 0],
        key=_round_number,
        reverse=True,
    )
    attempts = []
    if not rounds:
        return [], "unparsed", attempts

    last_round = rounds[0]
    round_id = _round_number(last_round)
    meta_path = last_round / "meta" / "raw_responses.txt"

    ranking, record = _attempt(
        meta_path,
        "cooperation_meta",
        round_id,
        candidates,
        allow_text=False,
    )
    attempts.append(record)
    if ranking:
        return ranking, f"cooperation_meta:round{round_id}:structured_array", attempts

    ranking, record = _attempt(
        meta_path,
        "cooperation_meta",
        round_id,
        candidates,
        allow_text=True,
    )
    attempts.append(record)
    if ranking:
        return (
            ranking,
            f"cooperation_meta:round{round_id}:candidate_text_fallback",
            attempts,
        )
    return [], "unparsed", attempts
