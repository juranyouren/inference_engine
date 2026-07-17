# -*- coding: utf-8 -*-
"""Reparse saved Competition/Cooperation responses and rescore without inference."""

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import config
from inference.common import IndexedCaseReader, load_json, save_infer_outputs
from run_three_method_experiment import evaluate_saved_results


METHODS = ("competition", "cooperation")


def _deduplicate(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        value = str(value).strip()
        if value and value not in result:
            result.append(value)
    return result


def _candidate_categories(label_data: Dict[str, Any]) -> List[str]:
    values = []
    for item in label_data.get("root_cause_candidates", []) or []:
        value = item.get("category") if isinstance(item, dict) else item
        if value is not None:
            values.append(str(value))
    return _deduplicate(values)


def _groundtruth(label_data: Dict[str, Any]) -> str:
    root_cause = label_data.get("root_cause", {})
    if isinstance(root_cause, dict):
        value = root_cause.get("category", "")
    else:
        value = root_cause
    return str(value).strip() if value is not None else ""


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
    """Extract every JSON/Python-style list containing candidate categories."""
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
    """Last-resort ranking from candidate order in the final-answer tail."""
    final_text = response_text.rsplit("</think>", 1)[-1][-6000:]
    positions = []
    for category in candidates:
        position = final_text.find(category)
        if position >= 0:
            positions.append((position, category))
    return [category for _position, category in sorted(positions)]


def _read_response(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _best_ranking(rankings: List[List[str]]) -> List[str]:
    if not rankings:
        return []
    # Prefer the most complete answer; for equal lengths, prefer the last one.
    return max(enumerate(rankings), key=lambda item: (len(item[1]), item[0]))[1]


def _consensus_ranking(
    rankings: List[List[str]],
    candidates: Sequence[str],
) -> List[str]:
    if not rankings:
        return []
    missing_rank = len(candidates) + 1
    scores = {}
    for category in candidates:
        ranks = [
            ranking.index(category) + 1 if category in ranking else missing_rank
            for ranking in rankings
        ]
        scores[category] = sum(ranks) / len(ranks)
    return sorted(candidates, key=lambda category: (scores[category], candidates.index(category)))


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
        if source == "competition_reasoners":
            return _consensus_ranking(rankings, candidates), record
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
    verifier = output_dir / f"{alarm_type}_analysis_Reasoner_{case_idx}" / "raw_responses.txt"
    reasoners = output_dir / f"{alarm_type}_analysis_Meta_{case_idx}" / "raw_responses.txt"
    attempts = []

    for path, source in (
        (verifier, "competition_verifier"),
        (reasoners, "competition_reasoners"),
    ):
        ranking, record = _attempt(path, source, None, candidates, allow_text=False)
        attempts.append(record)
        if ranking:
            return ranking, f"{source}:structured_array", attempts

    for path, source in (
        (verifier, "competition_verifier"),
        (reasoners, "competition_reasoners"),
    ):
        ranking, record = _attempt(path, source, None, candidates, allow_text=True)
        attempts.append(record)
        if ranking:
            return ranking, f"{source}:candidate_text_fallback", attempts
    return [], "unparsed", attempts


def parse_cooperation_case(
    output_dir: Path,
    case_idx: int,
    candidates: Sequence[str],
) -> Tuple[List[str], str, List[Dict[str, Any]]]:
    case_dir = output_dir / str(case_idx)
    rounds = sorted(
        [path for path in case_dir.glob("round*") if _round_number(path) >= 0],
        key=_round_number,
        reverse=True,
    )
    attempts = []

    # Start at the final round. Reasoner is preferred only when it contains a
    # real candidate ranking; Meta normally carries the cooperation ranking.
    for round_dir in rounds:
        round_id = _round_number(round_dir)
        sources = (
            (round_dir / "reasoner" / "raw_responses.txt", "cooperation_reasoner"),
            (round_dir / "meta" / "raw_responses.txt", "cooperation_meta"),
        )
        for path, source in sources:
            ranking, record = _attempt(path, source, round_id, candidates, allow_text=False)
            attempts.append(record)
            if ranking:
                return ranking, f"{source}:round{round_id}:structured_array", attempts
        for path, source in sources:
            ranking, record = _attempt(path, source, round_id, candidates, allow_text=True)
            attempts.append(record)
            if ranking:
                return ranking, f"{source}:round{round_id}:candidate_text_fallback", attempts
    return [], "unparsed", attempts


def _run_output_dirs(payload: Dict[str, Any]) -> Dict[str, Path]:
    return {
        str(run.get("scenario_name")): Path(run["output_dir"])
        for run in payload.get("meta", {}).get("runs", [])
        if run.get("scenario_name") and run.get("output_dir")
    }


def _load_label(
    item: Dict[str, Any],
    label_root: str,
) -> Tuple[Dict[str, Any], str]:
    label_path = item.get("label_file_path")
    if label_path and os.path.isfile(label_path):
        return load_json(label_path), label_path
    scenario = item.get("scenario_name") or item.get("alarm_type")
    reader = IndexedCaseReader(
        os.path.join(label_root, scenario),
        scenario,
    )
    _case_path, resolved_label = reader.find_case_and_label_path(int(item["case_idx"]))
    return load_json(resolved_label), resolved_label


def reparse_method(
    experiment_dir: str,
    output_root: str,
    method: str,
    label_root: str,
) -> Dict[str, Any]:
    source_file = os.path.join(experiment_dir, method, "predictions.json")
    payload = load_json(source_file)
    output_dirs = _run_output_dirs(payload)
    reparsed = []

    for item in payload.get("results", []):
        scenario = item.get("scenario_name") or item.get("alarm_type")
        case_idx = int(item["case_idx"])
        label_data, label_path = _load_label(item, label_root)
        candidates = _candidate_categories(label_data)
        raw_output_dir = output_dirs.get(str(scenario))
        if raw_output_dir is None:
            ranking, strategy, attempts = [], "missing_scenario_output_dir", []
        elif method == "competition":
            ranking, strategy, attempts = parse_competition_case(
                raw_output_dir,
                str(item.get("alarm_type") or scenario),
                case_idx,
                candidates,
            )
        else:
            ranking, strategy, attempts = parse_cooperation_case(
                raw_output_dir,
                case_idx,
                candidates,
            )

        truth = _groundtruth(label_data)
        rank = ranking.index(truth) + 1 if truth in ranking else None
        updated = dict(item)
        updated.update({
            "label_file_path": label_path,
            "groundtruth": truth,
            "pred_top1_rc": ranking[0] if ranking else None,
            "pred_rc": ranking,
            "rank": rank,
            "is_correct": rank == 1,
            "parse_strategy": strategy,
            "parse_attempts": attempts,
        })
        reparsed.append(updated)

    combined = {
        "meta": {
            **payload.get("meta", {}),
            "reparsed": True,
            "source_predictions": source_file,
        },
        "summary": {
            "processed_count": len(reparsed),
            "parsed_count": sum(bool(item.get("pred_rc")) for item in reparsed),
            "unparsed_count": sum(not item.get("pred_rc") for item in reparsed),
        },
        "results": reparsed,
    }
    save_infer_outputs(combined, os.path.join(output_root, method), "all")
    return combined


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="重新解析已有 LLM 原始回答并计算 Top-1/3/5、MRR。",
    )
    parser.add_argument(
        "--experiment-dir",
        default=os.path.join(
            getattr(config, "PREDICT_RES_DIR"),
            "experiments",
            "three_methods",
        ),
    )
    parser.add_argument(
        "--label-root",
        default=getattr(config, "ANOMALYDETECT_LABEL_ROOT"),
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_root = args.output_dir or os.path.join(args.experiment_dir, "reparsed")
    for method in args.methods:
        result = reparse_method(
            args.experiment_dir,
            output_root,
            method,
            args.label_root,
        )
        print(f"[{method}] {json.dumps(result['summary'], ensure_ascii=False)}")
    report = evaluate_saved_results(output_root, args.methods)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"重新解析结果: {output_root}")


if __name__ == "__main__":
    main()
