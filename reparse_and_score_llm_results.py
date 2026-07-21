# -*- coding: utf-8 -*-
"""Reparse saved Competition/Cooperation responses and rescore without inference."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import config
from inference.common import IndexedCaseReader, load_json, save_infer_outputs
from inference.result_parsing import (
    extract_structured_rankings,
    parse_competition_case,
    parse_cooperation_case,
)
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
