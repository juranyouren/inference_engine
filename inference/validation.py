# -*- coding: utf-8 -*-
"""Shared root-cause validation/reranking for already produced results.

The Tree model and the LLM model both produce a ranked ``pred_rc`` list.  This
module is deliberately model-agnostic: it adapts loaded cases to the val
feature extractor, maps the common truth field to ``rc``, and appends the
reranked fields while preserving the original prediction fields.
"""

import copy
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import config
from inference.common import load_json


class CaseRangeAdapter:
    """Minimal adapter expected by ``data_process_val``.

    The val implementation requests a half-open index range.  Keeping the
    adapter here allows Tree and LLM callers to share exactly the same
    feature-extraction path without coupling the scorer to either pipeline.
    """

    def __init__(
        self,
        cases: Sequence[Dict[str, Any]],
        indices: Sequence[int],
    ) -> None:
        if len(cases) != len(indices):
            raise ValueError("cases and indices must have the same length")
        self._cases_by_index = dict(zip(indices, cases))

    def get_cases_by_id_range(
        self,
        start_id: int,
        end_id: int,
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        selected_indices = sorted(
            idx for idx in self._cases_by_index
            if start_id <= idx < end_id
        )
        return (
            [self._cases_by_index[idx] for idx in selected_indices],
            selected_indices,
        )


def load_rule_config(
    scenario: Dict[str, Any],
    rule_root: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], str]:
    """Load the rule JSON for a scenario and return its alarm type/path."""
    root = rule_root or getattr(
        config,
        "TREE_VAL_RULE_ROOT",
        "/home/sbp/huangzeshun/RCA_Units/rt2feature",
    )
    alarm_type = str(scenario.get("alarm_type") or scenario.get("name"))
    candidates = [os.path.join(root, f"{alarm_type}.json")]
    name = scenario.get("name")
    if name and name != alarm_type:
        candidates.append(os.path.join(root, f"{name}.json"))

    for path in candidates:
        if os.path.exists(path):
            return alarm_type, load_json(path), path

    raise FileNotFoundError(
        f"未找到场景 {alarm_type} 的 val 规则文件；已检查: {candidates}"
    )


def summarize_validation_results(
    results: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    results = list(results)
    labeled = [
        item for item in results
        if item.get("is_correct_val") is not None
    ]
    correct = sum(
        1 for item in labeled
        if item.get("is_correct_val") is True
    )
    return {
        "labeled_total": len(labeled),
        "correct": correct,
        "accuracy": correct / len(labeled) if labeled else None,
        "rerank_skipped": sum(
            1 for item in results
            if item.get("rerank_skipped") is True
        ),
        "rerank_changed": sum(
            1 for item in results
            if item.get("rerank_changed") is True
        ),
        "pred_top1_changed": sum(
            1 for item in results
            if item.get("pred_top1_changed") is True
        ),
    }


def rerank_results_with_val(
    scenario: Dict[str, Any],
    cases: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    results: Sequence[Dict[str, Any]],
    rule_root: Optional[str] = None,
    template_dir: Optional[str] = None,
    kpi_mode: Optional[str] = None,
    neutral_score: Optional[float] = None,
    target_size: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rerank one scenario's results and preserve the raw prediction fields.

    ``data_process_val.rerank_root_cause_results`` mutates its input and fills
    short candidate lists.  It is therefore called on a deep copy; only the
    ``*_val`` and rerank-detail fields are merged back into the original result
    objects.  This makes raw-vs-val metrics directly comparable.
    """
    from rule_inferencer.data_process_val import rerank_root_cause_results

    if len(cases) != len(indices):
        raise ValueError("cases and indices must have the same length")

    rule_alarm_type, rule_config, rule_path = load_rule_config(
        scenario,
        rule_root=rule_root,
    )
    effective_template_dir = template_dir or getattr(
        config,
        "TREE_VAL_TEMPLATE_DIR",
        getattr(config, "TEMPLATE_DIR", "/home/sbp/huangzeshun/RCA_Units/template"),
    )
    effective_kpi_mode = kpi_mode or getattr(config, "TREE_VAL_KPI_MODE", "mix")
    effective_neutral_score = (
        neutral_score
        if neutral_score is not None
        else getattr(config, "TREE_VAL_NEUTRAL_SCORE", 0.5)
    )
    effective_target_size = (
        target_size
        if target_size is not None
        else getattr(config, "TREE_VAL_TARGET_SIZE", 5)
    )
    effective_seed = (
        random_seed
        if random_seed is not None
        else getattr(config, "TREE_VAL_RANDOM_SEED", 42)
    )

    original_results = [dict(item) for item in results]
    work_results = copy.deepcopy(original_results)
    for item in work_results:
        # New and old result producers use groundtruth; the scorer uses rc.
        item["rc"] = item.get("rc", item.get("groundtruth"))
        item.setdefault("alarm_type", rule_alarm_type)

    ret = rerank_root_cause_results(
        ret={"results": work_results},
        dataloader=CaseRangeAdapter(cases, indices),
        rule_configs={rule_alarm_type: rule_config},
        kpi_mode=effective_kpi_mode,
        neutral_score=effective_neutral_score,
        template_dir=effective_template_dir,
        target_size=effective_target_size,
        seed=effective_seed,
    )

    val_fields = (
        "pred_rc_val",
        "pred_top1_rc_val",
        "pred_rc_val_detail",
        "pred_rc_val_check",
        "rerank_skipped",
        "rerank_skip_reason",
        "rerank_changed",
        "pred_top1_changed",
        "is_correct_val",
    )
    reranked = ret.get("results", [])
    if len(reranked) != len(original_results):
        raise ValueError("val reranker changed the result count")

    merged: List[Dict[str, Any]] = []
    for original, reranked_item in zip(original_results, reranked):
        item = dict(original)
        for field in val_fields:
            if field in reranked_item:
                item[field] = reranked_item[field]
        groundtruth = item.get("groundtruth", item.get("rc"))
        pred_rc_val = item.get("pred_rc_val", [])
        item["rank_val"] = (
            pred_rc_val.index(groundtruth) + 1
            if groundtruth in pred_rc_val
            else None
        )
        # Keep rc as an explicit mapping field for downstream consumers.
        item["rc"] = groundtruth
        merged.append(item)

    metadata = {
        "enabled": True,
        "type": "root_cause_rerank",
        "rule_path": rule_path,
        "template_dir": effective_template_dir,
        "kpi_mode": effective_kpi_mode,
        "neutral_score": effective_neutral_score,
        "target_size": effective_target_size,
        "random_seed": effective_seed,
        "indices": list(indices),
        "summary": summarize_validation_results(merged),
    }
    return merged, metadata
