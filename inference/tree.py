# inference/tree.py
# -*- coding: utf-8 -*-

import argparse
import os
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

import config
from inference.common import (
    IndexedCaseReader,
    advance_checkpoint,
    build_output_dir,
    discover_label_indices,
    ensure_dir,
    get_incremental_indices,
    load_json,
    load_infer_state,
    save_infer_outputs,
    save_infer_state,
    save_json,
    scan_label_root,
    update_last_processed_index,
)
from inference.selection import (
    build_tree_summary_for_refiner,
    generate_selection_by_selector,
    load_selection,
    refine_selection_by_tree_summary,
)


def _load_tree_dependencies():
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier, export_text

    return np, LabelEncoder, DecisionTreeClassifier, export_text


# ============================================================
# 8. Tree 推理
# ============================================================


def extract_features_for_cases(
    cases: List[Dict[str, Any]],
    selection: Dict[str, Any],
):
    from rule_inferencer.data_process_v3 import extract_all_features

    X = []
    feature_names = None
    features_list = []

    for case_data in cases:
        if "semantic_labels" not in case_data:
            raise KeyError(
                "case 缺少 semantic_labels，"
                f"case_idx={case_data.get('case_idx')}"
            )

        names, values, features_dict = extract_all_features(
            case_data["semantic_labels"],
            selection,
        )

        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(
                "不同 case 抽取出的 feature_names 不一致，"
                f"case_idx={case_data.get('case_idx')}"
            )

        X.append(values)
        features_list.append(features_dict)

    return feature_names, X, features_list


def extract_labels(
    train_cases: List[Dict[str, Any]],
) -> Tuple[Any, Any]:
    np, LabelEncoder, _DecisionTreeClassifier, _export_text = (
        _load_tree_dependencies()
    )
    raw_labels = []

    for case in train_cases:
        root_cause = case.get("root_cause")

        if root_cause is None or root_cause == "":
            raise ValueError(
                "训练数据 root_cause 为空，"
                f"case_idx={case.get('case_idx')}"
            )

        raw_labels.append(root_cause)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_labels)
    return np.array(y), label_encoder


def train_tree_model(
    train_cases: List[Dict[str, Any]],
    selection: Dict[str, Any],
    max_depth: Optional[int] = None,
):
    np, _LabelEncoder, DecisionTreeClassifier, _export_text = (
        _load_tree_dependencies()
    )
    feature_names, X_train, _ = extract_features_for_cases(
        train_cases,
        selection,
    )
    y_train, label_encoder = extract_labels(train_cases)

    clf = DecisionTreeClassifier(
        max_depth=(
            max_depth
            if max_depth is not None
            else getattr(config, "MAX_DEPTH", 3)
        ),
        min_samples_leaf=getattr(config, "MIN_SAMPLES_LEAF", 10),
        random_state=getattr(config, "RANDOM_STATE", 42),
    )

    clf.fit(np.array(X_train), y_train)

    return clf, feature_names, label_encoder


def save_tree_rules(
    clf,
    feature_names,
    label_encoder,
    output_dir: str,
    scenario_name: str,
    tag: str,
) -> str:
    _np, _LabelEncoder, _DecisionTreeClassifier, export_text = (
        _load_tree_dependencies()
    )
    ensure_dir(output_dir)

    output_path = os.path.join(
        output_dir,
        f"{scenario_name}_{tag}.txt",
    )

    tree_rules = export_text(
        clf,
        feature_names=list(feature_names),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tree_rules)
        f.write("\n\nClass mapping:\n")

        for idx, cls in enumerate(label_encoder.classes_):
            f.write(f"class {idx} -> {cls}\n")

    return output_path


def build_tree_cot(
    clf,
    feature_names: List[str],
    sample_values,
    pred_id: Any,
) -> Dict[str, int]:
    """Return the exact root-to-leaf decision path used for one prediction."""
    tree = clf.tree_
    node_id = 0
    conditions: List[str] = []

    while tree.children_left[node_id] != tree.children_right[node_id]:
        feature_idx = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])
        value = float(sample_values[feature_idx])
        feature_name = str(feature_names[feature_idx])

        if value <= threshold:
            conditions.append(f"{feature_name} <= {threshold:.2f}")
            node_id = int(tree.children_left[node_id])
        else:
            conditions.append(f"{feature_name} > {threshold:.2f}")
            node_id = int(tree.children_right[node_id])

    path = " -> ".join(conditions) if conditions else "(root leaf)"
    return {path: int(pred_id)}


def predict_by_tree(
    infer_cases,
    infer_indices,
    clf,
    feature_names,
    label_encoder,
    selection,
):
    np, _LabelEncoder, _DecisionTreeClassifier, _export_text = (
        _load_tree_dependencies()
    )
    results = []

    for case, idx in zip(infer_cases, infer_indices):
        names, X_values, features_list = extract_features_for_cases(
            [case],
            selection,
        )

        if names != feature_names:
            raise ValueError(
                f"case_idx={idx} 的特征名和训练集不一致"
            )

        X = np.array(X_values)
        pred_id = clf.predict(X)[0]
        pred_top1 = label_encoder.inverse_transform([pred_id])[0]
        cot = build_tree_cot(
            clf=clf,
            feature_names=list(feature_names),
            sample_values=X[0],
            pred_id=pred_id,
        )

        pred_rc = [pred_top1]
        pred_scores = []

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            pairs = sorted(
                zip(clf.classes_, proba),
                key=lambda item: item[1],
                reverse=True,
            )

            class_ids = [int(item[0]) for item in pairs]
            scores = [float(item[1]) for item in pairs]
            pred_rc = label_encoder.inverse_transform(class_ids).tolist()
            pred_scores = [
                {"root_cause": rc, "score": score}
                for rc, score in zip(pred_rc, scores)
            ]

        groundtruth = case.get("root_cause")
        rank = None
        is_correct = None

        if groundtruth is not None and groundtruth != "":
            rank = (
                pred_rc.index(groundtruth) + 1
                if groundtruth in pred_rc
                else None
            )
            is_correct = pred_top1 == groundtruth

        results.append({
            "case_idx": idx,
            "alarm_type": case.get("alarm_type"),
            "alarm_time": case.get("alarm_time"),
            "case_file_path": case.get("_case_file_path"),
            "label_file_path": case.get("_label_file_path"),
            "groundtruth": groundtruth,
            "pred_top1_rc": pred_top1,
            "pred_rc": pred_rc,
            "pred_scores": pred_scores,
            "cot": cot,
            "rank": rank,
            "is_correct": is_correct,
            "features": features_list[0] if features_list else None,
        })

    return results


def summarize_tree_accuracy(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    labeled = [
        item for item in results
        if item.get("is_correct") is not None
    ]
    correct = sum(
        1 for item in labeled
        if item.get("is_correct") is True
    )
    total = len(labeled)
    return {
        "labeled_total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else None,
    }


class _TreeValCaseDataLoader:
    """Adapter for the val feature extractor used by the current Tree reader."""

    def __init__(self, cases: List[Dict[str, Any]], indices: List[int]):
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


def _load_tree_val_rule_config(
    scenario: Dict[str, Any],
    rule_root: str,
) -> Tuple[str, Dict[str, Any], str]:
    alarm_type = scenario["alarm_type"]
    candidates = [
        os.path.join(rule_root, f"{alarm_type}.json"),
    ]
    if scenario.get("name") != alarm_type:
        candidates.append(
            os.path.join(rule_root, f"{scenario['name']}.json")
        )

    for path in candidates:
        if os.path.exists(path):
            return alarm_type, load_json(path), path

    raise FileNotFoundError(
        f"未找到场景 {alarm_type} 的 Tree val 规则文件；"
        f"已检查: {candidates}"
    )


def _summarize_tree_val_results(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
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


def _rerank_tree_results_with_val(
    scenario: Dict[str, Any],
    infer_cases: List[Dict[str, Any]],
    infer_indices: List[int],
    results: List[Dict[str, Any]],
    rule_root: Optional[str] = None,
    template_dir: Optional[str] = None,
    kpi_mode: Optional[str] = None,
    neutral_score: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from rule_inferencer.data_process_val import (
        rerank_root_cause_results,
    )

    rule_root = rule_root or getattr(
        config,
        "TREE_VAL_RULE_ROOT",
        "/home/sbp/huangzeshun/RCA_Units/rt2feature",
    )
    template_dir = template_dir or getattr(
        config,
        "TREE_VAL_TEMPLATE_DIR",
        getattr(
            config,
            "TEMPLATE_DIR",
            "/home/sbp/huangzeshun/RCA_Units/template",
        ),
    )
    kpi_mode = kpi_mode or getattr(config, "TREE_VAL_KPI_MODE", "mix")
    neutral_score = (
        neutral_score
        if neutral_score is not None
        else getattr(config, "TREE_VAL_NEUTRAL_SCORE", 0.5)
    )
    target_size = getattr(config, "TREE_VAL_TARGET_SIZE", 5)
    random_seed = getattr(config, "TREE_VAL_RANDOM_SEED", 42)
    alarm_type, rule_config, rule_path = _load_tree_val_rule_config(
        scenario,
        rule_root,
    )

    for result in results:
        # val uses `rc`; retain the current `groundtruth` field as-is too.
        result["rc"] = result.get("groundtruth")

    ret = rerank_root_cause_results(
        ret={"results": results},
        dataloader=_TreeValCaseDataLoader(infer_cases, infer_indices),
        rule_configs={alarm_type: rule_config},
        kpi_mode=kpi_mode,
        neutral_score=neutral_score,
        template_dir=template_dir,
        target_size=target_size,
        seed=random_seed,
    )
    reranked_results = ret.get("results", [])
    for result in reranked_results:
        groundtruth = result.get("rc")
        pred_rc_val = result.get("pred_rc_val", [])
        result["rank_val"] = (
            pred_rc_val.index(groundtruth) + 1
            if groundtruth in pred_rc_val
            else None
        )
    return reranked_results, {
        "enabled": True,
        "rule_path": rule_path,
        "template_dir": template_dir,
        "kpi_mode": kpi_mode,
        "neutral_score": neutral_score,
        "target_size": target_size,
        "random_seed": random_seed,
        "indices": list(infer_indices),
        "summary": _summarize_tree_val_results(reranked_results),
    }


def run_tree_once(
    scenario: Dict[str, Any],
    train_cases: List[Dict[str, Any]],
    infer_cases: List[Dict[str, Any]],
    train_indices: List[int],
    infer_indices: List[int],
    selection: Dict[str, Any],
    output_dir: str,
    output_format: str,
    tag: str,
    enable_val: bool = False,
    val_rule_root: Optional[str] = None,
    val_template_dir: Optional[str] = None,
    val_kpi_mode: Optional[str] = None,
    val_neutral_score: Optional[float] = None,
) -> Dict[str, Any]:
    scenario_name = scenario["name"]
    alarm_type = scenario["alarm_type"]

    clf, feature_names, label_encoder = train_tree_model(
        train_cases,
        selection,
    )

    tree_rule_path = save_tree_rules(
        clf,
        feature_names,
        label_encoder,
        os.path.join(output_dir, "tree_rules"),
        scenario_name,
        tag,
    )

    results = predict_by_tree(
        infer_cases,
        infer_indices,
        clf,
        feature_names,
        label_encoder,
        selection,
    )

    validation = {
        "enabled": False,
        "type": "root_cause_rerank",
        "indices": [],
    }
    if enable_val:
        results, validation = _rerank_tree_results_with_val(
            scenario=scenario,
            infer_cases=infer_cases,
            infer_indices=infer_indices,
            results=results,
            rule_root=val_rule_root,
            template_dir=val_template_dir,
            kpi_mode=val_kpi_mode,
            neutral_score=val_neutral_score,
        )

    accuracy_summary = summarize_tree_accuracy(results)
    correct = accuracy_summary["correct"]
    total = accuracy_summary["labeled_total"]
    accuracy = accuracy_summary["accuracy"]

    payload = {
        "meta": {
            "mode": "tree_infer",
            "tag": tag,
            "scenario_name": scenario_name,
            "alarm_type": alarm_type,
            "data_dir": scenario.get("data_dir"),
            "train_indices": train_indices,
            "infer_indices": infer_indices,
            "tree_rule_path": tree_rule_path,
            "max_depth": getattr(config, "MAX_DEPTH", 3),
            "tree_val_enabled": enable_val,
            "min_samples_leaf": getattr(
                config,
                "MIN_SAMPLES_LEAF",
                10,
            ),
            "random_state": getattr(config, "RANDOM_STATE", 42),
            "output_dir": output_dir,
        },
        "selection": selection,
        "validation": validation,
        "summary": {
            "labeled_total": total,
            "correct": correct,
            "accuracy": accuracy,
            "processed_count": len(results),
            "processed_indices": [
                item["case_idx"] for item in results
            ],
        },
        "results": results,
    }

    save_infer_outputs(payload, output_dir, output_format)
    return payload


def tree_infer_incremental_one_scenario(
    scenario: Dict[str, Any],
    infer_indices: List[int],
    train_n: int,
    selection_path: Optional[str],
    selection_source: str,
    refiner_rounds: int,
    output_dir: Optional[str],
    output_format: str,
    enable_val: bool = False,
    val_rule_root: Optional[str] = None,
    val_template_dir: Optional[str] = None,
    val_kpi_mode: Optional[str] = None,
    val_neutral_score: Optional[float] = None,
) -> Dict[str, Any]:
    scenario_name = scenario["name"]
    alarm_type = scenario["alarm_type"]
    data_dir = scenario["data_dir"]

    if not infer_indices:
        raise ValueError(f"[{scenario_name}] infer_indices 为空")

    all_indices = discover_label_indices(data_dir, alarm_type)
    first_infer_idx = min(infer_indices)

    train_candidates = [
        idx for idx in all_indices
        if idx < first_infer_idx
    ]
    train_indices = train_candidates[-train_n:]

    if len(train_indices) < train_n:
        raise ValueError(
            f"[{scenario_name}] 训练数据不足："
            f"需要 {train_n} 条，当前只有 {len(train_indices)} 条；"
            f"first_infer_idx={first_infer_idx}"
        )

    run_name = (
        f"train_{min(train_indices)}_{max(train_indices)}_"
        f"infer_{min(infer_indices)}_{max(infer_indices)}"
    )

    print("=" * 100)
    print(f"[tree_infer][incremental] 场景: {scenario_name}")
    print(f"data_dir         : {data_dir}")
    print(f"train_indices    : {train_indices}")
    print(f"infer_indices    : {infer_indices}")
    print(f"selection_source : {selection_source}")
    print(f"tree_val_enabled : {enable_val}")
    print("=" * 100)

    reader = IndexedCaseReader(
        data_dir=data_dir,
        alarm_type=alarm_type,
    )

    train_cases = reader.load_cases(train_indices)
    infer_cases = reader.load_cases(infer_indices)

    selection_work_dir = build_output_dir(
        "selection_pipeline",
        scenario_name,
        output_dir,
        run_name,
    )

    final_output_dir = build_output_dir(
        "tree_infer",
        scenario_name,
        output_dir,
        run_name,
    )

    if selection_source == "file":
        selection = load_selection(scenario, selection_path)

    elif selection_source == "selector":
        selection = generate_selection_by_selector(
            scenario,
            selection_work_dir,
        )

    elif selection_source == "selector_refiner":
        selection = generate_selection_by_selector(
            scenario,
            selection_work_dir,
        )

        for round_id in range(refiner_rounds):
            round_dir = os.path.join(
                selection_work_dir,
                f"refiner_round_{round_id}",
                "tree",
            )
            ensure_dir(round_dir)

            tmp_payload = run_tree_once(
                scenario=scenario,
                train_cases=train_cases,
                infer_cases=infer_cases,
                train_indices=train_indices,
                infer_indices=infer_indices,
                selection=selection,
                output_dir=round_dir,
                output_format="json",
                tag=f"refiner_round_{round_id}",
                # Refiner summaries should describe the raw Tree output;
                # apply root-cause val only to the final Tree result.
                enable_val=False,
            )

            tree_summary = build_tree_summary_for_refiner(tmp_payload)

            save_json(
                tree_summary,
                os.path.join(
                    selection_work_dir,
                    f"tree_summary_round_{round_id}.json",
                ),
            )

            selection = refine_selection_by_tree_summary(
                scenario=scenario,
                previous_selection=selection,
                tree_summary=tree_summary,
                output_dir=selection_work_dir,
                round_id=round_id,
            )

    else:
        raise ValueError(
            f"未知 selection_source: {selection_source}"
        )

    save_json(
        selection,
        os.path.join(final_output_dir, "selection_final.json"),
    )

    payload = run_tree_once(
        scenario=scenario,
        train_cases=train_cases,
        infer_cases=infer_cases,
        train_indices=train_indices,
        infer_indices=infer_indices,
        selection=selection,
        output_dir=final_output_dir,
        output_format=output_format,
        tag="final",
        enable_val=enable_val,
        val_rule_root=val_rule_root,
        val_template_dir=val_template_dir,
        val_kpi_mode=val_kpi_mode,
        val_neutral_score=val_neutral_score,
    )

    accuracy = payload.get("summary", {}).get("accuracy")
    print(f"[Done][tree_infer] {scenario_name}")
    print(f"output_dir: {final_output_dir}")

    if accuracy is not None:
        print(f"accuracy  : {accuracy:.4f}")

    return payload


def tree_infer(args: argparse.Namespace) -> List[Dict[str, Any]]:
    scenarios = scan_label_root(
        args.anomalydetect_label_root,
        args.scenario,
    )

    train_n = (
        args.train_n
        if args.train_n is not None
        else getattr(config, "TRAIN_N", 50)
    )

    state = load_infer_state(args.state_file)
    payloads = []

    for scenario in scenarios:
        indices, should_update_state, previous_last_index = (
            get_incremental_indices(
                scenario=scenario,
                mode="tree_infer",
                args=args,
                state=state,
            )
        )

        if not indices:
            print(
                f"[tree_infer] {scenario['name']} 没有新增数据，跳过"
            )
            continue

        try:
            payload = tree_infer_incremental_one_scenario(
                scenario=scenario,
                infer_indices=indices,
                train_n=train_n,
                selection_path=args.selection_path,
                selection_source=args.selection_source,
                refiner_rounds=args.refiner_rounds,
                output_dir=args.output_dir,
                output_format=args.output_format,
                enable_val=getattr(args, "tree_val", False),
                val_rule_root=getattr(args, "tree_val_rule_root", None),
                val_template_dir=getattr(args, "tree_val_template_dir", None),
                val_kpi_mode=getattr(args, "tree_val_kpi_mode", None),
                val_neutral_score=getattr(args, "tree_val_neutral_score", None),
            )
            payloads.append(payload)

            success_indices = set(
                payload.get("summary", {}).get(
                    "processed_indices",
                    [],
                )
            )

            if should_update_state:
                new_checkpoint = advance_checkpoint(
                    selected_indices=indices,
                    success_indices=success_indices,
                    previous_last_index=previous_last_index,
                )

                if new_checkpoint > previous_last_index:
                    update_last_processed_index(
                        state=state,
                        mode="tree_infer",
                        scenario=scenario,
                        last_index=new_checkpoint,
                    )
                    save_infer_state(state, args.state_file)

        except Exception as exc:
            print(
                f"[Error][tree_infer] {scenario['name']} 处理失败: {exc}"
            )
            traceback.print_exc()

            if args.strict:
                raise

    return payloads
