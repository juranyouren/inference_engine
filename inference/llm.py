# inference/llm.py
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
    ensure_dir,
    get_incremental_indices,
    iter_batches,
    load_infer_state,
    load_json,
    save_infer_outputs,
    save_infer_state,
    scan_label_root,
    update_last_processed_index,
)


# ============================================================
# 9. LLM 推理：完整 Competition / Cooperation
# ============================================================


def _lazy_import_llm_modules():
    from llm_inference.generator import RCAGenerator
    from utils.public_functions import (
        load_alarm_data_meta,
        load_alarm_data_reasoner_cooperation,
        load_alarm_data_verifier,
    )

    return (
        RCAGenerator,
        load_alarm_data_meta,
        load_alarm_data_reasoner_cooperation,
        load_alarm_data_verifier,
    )


def llm_find_case_and_label(
    data_dir: str,
    alarm_type: str,
    idx: int,
) -> Optional[Tuple[str, str]]:
    reader = IndexedCaseReader(data_dir, alarm_type)

    try:
        return reader.find_case_and_label_path(idx)
    except FileNotFoundError:
        return None


def get_sop_value_from_label(label_file_path: str) -> Any:
    """读取 SOP：优先 semantic_labels["sop"]，为空时 fallback 到顶层 label["sop"]。"""
    label_data = load_json(label_file_path)
    semantic_labels = label_data.get("semantic_labels", {})

    if not isinstance(semantic_labels, dict):
        return label_data.get("sop", "")

    sop = semantic_labels.get("sop")
    if sop is not None and (not isinstance(sop, str) or sop.strip()):
        return sop
    return label_data.get("sop", "")


def is_non_empty_sop(sop: Any) -> bool:
    if sop is None:
        return False
    if isinstance(sop, str):
        return bool(sop.strip())
    if isinstance(sop, (list, dict, tuple, set)):
        return len(sop) > 0
    return bool(str(sop).strip())


def require_sop(sop: Any, label_file: str) -> str:
    """确保 SOP 非空；若缺失则抛出 ValueError 并指明文件路径。"""
    import json as _json

    sop_str = ""
    if isinstance(sop, str):
        sop_str = sop.strip()
    elif isinstance(sop, (dict, list)):
        sop_str = _json.dumps(sop, ensure_ascii=False) if sop else ""
    elif sop is not None:
        sop_str = str(sop).strip()

    if not sop_str:
        raise ValueError(
            f"SOP 为空：{label_file}，请检查 label 数据中是否包含 sop 字段"
        )
    return sop_str


def split_indices_by_sop(
    data_dir: str,
    alarm_type: str,
    indices: List[int],
) -> Tuple[List[int], List[int]]:
    """
    分流规则：
        semantic_labels["sop"] 非空 -> competition
        semantic_labels["sop"] 为空 -> cooperation
    """
    cooperation_indices: List[int] = []
    competition_indices: List[int] = []

    for idx in indices:
        found = llm_find_case_and_label(data_dir, alarm_type, idx)
        if found is None:
            continue

        _case_file, label_file = found
        sop = get_sop_value_from_label(label_file)

        if is_non_empty_sop(sop):
            competition_indices.append(idx)
        else:
            cooperation_indices.append(idx)

    return cooperation_indices, competition_indices


def build_competition_alarm_data(
    label_file: str,
    alarm_type: str,
) -> Dict[str, Any]:
    """直接从 label 构建 Competition 输入，确保 SOP 来源正确。"""
    label_data = load_json(label_file)
    semantic_labels = label_data.get("semantic_labels", {})
    if not isinstance(semantic_labels, dict):
        semantic_labels = {}

    sop = semantic_labels.get("sop", "")
    if isinstance(sop, str) and not sop.strip():
        sop = label_data.get("sop", "")
    elif sop is None:
        sop = label_data.get("sop", "")

    require_sop(sop, label_file)

    return {
        "alarm_type": alarm_type,
        "semantic_labels": semantic_labels,
        "alarm_time": label_data.get(
            "alarm_time",
            semantic_labels.get("alarm_time"),
        ),
        "sop": sop,
        "root_cause_candidates": label_data.get("root_cause_candidates", []),
    }


def llm_competition_one_scenario(
    scenario: Dict[str, Any],
    indices: List[int],
    rca_generator,
    load_alarm_data_verifier,
    output_dir: Optional[str],
    output_format: str,
    batch_size: int,
) -> Dict[str, Any]:
    """
    Competition 完整流程：
        多采样 Competition -> Verifier
    """
    scenario_name = scenario["name"]
    alarm_type = scenario["alarm_type"]
    data_dir = scenario["data_dir"]

    run_name = f"idx_{min(indices)}_{max(indices)}" if indices else "empty"
    final_output_dir = build_output_dir(
        "llm_infer_competition",
        scenario_name,
        output_dir,
        run_name,
    )

    processed: List[int] = []
    skipped: List[int] = []

    for batch_indices in iter_batches(indices, batch_size):
        alarm_data_list = []
        label_files_list = []
        valid_indices = []
        meta_output_dirs = []
        verifier_output_dirs = []

        for idx in batch_indices:
            found = llm_find_case_and_label(data_dir, alarm_type, idx)
            if found is None:
                skipped.append(idx)
                continue

            _case_file, label_file = found
            try:
                alarm_data = build_competition_alarm_data(label_file, alarm_type)

                meta_output_dir = os.path.join(
                    final_output_dir,
                    f"{alarm_type}_analysis_Meta_{idx}",
                )
                verifier_output_dir = os.path.join(
                    final_output_dir,
                    f"{alarm_type}_analysis_Reasoner_{idx}",
                )
                ensure_dir(meta_output_dir)
                ensure_dir(verifier_output_dir)

                alarm_data_list.append(alarm_data)
                label_files_list.append(label_file)
                valid_indices.append(idx)
                meta_output_dirs.append(meta_output_dir)
                verifier_output_dirs.append(verifier_output_dir)
            except Exception as exc:
                print(f"idx={idx}: Competition 数据准备失败 - {exc}")
                skipped.append(idx)

        if not alarm_data_list:
            continue

        try:
            meta_responses = rca_generator.generate_rca_analysis_competition_batch(
                alarm_data_list,
                meta_output_dirs,
            )
        except Exception as exc:
            print(f"Competition 分析失败: {exc}")
            skipped.extend(valid_indices)
            continue

        meta_success_count = min(len(meta_responses), len(valid_indices))
        skipped.extend(valid_indices[meta_success_count:])

        verifier_data_list = []
        valid_verifier_dirs = []
        valid_verifier_indices = []

        for pos in range(meta_success_count):
            idx = valid_indices[pos]
            label_file = label_files_list[pos]
            current_reasoner_file = os.path.join(
                meta_output_dirs[pos],
                "raw_responses.txt",
            )

            if not os.path.exists(current_reasoner_file):
                print(f"idx={idx}: 未找到 Competition 输出文件: {current_reasoner_file}")
                skipped.append(idx)
                continue

            try:
                verifier_data = load_alarm_data_verifier(
                    label_file,
                    current_reasoner_file,
                    alarm_type,
                )
                sop = get_sop_value_from_label(label_file)
                require_sop(sop, label_file)
                verifier_data["sop"] = sop
                verifier_data_list.append(verifier_data)
                valid_verifier_dirs.append(verifier_output_dirs[pos])
                valid_verifier_indices.append(idx)
            except Exception as exc:
                print(f"idx={idx}: Verifier 数据准备失败 - {exc}")
                skipped.append(idx)

        if not verifier_data_list:
            continue

        try:
            verifier_responses = rca_generator.generate_rca_analysis_verifier_batch(
                verifier_data_list,
                valid_verifier_dirs,
            )
            success_count = min(len(verifier_responses), len(valid_verifier_indices))
            processed.extend(valid_verifier_indices[:success_count])
            skipped.extend(valid_verifier_indices[success_count:])
        except Exception as exc:
            print(f"Verifier 分析失败: {exc}")
            skipped.extend(valid_verifier_indices)

    processed = sorted(set(processed))
    skipped = sorted(set(skipped) - set(processed))

    payload = {
        "meta": {
            "mode": "llm_infer",
            "llm_mode": "competition",
            "meta_only": False,
            "pipeline": "competition_then_verifier",
            "scenario_name": scenario_name,
            "alarm_type": alarm_type,
            "data_dir": data_dir,
            "indices": indices,
            "batch_size": batch_size,
            "model_path": getattr(config, "MODEL_PATH"),
            "output_dir": final_output_dir,
        },
        "summary": {
            "processed_count": len(processed),
            "skipped_count": len(skipped),
            "processed_indices": processed,
            "skipped_indices": skipped,
        },
        "results": [],
    }
    save_infer_outputs(payload, final_output_dir, output_format)
    return payload


def llm_cooperation_one_scenario(
    scenario: Dict[str, Any],
    indices: List[int],
    rca_generator,
    load_alarm_data_meta,
    load_alarm_data_reasoner_cooperation,
    output_dir: Optional[str],
    output_format: str,
    batch_size: int,
    max_rounds: int,
) -> Dict[str, Any]:
    """
    Cooperation 完整流程：
        Meta -> Reasoner -> 下一轮 Meta/Reasoner，直到早停或 max_rounds。
    """
    scenario_name = scenario["name"]
    alarm_type = scenario["alarm_type"]
    data_dir = scenario["data_dir"]

    run_name = f"idx_{min(indices)}_{max(indices)}" if indices else "empty"
    final_output_dir = build_output_dir(
        "llm_infer_cooperation",
        scenario_name,
        output_dir,
        run_name,
    )

    processed: List[int] = []
    skipped: List[int] = []
    early_stop_marker = '["(上一轮次已经符合要求)"]'

    for batch_indices in iter_batches(indices, batch_size):
        label_file_by_idx: Dict[int, str] = {}
        valid_indices: List[int] = []

        for idx in batch_indices:
            found = llm_find_case_and_label(data_dir, alarm_type, idx)
            if found is None:
                skipped.append(idx)
                continue
            _case_file, label_file = found
            label_file_by_idx[idx] = label_file
            valid_indices.append(idx)

        if not valid_indices:
            continue

        early_stop_flags = {idx: False for idx in valid_indices}
        current_rounds = {idx: 0 for idx in valid_indices}
        failed_indices: Set[int] = set()

        for round_num in range(max_rounds):
            print(f"[Cooperation] round={round_num}, batch={batch_indices}")
            active_indices = [
                idx for idx in valid_indices
                if (
                    not early_stop_flags[idx]
                    and idx not in failed_indices
                    and current_rounds[idx] == round_num
                )
            ]
            if not active_indices:
                break

            meta_data_list = []
            meta_output_dirs = []
            meta_indices = []

            for idx in active_indices:
                label_file = label_file_by_idx[idx]
                current_round = current_rounds[idx]
                last_meta_output = None
                last_reasoner_output = None

                if current_round > 0:
                    prev_round = current_round - 1
                    last_meta_output = os.path.join(
                        final_output_dir,
                        str(idx),
                        f"round{prev_round}",
                        "meta",
                        "raw_responses.txt",
                    )
                    last_reasoner_output = os.path.join(
                        final_output_dir,
                        str(idx),
                        f"round{prev_round}",
                        "reasoner",
                        "raw_responses.txt",
                    )

                    if os.path.exists(last_meta_output):
                        try:
                            with open(last_meta_output, "r", encoding="utf-8") as f:
                                content = f.read()
                            if early_stop_marker in content:
                                print(f"idx={idx}: 上一轮 Meta 已符合要求，早停")
                                early_stop_flags[idx] = True
                                continue
                        except Exception as exc:
                            print(f"idx={idx}: 读取上一轮 Meta 失败 - {exc}")
                            failed_indices.add(idx)
                            continue

                output_dir_meta = os.path.join(
                    final_output_dir,
                    str(idx),
                    f"round{current_round}",
                    "meta",
                )
                ensure_dir(output_dir_meta)

                try:
                    meta_data = load_alarm_data_meta(
                        label_file,
                        alarm_type,
                        last_meta_output,
                        last_reasoner_output,
                    )
                    sop = get_sop_value_from_label(label_file)
                    require_sop(sop, label_file)
                    meta_data["sop"] = sop
                    meta_data_list.append(meta_data)
                    meta_output_dirs.append(output_dir_meta)
                    meta_indices.append(idx)
                except Exception as exc:
                    print(f"idx={idx}: Cooperation Meta 数据加载失败 - {exc}")
                    failed_indices.add(idx)

            if not meta_data_list:
                continue

            try:
                meta_responses = rca_generator.generate_rca_analysis_meta(
                    meta_data_list,
                    meta_output_dirs,
                    meta_indices,
                )
            except Exception as exc:
                print(f"Cooperation Meta 批处理失败: {exc}")
                failed_indices.update(meta_indices)
                continue

            meta_success_count = min(len(meta_responses), len(meta_indices))
            meta_success_indices = meta_indices[:meta_success_count]
            failed_indices.update(meta_indices[meta_success_count:])

            for idx in meta_success_indices:
                meta_file = os.path.join(
                    final_output_dir,
                    str(idx),
                    f"round{current_rounds[idx]}",
                    "meta",
                    "raw_responses.txt",
                )
                if not os.path.exists(meta_file):
                    print(f"idx={idx}: Meta 输出不存在: {meta_file}")
                    failed_indices.add(idx)
                    continue
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    if early_stop_marker in content:
                        print(f"idx={idx}: 本轮 Meta 已符合要求，早停")
                        early_stop_flags[idx] = True
                except Exception as exc:
                    print(f"idx={idx}: 读取 Meta 输出失败 - {exc}")
                    failed_indices.add(idx)

            reasoner_data_list = []
            reasoner_output_dirs = []
            reasoner_indices = []

            for idx in meta_success_indices:
                if early_stop_flags[idx] or idx in failed_indices:
                    continue

                current_round = current_rounds[idx]
                meta_file = os.path.join(
                    final_output_dir,
                    str(idx),
                    f"round{current_round}",
                    "meta",
                    "raw_responses.txt",
                )
                try:
                    reasoner_data = load_alarm_data_reasoner_cooperation(
                        label_file_by_idx[idx],
                        meta_file,
                        alarm_type,
                    )
                    output_dir_reasoner = os.path.join(
                        final_output_dir,
                        str(idx),
                        f"round{current_round}",
                        "reasoner",
                    )
                    ensure_dir(output_dir_reasoner)
                    reasoner_data_list.append(reasoner_data)
                    reasoner_output_dirs.append(output_dir_reasoner)
                    reasoner_indices.append(idx)
                except Exception as exc:
                    print(f"idx={idx}: Reasoner 数据加载失败 - {exc}")
                    failed_indices.add(idx)

            if not reasoner_data_list:
                continue

            try:
                reasoner_responses = rca_generator.generate_rca_analysis_reasoner(
                    reasoner_data_list,
                    reasoner_output_dirs,
                    reasoner_indices,
                )
            except Exception as exc:
                print(f"Reasoner 批处理失败: {exc}")
                failed_indices.update(reasoner_indices)
                continue

            reasoner_success_count = min(len(reasoner_responses), len(reasoner_indices))
            reasoner_success_indices = reasoner_indices[:reasoner_success_count]
            failed_indices.update(reasoner_indices[reasoner_success_count:])
            for idx in reasoner_success_indices:
                current_rounds[idx] += 1

        batch_processed = [idx for idx in valid_indices if idx not in failed_indices]
        processed.extend(batch_processed)
        skipped.extend(sorted(failed_indices))

    processed = sorted(set(processed))
    skipped = sorted(set(skipped) - set(processed))

    payload = {
        "meta": {
            "mode": "llm_infer",
            "llm_mode": "cooperation",
            "meta_only": False,
            "pipeline": "multi_round_meta_reasoner",
            "scenario_name": scenario_name,
            "alarm_type": alarm_type,
            "data_dir": data_dir,
            "indices": indices,
            "batch_size": batch_size,
            "max_rounds": max_rounds,
            "model_path": getattr(config, "MODEL_PATH"),
            "output_dir": final_output_dir,
        },
        "summary": {
            "processed_count": len(processed),
            "skipped_count": len(skipped),
            "processed_indices": processed,
            "skipped_indices": skipped,
        },
        "results": [],
    }
    save_infer_outputs(payload, final_output_dir, output_format)
    return payload


def llm_infer(args: argparse.Namespace) -> List[Dict[str, Any]]:
    scenarios = scan_label_root(
        args.agentdigest_label_root,
        args.scenario,
    )

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else getattr(config, "BATCH_SIZE", 8)
    )

    state = load_infer_state(args.state_file)
    plans = []

    for scenario in scenarios:
        indices, should_update_state, previous_last_index = get_incremental_indices(
            scenario=scenario,
            mode="llm_infer",
            args=args,
            state=state,
        )
        if not indices:
            print(f"[llm_infer] {scenario['name']} 没有新增数据，跳过")
            continue
        plans.append((scenario, indices, should_update_state, previous_last_index))

    if not plans:
        return []

    (
        RCAGenerator,
        load_alarm_data_meta,
        load_alarm_data_reasoner_cooperation,
        load_alarm_data_verifier,
    ) = _lazy_import_llm_modules()
    rca_generator = RCAGenerator(getattr(config, "MODEL_PATH"))

    payloads = []

    for scenario, indices, should_update_state, previous_last_index in plans:
        if args.llm_mode == "competition":
            competition_indices = list(indices)
            cooperation_indices = []
        elif args.llm_mode == "cooperation":
            cooperation_indices = list(indices)
            competition_indices = []
        else:
            cooperation_indices, competition_indices = split_indices_by_sop(
                scenario["data_dir"],
                scenario["alarm_type"],
                indices,
            )

        print("=" * 100)
        print(f"[llm_infer][incremental] 场景: {scenario['name']}")
        print(f"data_dir          : {scenario['data_dir']}")
        print(f"indices           : {indices}")
        print(f"llm_mode          : {args.llm_mode}")
        print("full_pipeline     : True")
        print(
            "cooperation count : "
            f"{len(cooperation_indices)}，条件：semantic_labels['sop'] 为空"
        )
        print(
            "competition count : "
            f"{len(competition_indices)}，条件：semantic_labels['sop'] 非空"
        )
        print("=" * 100)

        scenario_payloads = []
        success_indices: Set[int] = set()

        try:
            if cooperation_indices:
                payload = llm_cooperation_one_scenario(
                    scenario=scenario,
                    indices=cooperation_indices,
                    rca_generator=rca_generator,
                    load_alarm_data_meta=load_alarm_data_meta,
                    load_alarm_data_reasoner_cooperation=load_alarm_data_reasoner_cooperation,
                    output_dir=args.output_dir,
                    output_format=args.output_format,
                    batch_size=batch_size,
                    max_rounds=args.max_rounds,
                )
                scenario_payloads.append(payload)
                success_indices.update(
                    payload.get("summary", {}).get("processed_indices", [])
                )

            if competition_indices:
                payload = llm_competition_one_scenario(
                    scenario=scenario,
                    indices=competition_indices,
                    rca_generator=rca_generator,
                    load_alarm_data_verifier=load_alarm_data_verifier,
                    output_dir=args.output_dir,
                    output_format=args.output_format,
                    batch_size=batch_size,
                )
                scenario_payloads.append(payload)
                success_indices.update(
                    payload.get("summary", {}).get("processed_indices", [])
                )

            payloads.extend(scenario_payloads)

            if should_update_state:
                new_checkpoint = advance_checkpoint(
                    selected_indices=indices,
                    success_indices=success_indices,
                    previous_last_index=previous_last_index,
                )
                if new_checkpoint > previous_last_index:
                    update_last_processed_index(
                        state=state,
                        mode="llm_infer",
                        scenario=scenario,
                        last_index=new_checkpoint,
                    )
                    save_infer_state(state, args.state_file)

        except Exception as exc:
            print(f"[Error][llm_infer] {scenario['name']} 处理失败: {exc}")
            traceback.print_exc()
            if args.strict:
                raise

    return payloads
