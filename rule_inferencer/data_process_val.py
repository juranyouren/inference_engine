import os
import json
import ast
import glob
import random
from typing import List, Dict, Any, Tuple, Literal, Optional
import numpy as np
from utils.public_functions import CaseDataLoader
import difflib
import math
from numbers import Real
from copy import deepcopy
from collections import Counter



def get_feature_value(
    target_name: str,
    feature_names,
    feature_values,
    cutoff: float = 0.6
):
    """
    在 feature_names 中寻找与 target_name 最相似的特征名，并返回其值
    """
    matches = difflib.get_close_matches(
        target_name,
        feature_names,
        n=1,
        cutoff=cutoff
    )
    if not matches:
        raise KeyError(f"No similar feature found for {target_name}")

    matched_name = matches[0]
    idx = feature_names.index(matched_name)
    return feature_values[idx], matched_name

def compute_flash_count(flash_dict: Dict[str, Any]) -> int:
    """
    根据日志字典计算闪断次数：
    - active / clear 成对 → min(active, clear)
    - PEER_STATE_CHG → 直接计数
    - 最终返回所有模板中的最大值
    """
    FLASH_EVENT_PAIRS = [
        ("linkDown_active(l)", "linkDown_clear(l)"),
        ("hwPortDown(l)", "hwPortUp(l)"),
        ("hwBgpBackwardTransition_active(l)", "hwBgpBackwardTransition_clear(l)"),
        ("bgpBackwardTransition_active(l)", "bgpBackwardTransition_clear(l)"),
    ]

    FLASH_SINGLE_EVENTS = [
        "PEER_STATE_CHG(l)",
    ]
    if not isinstance(flash_dict, dict):
        return 0

    counts = []

    # 1. 成对事件
    for active_key, clear_key in FLASH_EVENT_PAIRS:
        active_cnt = int(flash_dict.get(active_key, 0))
        clear_cnt = int(flash_dict.get(clear_key, 0))
        counts.append(min(active_cnt, clear_cnt))

    # 2. 单事件（直接计数）
    for key in FLASH_SINGLE_EVENTS:
        counts.append(int(flash_dict.get(key, 0)))

    return max(counts) if counts else 0

def extract_anomaly_logs_features(
    anomaly_logs: Dict[str, Any]
) -> Dict[str, int]:
    """
    从 semantic_labels["anomaly_logs"] 中提取本端闪断次数。
    """
    local_flash = anomaly_logs.get("local_flash", {})

    return {
        "anomaly_logs-local_flash_count": compute_flash_count(local_flash)
    }


def extract_other_info_features(
    other_info: Dict[str, Any],
    alarm_time: Any
) -> Dict[str, int]:
    features = {}

    # ---------- command_echo_info ----------
    command_info = other_info.get("command_echo_info", {})

    # BGP 状态：0 表示正常，1 表示异常
    bgp_status = command_info.get("BGP_protocol_status")
    if bgp_status is not None:
        features["other_info-command_echo_info-bgp_status"] = (
            0 if bgp_status == "Established" else 1
        )

    def normalize_interface_status(value: Any) -> str:
        """统一接口状态格式，便于兼容大小写和首尾空格。"""
        return value.strip().upper() if isinstance(value, str) else ""

    # ---------- 本端接口状态 ----------
    local_status_raw = command_info.get("local_interface_status")

    if local_status_raw is not None:
        local_status = normalize_interface_status(local_status_raw)

        # 本端是否主动 shutdown
        features["other_info-command_echo_info-local_shutdown"] = (
            1 if local_status == "ADMINISTRATIVELY DOWN" else 0
        )

        # 本端是否发生 ERROR DOWN
        features["other_info-command_echo_info-local_errordown"] = (
            1 if local_status == "ERROR DOWN" else 0
        )

        # 本端接口状态：UP 为正常 0，其他状态为异常 1
        features["other_info-command_echo_info-local_interface_status"] = (
            0 if local_status == "UP" else 1
        )

    # ---------- 对端接口状态 ----------
    remote_status_raw = command_info.get("remote_interface_status")

    if remote_status_raw is not None:
        remote_status = normalize_interface_status(remote_status_raw)

        # 对端是否主动 shutdown
        features["other_info-command_echo_info-peer_shutdown"] = (
            1 if remote_status == "ADMINISTRATIVELY DOWN" else 0
        )

    # ---------- sys_running_time 派生特征 ----------
    sys_time = command_info.get("sys_running_time")

    # 登录状态
    features[
        "other_info-command_echo_info-sys_running_time-login_status"
    ] = (
        0
        if isinstance(sys_time, (int, float)) and sys_time >= 0
        else 1
    )

    # 重启状态
    if (
        isinstance(sys_time, (int, float))
        and isinstance(alarm_time, (int, float))
        and (sys_time - alarm_time + 3600 * 1000) > 0
    ):
        reboot_status = 0
    else:
        reboot_status = 1

    features[
        "other_info-command_echo_info-sys_running_time-reboot_status"
    ] = reboot_status

    # ---------- device_info ----------
    device_info = other_info.get("device_info", {})

    def map_owner(value: Any) -> int:
        return 0 if isinstance(value, str) and "DCN" in value else 1

    def map_resource_status(value: Any) -> int:
        return 0 if value in (None, "", "Running") else 1

    for side in ["local_device", "remote_device"]:
        dev = device_info.get(side, {})

        features[f"other_info-device_info-{side}-owner"] = (
            map_owner(dev.get("owner"))
        )
        features[f"other_info-device_info-{side}-resource_status"] = (
            map_resource_status(dev.get("resource_status"))
        )

    # local_ifname 中包含 Vlan，表示对端为服务器
    local_ifname = device_info.get("local_ifname")
    if local_ifname is not None:
        features["other_info-device_info-peer_is_server"] = (
            1
            if isinstance(local_ifname, str)
            and "vlan" in local_ifname.lower()
            else 0
        )

    # ---------- local_server_status ----------
    server_status = device_info.get("local_server_status", {})
    life_cycle = server_status.get("life_cycle_status")

    features[
        "other_info-device_info-local_server_status-life_cycle_status"
    ] = (
        0 if life_cycle is None or life_cycle == "使用中" else 1
    )

    peer_is_server_key = "other_info-device_info-peer_is_server"
    remote_owner_key = "other_info-device_info-remote_device-owner"

    peer_is_server = features.get(peer_is_server_key)
    remote_owner = features.get(remote_owner_key)

    # 对端是服务器，或对端设备不属于 DCN
    if peer_is_server != 0 or remote_owner != 0:
        features[
            "other_info-device_info-peer_server_or_non_dcn_owner"
        ] = 1
    else:
        features[
            "other_info-device_info-peer_server_or_non_dcn_owner"
        ] = 0

    return features

def extract_kpi_duration(segments):
    if not isinstance(segments, list) or len(segments) == 0:
        return 0.0

    valid_segments = [
        seg for seg in segments
        if "start_time" in seg and "end_time" in seg
    ]

    if len(valid_segments) == 0:
        return 0.0

    min_start = min(seg.get("start_time", 0) for seg in valid_segments)
    max_end = max(seg.get("end_time", 0) for seg in valid_segments)

    return float(max_end - min_start) / 60


def select_nearest_abnormal_segment(alarm_time, abnormal_timing_segments):
    """
    根据 alarm_time 选择距离最近的异常段；
    当 alarm_time 为 None 时，选择最后一个异常段（按 end_time 最大）。

    参数:
        alarm_time: int | None
            告警时间戳。支持毫秒级或秒级；若为 None，则选择最后一个异常段。
        abnormal_timing_segments: list[dict]
            异常段列表，每个元素形如:
            {
                "start_time": 1761071460,
                "end_time": 1761072120,
                "value": [...],
                "pattern": "levelUp"
            }

    返回:
        dict，形如:
        {
            "value": [...],
            "duration": 11.0,
            "start_time": 1761071460,
            "end_time": 1761072120,
            "pattern": "levelUp"
        }

        若 abnormal_timing_segments 为空，则返回 None
    """
    if not abnormal_timing_segments:
        return None

    # alarm_time 为 None 时，选择最后一个异常段（按 end_time 最大）
    if alarm_time is None:
        target_seg = max(
            abnormal_timing_segments,
            key=lambda seg: seg.get("end_time", -1)
        )
    else:
        # alarm_time 若为毫秒级，则转为秒级
        alarm_time_sec = alarm_time / 1000 if alarm_time > 1e12 else alarm_time

        def segment_distance(seg):
            start = seg["start_time"]
            end = seg["end_time"]

            # 告警时间落在异常段内，距离为 0
            if start <= alarm_time_sec <= end:
                return 0

            # 否则取到区间边界的最小距离
            return min(abs(alarm_time_sec - start), abs(alarm_time_sec - end))

        target_seg = min(abnormal_timing_segments, key=segment_distance)

    duration_minutes = (target_seg["end_time"] - target_seg["start_time"]) / 60

    return {
        "value": target_seg.get("value", []),
        "duration": duration_minutes,
        "start_time": target_seg.get("start_time"),
        "end_time": target_seg.get("end_time"),
        "pattern": target_seg.get("pattern"),
    }

def log_normalize_value(x: float) -> float:
    """
    对特征值做 log1p 归一化。
    KPI 和 duration 通常为非负值；若出现负值，则使用 signed log1p 保留符号。
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0

    if x >= 0:
        return float(np.log1p(x))
    else:
        return -float(np.log1p(abs(x)))



def extract_anomaly_kpi_features(
    anomaly_kpi: Dict[str, Any],
    alarm_time: Any,
    segment_mode: Literal["mix", "alarm"] = "mix"
) -> Dict[str, int]:
    """
    提取 anomaly_kpi 特征。

    segment_mode:
    - mix：使用所有异常段
    - alarm：使用距离 alarm_time 最近的异常段

    每个 KPI 提取：
    - 最大值是否位于 [90, 100]
    - 最大值是否大于 100
    - 持续时间是否大于 15
    - 平均值是否小于 80

    缺少有效异常段或数值时，不写入对应特征。
    """
    if segment_mode not in {"mix", "alarm"}:
        raise ValueError(
            f"Unsupported segment_mode: {segment_mode}"
        )

    features = {}

    kpi_keys = [
        "traffic_in",
        # "traffic_out",
        # "drop_packet_rate",
        # "error_packet_rate",
        "offline_loss_rate",
        # "cpu_utilization",
        # "memory_utilization",
        # "temperature",
    ]

    for kpi in kpi_keys:
        kpi_data = anomaly_kpi.get(kpi, {})
        segments = kpi_data.get("abnormal_timing_segments", [])

        if not isinstance(segments, list) or not segments:
            continue

        values = []
        duration = None

        # 使用全部异常段
        if segment_mode == "mix":
            for segment in segments:
                segment_values = segment.get("value", [])

                if isinstance(segment_values, list):
                    values.extend(segment_values)

            duration = extract_kpi_duration(segments)

        # 使用距离告警时间最近的异常段
        else:
            segment = select_nearest_abnormal_segment(
                alarm_time,
                segments
            )

            if segment is None:
                continue

            segment_values = segment.get("value", [])

            if isinstance(segment_values, list):
                values = segment_values

            duration = segment.get("duration")

        # 过滤无法参与计算的值，不做归一化
        valid_values = [
            value
            for value in values
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(value)
            )
        ]

        if not valid_values:
            continue

        values_array = np.asarray(valid_values, dtype=float)

        max_value = float(values_array.max())
        mean_value = float(values_array.mean())

        prefix = f"anomaly_kpi-{kpi}"

        features[f"{prefix}-max_in_90_100"] = int(
            90 <= max_value < 100
        )

        features[f"{prefix}-max_over_100"] = int(
            max_value >= 100
        )

        features[f"{prefix}-max_in_90_110"] = int(
            90 <= max_value <= 110
        )

        features[f"{prefix}-max_over_110"] = int(
            max_value > 110
        )

        features[f"{prefix}-mean_below_80"] = int(
            mean_value < 80
        )

        if isinstance(duration, (int, float)):
            features[f"{prefix}-duration_over_15"] = int(
                duration > 15
            )

    return features

def uniform_feature_subsample(
    feature_names: List[str],
    feature_values: List[float],
    keep_ratio: float,
):
    """
    按比例均匀下采样特征（保持顺序一致）
    """
    assert 0 < keep_ratio <= 1.0

    n = len(feature_names)
    if keep_ratio == 1.0:
        return feature_names, feature_values

    step = int(1 / keep_ratio)
    if step <= 1:
        return feature_names, feature_values

    kept_indices = list(range(0, n, step))

    new_feature_names = [feature_names[i] for i in kept_indices]
    new_feature_values = [feature_values[i] for i in kept_indices]

    return new_feature_names, new_feature_values

def extract_all_features(
    semantic_labels: Dict[str, Any],
    kpi_mode: str = "mix"
) -> Tuple[List[str], List[float], Dict[str, float]]:
    """
    组织所有特征提取模块，生成：
    - feature_names
    - feature_values
    - features_dict

    kpi_mode:
    - mix：使用全部异常段
    - alarm：使用距离告警时间最近的异常段
    """
    features: Dict[str, float] = {}
    alarm_time = semantic_labels.get("alarm_time")

    # ========= anomaly_logs =========
    anomaly_logs = semantic_labels.get("anomaly_logs", {})

    if isinstance(anomaly_logs, str):
        try:
            anomaly_logs = json.loads(anomaly_logs)
        except Exception as exc:
            raise TypeError(
                f"anomaly_logs 应为 dict，但当前为 str: {anomaly_logs}"
            ) from exc

    if not isinstance(anomaly_logs, dict):
        anomaly_logs = {}

    features.update(
        extract_anomaly_logs_features(anomaly_logs)
    )

    # ========= anomaly_kpi =========
    anomaly_kpi = semantic_labels.get("anomaly_kpi", {})

    if not isinstance(anomaly_kpi, dict):
        anomaly_kpi = {}

    features.update(
        extract_anomaly_kpi_features(
            anomaly_kpi,
            alarm_time,
            segment_mode=kpi_mode
        )
    )

    # ========= other_info =========
    other_info = semantic_labels.get("other_info", {})

    if not isinstance(other_info, dict):
        other_info = {}

    features.update(
        extract_other_info_features(other_info, alarm_time)
    )

    # 按特征名排序，保证不同 case 的特征顺序一致
    feature_names = sorted(features)
    feature_values = [features[name] for name in feature_names]

    return feature_names, feature_values, features


def extract_features_from_case(
    case_data: Dict[str, Any],
    kpi_mode: str = "mix"
) -> Tuple[List[str], List[float], Dict[str, float]]:
    """
    从单个 case_data 中提取特征。

    返回：
    - feature_names
    - feature_values
    - features_dict
    """
    semantic_labels = case_data["semantic_labels"]

    feature_names, feature_values, features_dict = extract_all_features(
        semantic_labels,
        kpi_mode=kpi_mode
    )

    return feature_names, feature_values, features_dict

def extract_features_from_case_range(
    dataloader: CaseDataLoader,
    start_id: int,
    end_id: int,
    kpi_mode: str = "mix",
    strict_feature_order: bool = True
):
    """
    使用 dataloader 按 ID 区间读取 case，并提取特征。

    返回：
    - feature_names
    - X
    - valid_indices
    - features_dict_by_idx
    """
    cases, indices = dataloader.get_cases_by_id_range(
        start_id,
        end_id
    )

    X = []
    valid_indices = []
    feature_names = None

    # case_idx -> features_dict
    features_dict_by_idx = {}

    for case_data, idx in zip(cases, indices):
        try:
            fnames, fvalues, features_dict = (
                extract_features_from_case(
                    case_data,
                    kpi_mode=kpi_mode
                )
            )

            # print(
            #     f"Case {idx}: feature number is {len(fnames)}"
            # )

        except Exception as exc:
            print(
                f"[Warning] feature extraction failed "
                f"for case {idx}: {exc}"
            )
            continue

        if feature_names is None:
            feature_names = fnames

        elif strict_feature_order:
            assert fnames == feature_names, (
                f"Feature mismatch in case {idx}"
            )

        X.append(fvalues)
        valid_indices.append(idx)
        features_dict_by_idx[idx] = features_dict

    return (
        feature_names,
        X,
        valid_indices,
        features_dict_by_idx
    )

# def extract_features_from_case_range(
#     dataloader: CaseDataLoader,
#     start_id: int,
#     end_id: int
# ):
#     """
#     使用 dataloader 按 id 区间读取 case，并提取特征
#     """
#     cases, indices = dataloader.get_cases_by_id_range(start_id, end_id)

#     X = []
#     valid_indices = []
#     feature_names = None

#     for case_data, idx in zip(cases, indices):
#         try:
#             fnames, fvalues = extract_features_from_case(case_data)
#             print(f"Feature number is {len(fnames)}")
#             if len(fnames) != 78:
#                 print(f"ERROR: Feature number")
#         except Exception as e:
#             print(f"[Warning] feature extraction failed for case {idx}: {e}")
#             continue

#         if feature_names is None:
#             feature_names = fnames
#         else:
#             # 强约束：特征顺序必须一致
#             assert fnames == feature_names, f"Feature mismatch in case {idx}"

#         X.append(fvalues)
#         valid_indices.append(idx)

#     return feature_names, X, valid_indices

def extract_root_cause_labels(case_list):
    """
    从 case_data 列表中提取 root_cause.category，并编码成整数标签
    """
    from sklearn.preprocessing import LabelEncoder

    raw_labels = []

    for case_data in case_list:
        category = case_data["root_cause"]
        raw_labels.append(category)

    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_labels)

    return y, encoder

def is_missing_value(value: Any) -> bool:
    """
    判断特征是否缺失。
    None 和 NaN 都视为缺失。
    """
    if value is None:
        return True

    if isinstance(value, Real):
        return math.isnan(float(value))

    return False


def satisfy_relation(
    value: Any,
    relation: str,
    threshold: Any
) -> bool:
    """
    判断特征值是否满足根因规则。
    """
    if relation == "eq":
        return value == threshold

    if relation == "ne":
        return value != threshold

    if relation == "gt":
        return value > threshold

    if relation == "ge":
        return value >= threshold

    if relation == "lt":
        return value < threshold

    if relation == "le":
        return value <= threshold

    if relation == "between":
        lower, upper = threshold
        return lower <= value <= upper

    raise ValueError(f"不支持的 relation：{relation}")

def _parse_root_cause_list(value: Any) -> List[str]:
    """将 list 或字符串形式的 list 解析为根因列表。"""
    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if not isinstance(value, str):
        return []

    value = value.strip()
    if not value:
        return []

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except (ValueError, SyntaxError):
        pass

    return []


def _load_root_cause_universe(
    alarm_type: str,
    template_dir: str
) -> List[str]:
    """
    从对应告警模板中读取根因全集。

    模板匹配格式：
        {alarm_type}*_label_*.json
    """
    pattern = os.path.join(
        template_dir,
        f"{alarm_type}*_label_*.json"
    )
    matched_templates = sorted(glob.glob(pattern))

    if not matched_templates:
        raise FileNotFoundError(
            f"未找到告警 {alarm_type} 的模板文件，"
            f"匹配模式为：{pattern}"
        )

    template_path = matched_templates[0]

    with open(template_path, "r", encoding="utf-8") as file:
        template_data = json.load(file)

    root_cause_candidates = template_data.get(
        "root_cause_candidates",
        []
    )

    root_cause_universe = []
    seen = set()

    for item in root_cause_candidates:
        if not isinstance(item, dict):
            continue

        root_cause = item.get("category")

        if root_cause is None or root_cause in seen:
            continue

        seen.add(root_cause)
        root_cause_universe.append(root_cause)

    if not root_cause_universe:
        raise ValueError(
            "模板文件中未提取到有效的 "
            f"root_cause_candidates.category：{template_path}"
        )

    return root_cause_universe


def complete_pred_rc_before_rerank(
    ret: Dict[str, Any],
    template_dir: str = "/home/sbp/huangzeshun/RCA_Units/template",
    target_size: int = 5,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    在重排前将每个 case 的 pred_rc 随机补齐到指定数量，
    并重新计算 ground truth 在 pred_rc 中的排名。

    处理结果直接写回：
        result["pred_rc"]：补全后的原始根因列表
        result["rank"]：result["rc"] 在 pred_rc 中的排名

    规则：
    1. 原 pred_rc 去重但保持顺序；
    2. 原列表少于 target_size 时，从根因全集随机补齐；
    3. 原列表不少于 target_size 时不截断；
    4. rank 从 1 开始，ground truth 不存在时为 None；
    5. 相同 seed、alarm_type 和 case_idx 的补全结果固定。
    """
    if target_size <= 0:
        raise ValueError(
            f"target_size 必须大于 0，当前值为：{target_size}"
        )

    results = ret.get("results", [])

    if not isinstance(results, list):
        raise TypeError("ret['results'] 必须是 list")

    # 避免相同 alarm_type 重复读取模板
    universe_cache: Dict[str, List[str]] = {}

    for result_index, result in enumerate(results):
        if not isinstance(result, dict):
            continue

        alarm_type = result.get("alarm_type")
        case_idx = result.get("case_idx")

        if not alarm_type:
            raise ValueError(
                f"第 {result_index} 个结果缺少 alarm_type，"
                "无法确定对应的根因全集"
            )

        if alarm_type not in universe_cache:
            universe_cache[alarm_type] = (
                _load_root_cause_universe(
                    alarm_type=alarm_type,
                    template_dir=template_dir
                )
            )

        root_cause_universe = universe_cache[alarm_type]

        # 解析并去重原始预测列表
        original_pred_rc = _parse_root_cause_list(
            result.get("pred_rc", [])
        )

        pred_rc_unique = []
        seen = set()

        for root_cause in original_pred_rc:
            if root_cause not in seen:
                seen.add(root_cause)
                pred_rc_unique.append(root_cause)

        need_count = max(
            0,
            target_size - len(pred_rc_unique)
        )

        if need_count > 0:
            remaining_root_causes = [
                root_cause
                for root_cause in root_cause_universe
                if root_cause not in seen
            ]

            if len(remaining_root_causes) < need_count:
                raise ValueError(
                    f"case_idx={case_idx} 无法补齐到 {target_size} 个根因："
                    f"当前有 {len(pred_rc_unique)} 个，"
                    f"根因全集中仅剩 {len(remaining_root_causes)} 个可补充"
                )

            # 每个 case 使用独立随机种子，避免批次划分影响结果
            if seed is None:
                case_random = random.Random()
            else:
                case_seed = (
                    f"{seed}|{alarm_type}|"
                    f"{case_idx if case_idx is not None else result_index}"
                )
                case_random = random.Random(case_seed)

            added_root_causes = case_random.sample(
                remaining_root_causes,
                need_count
            )

            pred_rc_unique.extend(added_root_causes)

        # 写回补全后的列表
        result["pred_rc"] = pred_rc_unique

        # 重新计算原始列表的 rank
        groundtruth = result.get("rc")

        if groundtruth in pred_rc_unique:
            result["rank"] = (
                pred_rc_unique.index(groundtruth) + 1
            )
        else:
            result["rank"] = None

    return ret



def calculate_root_cause_score(
    features: Dict[str, Any],
    root_cause: str,
    rule_config: Dict[str, Any],
    neutral_score: float = 0.499
) -> Dict[str, Any]:
    """
    计算单个根因的合理性分数。

    参数：
    - features：当前 case 的特征字典
    - root_cause：待评估根因
    - rule_config：当前告警场景的根因规则
    - neutral_score：所有相关特征均缺失时的中性分数

    返回：
    - score：加权合理性分数
    - coverage：证据覆盖率
    - matched_weight：满足规则的权重
    - available_weight：实际参与计算的权重
    - total_weight：该根因全部规则权重
    - skip_case：是否跳过当前 case
    """
    root_cause_rules = rule_config["root_causes"]

    if root_cause not in root_cause_rules:
        raise KeyError(f"未找到根因映射规则：{root_cause}")

    defaults = rule_config.get("defaults", {})
    default_weight = float(defaults.get("weight", 1))
    default_missing = defaults.get("missing", "ignore")

    rules = root_cause_rules[root_cause]

    total_weight = 0.0
    available_weight = 0.0
    matched_weight = 0.0

    matched_features = []
    unmatched_features = []
    missing_features = []

    for rule in rules:
        feature_name = rule["feature"]
        relation = rule["relation"]
        threshold = rule["threshold"]

        weight = float(rule.get("weight", default_weight))
        missing_policy = rule.get(
            "missing",
            default_missing
        )

        total_weight += weight

        value = features.get(feature_name)

        if is_missing_value(value):
            missing_features.append(feature_name)

            if missing_policy == "skip_case":
                return {
                    "rc": root_cause,
                    "score": None,
                    "coverage": 0.0,
                    "matched_weight": matched_weight,
                    "available_weight": available_weight,
                    "total_weight": total_weight,
                    "matched_features": matched_features,
                    "unmatched_features": unmatched_features,
                    "missing_features": missing_features,
                    "skip_case": True
                }

            if missing_policy == "ignore":
                continue

            raise ValueError(
                f"不支持的缺失策略：{missing_policy}"
            )

        available_weight += weight

        if satisfy_relation(value, relation, threshold):
            matched_weight += weight
            matched_features.append(feature_name)
        else:
            unmatched_features.append(feature_name)

    coverage = (
        available_weight / total_weight
        if total_weight > 0
        else 0.0
    )

    # 所有关联特征均缺失时，表示证据未知，而不是不合理
    if available_weight == 0:
        score = neutral_score
    else:
        score = matched_weight / available_weight

    return {
        "rc": root_cause,
        "score": score,
        "coverage": coverage,
        "matched_weight": matched_weight,
        "available_weight": available_weight,
        "total_weight": total_weight,
        "matched_features": matched_features,
        "unmatched_features": unmatched_features,
        "missing_features": missing_features,
        "skip_case": False
    }

def build_feature_check_tuples(
    features: Dict[str, Any],
    root_cause: str,
    rule_config: Dict[str, Any]
) -> tuple:
    """
    生成某个根因下每条特征规则的校验详情。

    每条记录格式：
    (
        feature_name,
        actual_value,
        relation,
        threshold,
        weight,
        missing_policy,
        check_result
    )
    """
    root_cause_rules = rule_config["root_causes"]

    if root_cause not in root_cause_rules:
        raise KeyError(f"未找到根因映射规则：{root_cause}")

    defaults = rule_config.get("defaults", {})
    default_weight = float(defaults.get("weight", 1))
    default_missing = defaults.get("missing", "ignore")

    check_details = []

    for rule in root_cause_rules[root_cause]:
        feature_name = rule["feature"]
        relation = rule["relation"]
        threshold = rule["threshold"]

        weight = float(
            rule.get("weight", default_weight)
        )
        missing_policy = rule.get(
            "missing",
            default_missing
        )

        value = features.get(feature_name)

        # 区间由 list 转为 tuple，写入 CSV 后更容易识别
        threshold_value = (
            tuple(threshold)
            if isinstance(threshold, list)
            else threshold
        )

        if is_missing_value(value):
            check_result = f"missing_{missing_policy}"

        else:
            matched = satisfy_relation(
                value=value,
                relation=relation,
                threshold=threshold
            )

            check_result = (
                "matched" if matched else "unmatched"
            )

        check_details.append(
            (
                feature_name,
                value,
                relation,
                threshold_value,
                weight,
                missing_policy,
                check_result
            )
        )

    return tuple(check_details)

def rerank_root_cause_results(
    ret: Dict[str, Any],
    dataloader: CaseDataLoader,
    rule_configs: Dict[str, Dict[str, Any]],
    kpi_mode: str = "mix",
    neutral_score: float = 0.5,
    in_place: bool = True,
    template_dir: Optional[str] = None,
    target_size: int = 5,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    根据重新提取的 case 特征，对 pred_rc 进行合理性重排。

    新增字段：
    - pred_rc_val：
        合理性重排后的根因列表

    - pred_top1_rc_val：
        重排后的 Top-1 根因

    - pred_rc_val_detail：
        各根因的完整字典形式评分详情

    - pred_rc_val_check：
        便于写入 CSV 和人工检查的元组形式校验详情

    - is_correct_val：
        重排后的 Top-1 是否正确

    - rerank_skipped：
        是否跳过重排

    - rerank_changed：
        根因列表顺序是否发生变化

    - pred_top1_changed：
        Top-1 根因是否发生变化
    """
    ret = complete_pred_rc_before_rerank(
        ret=ret,
        template_dir=(
            template_dir
            or "/home/sbp/huangzeshun/RCA_Units/template"
        ),
        target_size=target_size,
        seed=seed,
    )

    output = ret if in_place else deepcopy(ret)
    valid_results = output.get("results", [])

    if not valid_results:
        return output

    # 收集当前批次的 case ID
    case_ids = [
        result["case_idx"]
        for result in valid_results
        if result.get("case_idx") is not None
    ]

    if not case_ids:
        raise ValueError(
            "ret['results'] 中没有有效的 case_idx"
        )

    start_id = min(case_ids)
    # get_cases_by_id_range uses a half-open interval [start_id, end_id).
    # Include the largest case index in the current batch.
    end_id = max(case_ids) + 1

    # 一次性提取当前区间的特征字典
    _, _, _, features_dict_by_idx = (
        extract_features_from_case_range(
            dataloader=dataloader,
            start_id=start_id,
            end_id=end_id,
            kpi_mode=kpi_mode,
            strict_feature_order=False
        )
    )

    for result in valid_results:
        case_idx = result.get("case_idx")
        pred_rc = result.get("pred_rc", [])

        if not isinstance(pred_rc, list):
            raise TypeError(
                f"case {case_idx} 的 pred_rc 必须是列表"
            )

        original_top1 = pred_rc[0] if pred_rc else None

        # ==============================
        # 特征提取失败：保留原始排序
        # ==============================
        if case_idx not in features_dict_by_idx:
            result["pred_rc_val"] = list(pred_rc)
            result["pred_top1_rc_val"] = original_top1
            result["pred_rc_val_detail"] = []

            result["pred_rc_val_check"] = (
                (
                    "rerank_skipped",
                    "feature_extraction_failed"
                ),
            )

            result["rerank_skipped"] = True
            result["rerank_skip_reason"] = (
                "feature_extraction_failed"
            )

            result["rerank_changed"] = False
            result["pred_top1_changed"] = False

            result["is_correct_val"] = (
                original_top1 == result.get("rc")
            )
            continue

        features_dict = features_dict_by_idx[case_idx]
        alarm_type = result.get("alarm_type")

        if alarm_type not in rule_configs:
            raise KeyError(
                f"case {case_idx} 未找到告警场景规则："
                f"{alarm_type}"
            )

        rule_config = rule_configs[alarm_type]

        score_details = []
        should_skip_case = False

        # ==============================
        # 对每个候选根因评分
        # ==============================
        for original_rank, root_cause in enumerate(pred_rc, start=1):
            try:
                detail = calculate_root_cause_score(
                    features=features_dict,
                    root_cause=root_cause,
                    rule_config=rule_config,
                    neutral_score=neutral_score
                )

                detail["feature_checks"] = build_feature_check_tuples(
                    features=features_dict,
                    root_cause=root_cause,
                    rule_config=rule_config
                )

            except Exception as exc:
                raise RuntimeError(
                    f"case {case_idx} 计算根因 {root_cause} 的合理性失败"
                ) from exc

            # 以原候选根因为准，避免评分函数改变根因名称
            detail["rc"] = root_cause
            detail["original_rank"] = original_rank
            score_details.append(detail)

            if detail["skip_case"]:
                should_skip_case = True

        # ==============================
        # 关键特征缺失：不执行重排
        # ==============================
        if should_skip_case:
            ranked_details = score_details

            for detail in ranked_details:
                detail["adjusted_rank"] = detail[
                    "original_rank"
                ]

            result["pred_rc_val"] = list(pred_rc)
            result["pred_top1_rc_val"] = original_top1
            result["rerank_skipped"] = True
            result["rerank_skip_reason"] = (
                "required_feature_missing"
            )

        # ==============================
        # 正常进行重排
        # ==============================
        else:
            ranked_details = sorted(
                score_details,
                key=lambda item: (
                    -item["score"],
                    -item["coverage"],
                    item["original_rank"]
                )
            )

            for adjusted_rank, detail in enumerate(
                ranked_details,
                start=1
            ):
                detail["adjusted_rank"] = adjusted_rank

            result["pred_rc_val"] = [
                detail["rc"]
                for detail in ranked_details
            ]

            if Counter(result["pred_rc_val"]) != Counter(pred_rc):
                raise RuntimeError(
                    f"case {case_idx} 重排前后候选根因发生变化：\n"
                    f"重排前：{pred_rc}\n"
                    f"重排后：{result['pred_rc_val']}"
                )

            result["pred_top1_rc_val"] = (
                result["pred_rc_val"][0]
                if result["pred_rc_val"]
                else None
            )

            result["rerank_skipped"] = False
            result["rerank_skip_reason"] = None

        # 保留完整字典信息
        result["pred_rc_val_detail"] = ranked_details

        # ==============================
        # 构造适合保存到 CSV 的校验元组
        # ==============================
        result["pred_rc_val_check"] = tuple(
            (
                detail["rc"],
                detail["original_rank"],
                detail["adjusted_rank"],
                (
                    round(detail["score"], 4)
                    if detail["score"] is not None
                    else None
                ),
                round(detail["coverage"], 4),
                detail["matched_weight"],
                detail["available_weight"],
                detail["feature_checks"]
            )
            for detail in ranked_details
        )

        result["rerank_changed"] = (
            result["pred_rc_val"] != pred_rc
        )

        result["pred_top1_changed"] = (
            result["pred_top1_rc_val"]
            != original_top1
        )

        result["is_correct_val"] = (
            result["pred_top1_rc_val"]
            == result.get("rc")
        )

    return output
