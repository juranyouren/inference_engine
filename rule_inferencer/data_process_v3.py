import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from utils.public_functions import CaseDataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
import difflib



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

def extract_anomaly_logs_features(anomaly_logs: Dict[str, Any], selection) -> Dict[str, int]:
    """
    从 semantic_labels['anomaly_logs'] 中提取特征
    """
    features = {}

    # 1. 直接展开的字典型键
    ANOMALY_LOGS_SCHEMA = {
        "local_exception": [
            "Reason=Interface physical link is down",
            "Reason=BGP direct connect-interface down",
            "The local fault alarm has occurred",
            "The physical status of the port changed to down",
            "The remote fault alarm has occurred",
            "Optical Module is invalid",
            "The BFD session went Down",
            "hwBfdSessDown(t)",
            "CurrState=ESTABLISHED",
            "CMDRECORD",
            "REBOOT",
        ],
        "remote_exception": [
            "Reason=Interface physical link is down",
            "Reason=BGP direct connect-interface down",
            "The local fault alarm has occurred",
            "The physical status of the port changed to down",
            "The remote fault alarm has occurred",
            "Optical Module is invalid",
            "The BFD session went Down",
            "hwBfdSessDown(t)",
            "CurrState=ESTABLISHED",
            "CMDRECORD",
            "REBOOT",
        ],
    }

    ANOMALY_LOGS_SCHEMA_flash = {
        "local_flash": [
            "linkDown_active(l)",
            "linkDown_clear(l)",
            "hwPortDown(l)",
            "hwPortUp(l)",
            "hwBgpBackwardTransition_active(l)",
            "hwBgpBackwardTransition_clear(l)",
            "bgpBackwardTransition_active(l)",
            "bgpBackwardTransition_clear(l)",
            "PEER_STATE_CHG(l)",
        ],
        "remote_flash": [
            "linkDown_active(l)",
            "linkDown_clear(l)",
            "hwPortDown(l)",
            "hwPortUp(l)",
            "hwBgpBackwardTransition_active(l)",
            "hwBgpBackwardTransition_clear(l)",
            "bgpBackwardTransition_active(l)",
            "bgpBackwardTransition_clear(l)",
            "PEER_STATE_CHG(l)",
        ]
    }
    selection = selection["log"]


    for main_key, sub_keys in ANOMALY_LOGS_SCHEMA.items():
        sub_dict = anomaly_logs.get(main_key, {})

        for sub_key in sub_keys:
            if sub_key not in selection:
                continue
            feature_name = f"anomaly_logs-{main_key}-{sub_key}"

            if isinstance(sub_dict, dict) and sub_key in sub_dict:
                features[feature_name] = int(sub_dict.get(sub_key, 0))
            else:
                # 缺失字段 → 明确补 0
                features[feature_name] = 0

    

    if "template_occur_count" in selection:
        # print("template_occur_count in selection")
        for main_key, sub_keys in ANOMALY_LOGS_SCHEMA_flash.items():
            sub_dict = anomaly_logs.get(main_key, {})

            for sub_key in sub_keys:
                if sub_key not in selection:
                    continue
                feature_name = f"anomaly_logs-{main_key}-{sub_key}"

                if isinstance(sub_dict, dict) and sub_key in sub_dict:
                    features[feature_name] = int(sub_dict.get(sub_key, 0))
                else:
                    # 缺失字段 → 明确补 0
                    features[feature_name] = 0
    if "flash_count" in selection:
        local_flash_dict = anomaly_logs.get("local_flash", {})
        remote_flash_dict = anomaly_logs.get("remote_flash", {})

        features["anomaly_logs-local_flash_count"] = compute_flash_count(local_flash_dict)
        features["anomaly_logs-remote_flash_count"] = compute_flash_count(remote_flash_dict)

    # 2. shutdown：是否存在内容 → 二值特征
    shutdown_keys = ["local_shutdown", "remote_shutdown"]

    for key in shutdown_keys:
        sub_dict = anomaly_logs.get(key, {})
        feature_name = f"anomaly_logs-{key}"
        features[feature_name] = 1 if isinstance(sub_dict, dict) and len(sub_dict) > 0 else 0

    return features

def extract_anomaly_netstreams_features(
    anomaly_netstreams: Any
) -> Dict[str, int]:
    """
    anomaly_netstreams 是一个列表，
    特征值 = 列表中元素的个数
    """
    features = {}

    if isinstance(anomaly_netstreams, list):
        features["anomaly_netstreams-count"] = len(anomaly_netstreams)
    else:
        features["anomaly_netstreams-count"] = 0

    return features

def extract_other_info_features(
    other_info: Dict[str, Any],
    alarm_time: Any
) -> Dict[str, int]:
    features = {}

    # ---------- command_echo_info ----------
    command_info = other_info.get("command_echo_info", {})

    # BGP_protocol_status
    bgp_status = command_info.get("BGP_protocol_status")
    features["other_info-command_echo_info-BGP_protocol_status"] = (
        0 if bgp_status == "Established" else 1
    )

    # interface status mapping
    def map_interface_status(value):
        if value == "UP":
            return 0
        if value is None:
            return 1
        if value == "ERROR DOWN":
            return 2
        if value == "Administratively DOWN":
            return 3
        return 4

    features["other_info-command_echo_info-local_interface_status"] = \
        map_interface_status(command_info.get("local_interface_status"))

    features["other_info-command_echo_info-remote_interface_status"] = \
        map_interface_status(command_info.get("remote_interface_status"))

    # ---------- sys_running_time 派生两个特征 ----------
    sys_time = command_info.get("sys_running_time")

    # 1. 登录状态
    features["other_info-command_echo_info-sys_running_time-login_status"] = (
        0 if isinstance(sys_time, (int, float)) and sys_time >= 0 else 1
    )

    # 2. 重启状态（结合 alarm_time）
    if (
        isinstance(sys_time, (int, float)) and
        isinstance(alarm_time, (int, float)) and
        (sys_time - alarm_time + 3600 * 1000) > 0
    ):
        reboot_status = 0
    else:
        reboot_status = 1

    features["other_info-command_echo_info-sys_running_time-reboot_status"] = reboot_status

    # ---------- device_info ----------
    device_info = other_info.get("device_info", {})

    def map_owner(value):
        return 0 if isinstance(value, str) and "DCN" in value else 1

    def map_resource_status(value):
        return 0 if value in (None, "", "Running") else 1

    for side in ["local_device", "remote_device"]:
        dev = device_info.get(side, {})
        features[f"other_info-device_info-{side}-owner"] = \
            map_owner(dev.get("owner"))
        features[f"other_info-device_info-{side}-resource_status"] = \
            map_resource_status(dev.get("resource_status"))

    # local_ifname
    local_ifname = device_info.get("local_ifname")
    features["other_info-device_info-local_ifname"] = (
        1 if isinstance(local_ifname, str) and "Vlan" in local_ifname else 0
    )

    # local_server_status -> life_cycle_status
    server_status = device_info.get("local_server_status", {})
    life_cycle = server_status.get("life_cycle_status")
    features["other_info-device_info-local_server_status-life_cycle_status"] = (
        0 if life_cycle is None or life_cycle == "使用中" else 1
    )

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
    selection,
    use_log_norm: bool = True
) -> Dict[str, float]:
    """
    对 anomaly_kpi 中的各个 KPI 提取特征。

    规则：
    - mix: 基于所有异常段融合后的 values 统计 max/min/mean，duration 为最早 start 到最晚 end
    - alarming: 基于距离 alarm_time 最近的异常段统计 max/min/mean，duration 为该段持续时间
    - 所有被 selection 选中的特征都必须写入，缺失时补 0.0，保证不同 case 特征长度一致
    - 若 use_log_norm=True，则对最终写入的 KPI 特征值做 log1p 归一化
    """
    features = {}

    def add_feature(name: str, value: float):
        """
        统一写入特征，必要时做 log 归一化
        """
        if use_log_norm:
            features[name] = log_normalize_value(value)
        else:
            features[name] = float(value)

    kpi_keys = [
        "traffic_in",
        "traffic_out",
        "drop_packet_rate",
        "error_packet_rate",
        "offline_loss_rate",
        "cpu_utilization",
        "memory_utilization",
        "temperature",
    ]

    for kpi in kpi_keys:
        sub_selection = selection["kpi"][kpi]
        kpi_data = anomaly_kpi.get(kpi, {})
        segments = kpi_data.get("abnormal_timing_segments", [])

        # -------- mix --------
        mix_values: List[float] = []
        duration_mix = 0.0

        if isinstance(segments, list) and "mix" in sub_selection:
            for seg in segments:
                seg_values = seg.get("value", [])
                if isinstance(seg_values, list):
                    mix_values.extend(seg_values)

            if "duration" in sub_selection:
                duration_mix = extract_kpi_duration(segments)

        if len(mix_values) == 0:
            max_mix = min_mix = mean_mix = 0.0
        else:
            arr_mix = np.array(mix_values, dtype=float)
            max_mix = float(arr_mix.max())
            min_mix = float(arr_mix.min())
            mean_mix = float(arr_mix.mean())

        # -------- alarming --------
        alarming_values: List[float] = []
        duration_alarming = 0.0

        if isinstance(segments, list) and "alarming" in sub_selection:
            result = select_nearest_abnormal_segment(alarm_time, segments)
            if result is not None:
                alarming_values = result.get("value", [])
                duration_alarming = result.get("duration", 0.0)

        if len(alarming_values) == 0:
            max_alarming = min_alarming = mean_alarming = 0.0
        else:
            arr_alarming = np.array(alarming_values, dtype=float)
            max_alarming = float(arr_alarming.max())
            min_alarming = float(arr_alarming.min())
            mean_alarming = float(arr_alarming.mean())

        # -------- 写入特征：保证所有被选中特征固定输出 --------
        if "mix" in sub_selection:
            if "max" in sub_selection:
                add_feature(f"anomaly_kpi-{kpi}-max_mix", max_mix)
            if "min" in sub_selection:
                add_feature(f"anomaly_kpi-{kpi}-min_mix", min_mix)
            if "mean" in sub_selection:
                add_feature(f"anomaly_kpi-{kpi}-mean_mix", mean_mix)
            if "duration" in sub_selection:
                add_feature(f"anomaly_kpi-{kpi}-duration_mix", duration_mix)

        if "alarming" in sub_selection:
            if "max" in sub_selection:
                add_feature(f"anomaly_kpi-{kpi}-max_alarming", max_alarming)
            if "min" in sub_selection:
                add_feature(f"anomaly_kpi-{kpi}-min_alarming", min_alarming)
            if "mean" in sub_selection:
                add_feature(f"anomaly_kpi-{kpi}-mean_alarming", mean_alarming)
            if "duration" in sub_selection:
                add_feature(f"anomaly_kpi-{kpi}-duration_alarming", duration_alarming)

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

# 传入selection
def extract_all_features(
    semantic_labels: Dict[str, Any],
    selection
) -> Tuple[List[str], List[float], Dict[str, float]]:
    """
    组织所有特征提取模块，生成：
    - feature_names: List[str]
    - feature_values: List[float]
    - features_dict: Dict[str, float]
    """

    features: Dict[str, float] = {}
    alarm_time = semantic_labels.get("alarm_time", None)

    # ========= anomaly_logs =========
    anomaly_logs = semantic_labels.get("anomaly_logs", {})
    if isinstance(anomaly_logs, str):
        try:
            anomaly_logs = json.loads(anomaly_logs)
        except Exception:
            raise TypeError(f"anomaly_logs 应为 dict，但当前为 str: {anomaly_logs}")
    features.update(
        extract_anomaly_logs_features(anomaly_logs, selection)
    )

    # ========= anomaly_netstreams =========
    anomaly_netstreams = semantic_labels.get("anomaly_netstreams", None)
    features.update(
        extract_anomaly_netstreams_features(anomaly_netstreams)
    )

    # ========= anomaly_kpi =========
    anomaly_kpi = semantic_labels.get("anomaly_kpi", {})
    if anomaly_kpi == "" or anomaly_kpi is None or not isinstance(anomaly_kpi, dict):
        anomaly_kpi = {}

    features.update(
        extract_anomaly_kpi_features(anomaly_kpi, alarm_time, selection)
    )

    # ========= other_info（需要 alarm_time） =========
    other_info = semantic_labels.get("other_info", {})
    

    features.update(
        extract_other_info_features(other_info, alarm_time)
    )

    # ========= dict → label / value =========
    # 使用排序保证跨样本顺序稳定（非常关键）
    feature_names: List[str] = list(features.keys())
    feature_values: List[float] = [features[name] for name in feature_names]
    feature_names, feature_values = uniform_feature_subsample(
        feature_names,
        feature_values,
        keep_ratio=1  # 例如保留 70%
    )

    return feature_names, feature_values, features

def extract_features_from_case(case_data, selection):
    """
    从单个 case_data 中提取特征
    """
    semantic_labels = case_data["semantic_labels"]
    feature_names, feature_values, _ = extract_all_features(semantic_labels, selection)
    return feature_names, feature_values

def extract_features_from_case_range(
    dataloader: CaseDataLoader,
    start_id: int,
    end_id: int,
    selection
):
    """
    使用 dataloader 按 id 区间读取 case，并提取特征
    """
    cases, indices = dataloader.get_cases_by_id_range(start_id, end_id)

    X = []
    valid_indices = []
    feature_names = None

    for case_data, idx in zip(cases, indices):
        try:
            fnames, fvalues = extract_features_from_case(case_data, selection)
            print(f"Feature number is {len(fnames)}")
            if len(fnames) != 78:
                print(f"ERROR: Feature number")
        except Exception as e:
            print(f"[Warning] feature extraction failed for case {idx}: {e}")
            continue

        if feature_names is None:
            feature_names = fnames
        else:
            # 强约束：特征顺序必须一致
            assert fnames == feature_names, f"Feature mismatch in case {idx}"

        X.append(fvalues)
        valid_indices.append(idx)

    return feature_names, X, valid_indices

def extract_root_cause_labels(case_list):
    """
    从 case_data 列表中提取 root_cause.category，并编码成整数标签
    """
    raw_labels = []

    for case_data in case_list:
        category = case_data["root_cause"]
        raw_labels.append(category)

    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_labels)

    return y, encoder

