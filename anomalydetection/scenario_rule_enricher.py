#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect one of the four supported RCA scenarios and append its SOP and
root-cause candidates to a generated label JSON.

Supported scenarios, in fixed order:

1. BGP邻接状态改变
2. 接口状态DOWN
3. 网络设备掉线
4. 入方向带宽利用率过高

The integrated pipeline imports ``enrich_label_with_scenario_rules``.
This file can also enrich an existing label JSON independently.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


SCENARIO_ORDER = (
    "BGP邻接状态改变",
    "接口状态DOWN",
    "网络设备掉线",
    "入方向带宽利用率过高",
)


BGP_SOP = (
    'Procedure 1:::Decision:::检查对端设备运行状态 resource_status 是否不在 '
    '["Running", ""] 中.:::Yes::Procedure 2::No::Procedure 3\n\n'
    'Procedure 2:::Terminal:::诊断根因为"根因1-对端-未运行"，即对端设备状态已查询，'
    '但不是 running 状态:::\n\n'
    'Procedure 3:::Decision:::检查 remote_device_level 是否包含 "Vlanif".'
    ':::Yes::Procedure 4::No::Procedure 5\n\n'
    'Procedure 4:::Decision:::检查 BGP 会话状态 bgp_state 是否包含 "Established".'
    ':::Yes::Procedure 6::No::Procedure 7\n\n'
    'Procedure 6:::Terminal:::诊断根因为"根因2-对端-服务器闪断"，即对端是服务器，'
    '发生闪断，BGP 状态已恢复:::\n\n'
    'Procedure 7:::Terminal:::诊断根因为"根因3-对端-服务器 BGP 未恢复"，'
    '即对端是服务器，当前 BGP 连接断开:::\n\n'
    'Procedure 5:::Decision:::检查本端接口状态 local_interface_state 是否包含 '
    '"ERROR DOWN".:::Yes::Procedure 8::No::Procedure 9\n\n'
    'Procedure 8:::Terminal:::诊断根因为"根因6-本端-errordown"，即端口或机器故障:::\n\n'
    'Procedure 9:::Decision:::检查本端接口状态 local_interface_state 是否包含 '
    '"Administratively DOWN".:::Yes::Procedure 10::No::Procedure 11\n\n'
    'Procedure 10:::Terminal:::诊断根因为"根因4-本端-主动 shutdown"，'
    '即需根据操作账号及其他信息排查，判断是否为变更或故障:::\n\n'
    'Procedure 11:::Decision:::检查对端接口状态 remote_interface_state 是否包含 '
    '"Administratively DOWN".:::Yes::Procedure 12::No::Procedure 13\n\n'
    'Procedure 12:::Terminal:::诊断根因为"根因5-对端-主动 shutdown"，'
    '即需根据操作账号及其他信息排查，判断是否为变更或故障:::\n\n'
    'Procedure 13:::Decision:::检查闪断次数 flash_times 是否大于等于 3.'
    ':::Yes::Procedure 14::No::Procedure 15\n\n'
    'Procedure 14:::Terminal:::诊断根因为"根因7-本端-频繁闪断"，'
    '即端口或设备故障/传输不稳定/变更引起的频繁闪断:::\n\n'
    'Procedure 15:::Decision:::检查闪断次数 flash_times 是否大于等于 0 且 '
    '本端接口状态包含 "UP".:::Yes::Procedure 16::No::Procedure 17\n\n'
    'Procedure 16:::Decision:::检查 BGP 会话状态 bgp_state 是否包含 "Established".'
    ':::Yes::Procedure 18::No::Procedure 19\n\n'
    'Procedure 18:::Terminal:::诊断根因为"根因8-本端-闪断"，即轻微闪断，属正常波动:::\n\n'
    'Procedure 19:::Terminal:::诊断根因为"根因9-本端-BGP未恢复（有闪断）"，'
    '即接口 up 且有闪断但 BGP 未恢复，需人工排查:::\n\n'
    'Procedure 17:::Decision:::检查本端接口状态 local_interface_state 是否包含 '
    '"DOWN".:::Yes::Procedure 20::No::Procedure 21\n\n'
    'Procedure 20:::Terminal:::诊断根因为"根因10-本对端-接口 down 异常"，'
    '即接口 down 且非 flap，需结合日志与规则或人工判断:::\n\n'
    'Procedure 21:::Decision:::检查闪断次数 flash_times 是否等于 0 且 '
    '本端接口状态包含 "UP".:::Yes::Procedure 22::No::Procedure 23\n\n'
    'Procedure 22:::Terminal:::诊断根因为"根因11-本端-BGP未恢复（无闪断）"，'
    '即接口 up 且无 flap，但 BGP 状态异常，需人工排查:::\n\n'
    'Procedure 23:::Terminal:::诊断根因为"根因12-未知-需人工复核"，'
    '即以上规则都未能匹配，保留原始特征供人工复核:::'
)


INTERFACE_DOWN_SOP = (
    'Procedure 1:::Decision:::检查本端设备类型和分区device level，是否为 L1 设备'
    '或本端使用状态正常.:::Yes::Procedure 2::No::Procedure 3\n\n'
    'Procedure 2:::Terminal:::诊断根因为"根因1-对端-服务器状态异常":::\n\n'
    'Procedure 3:::Decision:::检查本端接口状态 local_interface_state 是否 ERROR DOWN.'
    ':::Yes::Procedure 4::No::Procedure 5\n\n'
    'Procedure 4:::Terminal:::诊断根因为"根因2-本端-接口errordown".:::\n\n'
    'Procedure 5:::Decision:::检查本端接口 local_interface_state 是否 '
    'Administratively DOWN.:::Yes::Procedure 6::No::Procedure 7\n\n'
    'Procedure 6:::Terminal:::诊断根因为"根因3-本端-接口主动关闭".:::\n\n'
    'Procedure 7:::Decision:::检查对端接口 remote_interface_state 是否 '
    'Administratively DOWN.:::Yes::Procedure 8::No::Procedure 9\n\n'
    'Procedure 8:::Terminal:::诊断根因为"根因4-对端-接口主动关闭".:::\n\n'
    'Procedure 9:::Decision:::检查接口闪断次数是否大于等于 3.'
    ':::Yes::Procedure 10::No::Procedure 11\n\n'
    'Procedure 10:::Decision:::是否为 monitor-link 场景.'
    ':::Yes::Procedure 12::No::Procedure 13\n\n'
    'Procedure 12:::Terminal:::诊断根因为"根因6-本端-monitor-link场景,频繁闪断".:::\n\n'
    'Procedure 13:::Terminal:::诊断根因为"根因5-本端-接口闪断频繁".:::\n\n'
    'Procedure 11:::Decision:::检查闪断次数大于零但小于三，接口状态是否 up.'
    ':::Yes::Procedure 15::No::Procedure 14\n\n'
    'Procedure 14:::Terminal:::诊断根因为"根因7-本端-接口掉线且发生闪断(非flap)".:::\n\n'
    'Procedure 15:::Terminal:::诊断根因为"根因8-本端-发生闪断,但接口up".:::'
)


DEVICE_OFFLINE_SOP = (
    'Procedure 1:::Decision:::检查设备分区device level 是否为 L1 设备.'
    ':::Yes::Procedure 2::No::Procedure 4\n\n'
    'Procedure 2:::Decision:::检查服务器 resource_status是否正常运行.'
    ':::Yes::Procedure 3::No::Procedure 4\n\n'
    'Procedure 3:::Terminal:::诊断根因为"根因1-本端-服务器未运行".:::\n\n'
    'Procedure 4:::Decision:::检查设备信息sys_running_time,是否掉线'
    '(sys_running_time是否存在).:::Yes::Procedure 5::No::Procedure 6\n\n'
    'Procedure 5:::Decision:::检查设备信息告警首次发生时间alarm_time与'
    'sys_running_time的差是否小于360000ms（判断设备是否重启）.'
    ':::Yes::Procedure 6::No::Procedure 7\n\n'
    'Procedure 6:::Decision:::检查anomaly_logs中是否包含重启命令.'
    ':::Yes::Procedure 8::No::Procedure 9\n\n'
    'Procedure 7:::Terminal:::诊断根因为"根因2-本端-设备未掉线".:::\n\n'
    'Procedure 8:::Terminal:::诊断根因为"根因3-本端-设备重启（变更引起）".:::\n\n'
    'Procedure 9:::Decision:::检查告警时间前后1分钟接口丢包率 offline_loss_rate '
    '是否达到 1（100%）.:::Yes::Procedure 10::No::Procedure 11\n\n'
    'Procedure 10:::Terminal:::诊断根因为"根因4-本端-设备丢包率=100%".:::\n\n'
    'Procedure 11:::Terminal:::诊断根因为"根因5-本端-设备丢包率<100%".:::'
)


TRAFFIC_HIGH_SOP = (
    'Procedure 1:::Decision:::检查 local_ifname 是否包含"Vlanif".'
    ':::Yes::Procedure 2::No::Procedure 3\n\n'
    'Procedure 2:::Terminal:::诊断根因为"根因1-对端-服务器"，即对端设备为服务器，'
    '建议关单并联系server或升级大区处理:::\n\n'
    'Procedure 3:::Decision:::检查 owner 是否包含 "DCNOM(数据中心网络运维)".'
    ':::Yes::Procedure 4::No::Procedure 5\n\n'
    'Procedure 5:::Terminal:::诊断根因为"根因1-对端-服务器"，对端设备不是DCN所属组:::\n\n'
    'Procedure 4:::Decision:::检查在alarm_time时间点，语义标签中anomaly_kpi下的'
    'traffic_in的指标值是否超过阈值90.:::Yes::Procedure 6::No::Procedure 8\n\n'
    'Procedure 6:::Decision:::检查在alarm_time时间点，语义标签中anomaly_kpi下的'
    'traffic_in的指标值是否超过阈值110.:::Yes::Procedure 7::No::Procedure 9\n\n'
    'Procedure 7:::Terminal:::诊断根因为"根因5-本端-设备流量采集故障"，'
    '即接口带宽利用率已超110%，升级决策，确认设备是否存在采集故障.:::\n\n'
    'Procedure 9:::Terminal:::诊断根因为"根因2-本端-接口流量异常"，'
    '即接口带宽利用率已超90%，存在拥塞风险.:::\n\n'
    'Procedure 8:::Decision:::检查在alarm_time时间点后15分钟，语义标签中'
    'anomaly_kpi下的traffic_in的指标值是否降到80以下.'
    ':::Yes::Procedure 10::No::Procedure 11\n\n'
    'Procedure 10:::Terminal:::诊断根因为"根因3-本端-接口流量恢复正常"，'
    '即接口流量已降到80%以下，给大区留言.:::\n\n'
    'Procedure 11:::Terminal:::诊断根因为"根因4-本端-接口流量持续过高"，'
    '即接口带宽利用率持续高于80或高于90:::'
)


SCENARIO_RULES: dict[str, dict[str, Any]] = {
    "BGP邻接状态改变": {
        "sop": [BGP_SOP],
        "root_cause_candidates": [
            {
                "category": "根因1-对端-未运行",
                "description": "对端设备状态已查询，但不是 running 状态",
            },
            {
                "category": "根因2-对端-服务器闪断",
                "description": "对端是服务器，发生闪断，BGP 状态已恢复",
            },
            {
                "category": "根因3-对端-服务器 BGP 未恢复",
                "description": "对端是服务器，当前 BGP 连接断开",
            },
            {
                "category": "根因4-本端-主动 shutdown",
                "description": "需根据操作账号及其他信息排查，判断是否为变更或故障",
            },
            {
                "category": "根因5-对端-主动 shutdown",
                "description": "需根据操作账号及其他信息排查，判断是否为变更或故障",
            },
            {
                "category": "根因6-本端-errordown",
                "description": "端口或机器故障",
            },
            {
                "category": "根因7-本端-频繁闪断",
                "description": "端口或设备故障/传输不稳定/变更引起的频繁闪断",
            },
            {
                "category": "根因8-本端-闪断",
                "description": "轻微闪断，属正常波动",
            },
            {
                "category": "根因9-本端-BGP未恢复（有闪断）",
                "description": "接口 up 且有闪断但 BGP 未恢复，需人工排查",
            },
            {
                "category": "根因10-本对端-接口 down 异常",
                "description": "接口 down 且非 flap，需结合日志与规则或人工判断",
            },
            {
                "category": "根因11-本端-BGP未恢复（无闪断）",
                "description": "接口 up 且无 flap，但 BGP 状态异常，需人工排查",
            },
        ],
    },
    "接口状态DOWN": {
        "sop": INTERFACE_DOWN_SOP,
        "root_cause_candidates": [
            {
                "category": "根因1-对端-服务器状态异常",
                "description": "远端服务器未使用或故障，请检查服务器状态。",
            },
            {
                "category": "根因2-本端-接口errordown",
                "description": "端口或机器故障导致接口掉线，需检查硬件状态。",
            },
            {
                "category": "根因3-本端-接口主动关闭",
                "description": "接口主动关闭，需要根据操作账号及日志信息排查是否为变更或故障。",
            },
            {
                "category": "根因4-对端-接口主动关闭",
                "description": "接口主动关闭，需结合对端设备的操作日志排查原因。",
            },
            {
                "category": "根因5-本端-接口闪断频繁",
                "description": "端口或设备故障、传输不稳定或变更引起的频繁闪断。",
            },
            {
                "category": "根因6-本端-monitor-link场景,频繁闪断",
                "description": "物理接口利用率小于90%，可能导致频繁闪断，需进行优化。",
            },
            {
                "category": "根因7-本端-接口掉线且发生闪断(非flap)",
                "description": "接口 down 且非 flap,需结合日志和规则或人工判断是否为故障。",
            },
            {
                "category": "根因8-本端-发生闪断,但接口up",
                "description": "接口 up，正常关单，无需进一步处理。",
            },
        ],
    },
    "网络设备掉线": {
        "sop": DEVICE_OFFLINE_SOP,
        "root_cause_candidates": [
            {
                "side": "本端",
                "category": "根因1-本端-服务器未运行",
                "reason": "服务器故障，所属服务器状态异常。",
            },
            {
                "side": "本端",
                "category": "根因2-本端-设备未掉线",
                "reason": "设备状态正常，无掉线问题。",
            },
            {
                "side": "本端",
                "category": "根因3-本端-设备重启（变更引起）",
                "reason": "由于变更操作，设备发生重启。需要确认用户是否进行了操作。",
            },
            {
                "side": "本端",
                "category": "根因4-本端-设备丢包率=100%",
                "reason": "存在变更、接口状态为down或设备掉线。",
            },
            {
                "side": "本端",
                "category": "根因5-本端-设备丢包率<100%",
                "reason": "不确定根因，需查询日志人工确认问题。",
            },
        ],
    },
    "入方向带宽利用率过高": {
        "sop": TRAFFIC_HIGH_SOP,
        "root_cause_candidates": [
            {
                "category": "根因1-对端-服务器",
                "description": "对端设备为服务器，建议关单并联系server或升级大区处理",
            },
            {
                "category": "根因2-本端-接口流量异常",
                "description": "接口带宽利用率已超90%，存在拥塞风险",
            },
            {
                "category": "根因3-本端-接口流量恢复正常",
                "description": "接口流量已降到80%以下，给大区留言",
            },
            {
                "category": "根因4-本端-接口流量持续过高",
                "description": "接口带宽利用率持续高于80或高于90",
            },
            {
                "category": "根因5-本端-设备流量采集故障",
                "description": "接口带宽利用率已超110%，升级决策，确认设备是否存在采集故障",
            },
        ],
    },
}


SCENARIO_ALIASES = {
    "BGP邻接状态改变": (
        "BGP邻接状态改变",
        "BGP邻接状态变化",
        "BGP邻接状态",
        "BGP邻接",
        "BGP",
    ),
    "接口状态DOWN": (
        "接口状态DOWN",
        "接口状态 down",
        "接口DOWN",
        "接口 down",
        "interface down",
    ),
    "网络设备掉线": (
        "网络设备掉线",
        "网络设备离线",
        "设备掉线",
        "设备离线",
    ),
    "入方向带宽利用率过高": (
        "入方向带宽利用率过高",
        "入方向带宽利用率",
        "入方向带宽过高",
        "带宽利用率过高",
        "入方向流量过高",
    ),
}


SCENARIO_FIELD_NAMES = {
    "scenario",
    "scenario_name",
    "scenarioname",
    "case_type",
    "casetype",
    "case_name",
    "casename",
    "alarm_name",
    "alarmname",
    "alarm_type",
    "alarmtype",
    "fault_name",
    "faultname",
    "fault_type",
    "faulttype",
    "title",
    "场景",
    "场景名称",
    "告警名称",
    "告警类型",
    "故障类型",
}


def normalize_text(value: Any) -> str:
    """Normalize strings for robust scenario matching."""
    text = str(value or "").casefold()
    return re.sub(r"[\s_\-—–:：/\\()\[\]{}（）【】]+", "", text)


def iter_scenario_field_values(value: Any) -> Iterable[str]:
    """
    Recursively collect only explicit scenario/alarm/fault fields.

    Arbitrary log messages are intentionally ignored to prevent a BGP log
    inside another scenario from causing a false scenario classification.
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = normalize_text(key)
            if (
                key in SCENARIO_FIELD_NAMES
                or normalized_key in {
                    normalize_text(item) for item in SCENARIO_FIELD_NAMES
                }
            ):
                if isinstance(child, (str, int, float)):
                    yield str(child)

            if isinstance(child, (Mapping, list)):
                yield from iter_scenario_field_values(child)

    elif isinstance(value, list):
        for child in value:
            yield from iter_scenario_field_values(child)


def scenario_sources(
    case_path: Path | None,
    case_document: Mapping[str, Any] | None,
) -> list[str]:
    sources: list[str] = []

    if case_path is not None:
        resolved = Path(case_path)
        sources.extend(
            [
                resolved.name,
                resolved.stem,
                resolved.parent.name,
                resolved.parent.parent.name,
            ]
        )

    if case_document is not None:
        sources.extend(iter_scenario_field_values(case_document))

    return [source for source in sources if str(source).strip()]


def detect_scenario(
    *,
    case_path: Path | None = None,
    case_document: Mapping[str, Any] | None = None,
) -> str:
    """
    Detect the scenario using path names first and explicit case fields second.
    """
    sources = scenario_sources(case_path, case_document)
    matches: list[tuple[int, int, str, str]] = []

    for source_index, source in enumerate(sources):
        normalized_source = normalize_text(source)
        for scenario_index, scenario in enumerate(SCENARIO_ORDER):
            for alias in SCENARIO_ALIASES[scenario]:
                normalized_alias = normalize_text(alias)
                if normalized_alias and normalized_alias in normalized_source:
                    # Prefer earlier sources (path name/parent), then longer aliases,
                    # then the fixed scenario order supplied by the user.
                    score = len(normalized_alias)
                    matches.append(
                        (
                            -source_index,
                            score,
                            -scenario_index,
                            scenario,
                        )
                    )

    if not matches:
        source_preview = ", ".join(repr(item) for item in sources[:12])
        raise ValueError(
            "无法识别 case 场景。仅支持："
            + "、".join(SCENARIO_ORDER)
            + f"。已检查内容：{source_preview}"
        )

    matches.sort(reverse=True)
    return matches[0][3]


def get_scenario_rules(scenario: str) -> dict[str, Any]:
    """Return an independent copy so callers cannot mutate global templates."""
    if scenario not in SCENARIO_RULES:
        raise KeyError(f"Unsupported scenario: {scenario}")
    return copy.deepcopy(SCENARIO_RULES[scenario])


def enrich_label_with_scenario_rules(
    label_document: MutableMapping[str, Any],
    *,
    case_path: Path | None = None,
    case_document: Mapping[str, Any] | None = None,
) -> str:
    """
    Append ``sop`` and ``root_cause_candidates`` at the JSON root.

    Returns the detected canonical scenario name.
    """
    scenario = detect_scenario(
        case_path=case_path,
        case_document=case_document,
    )
    rules = get_scenario_rules(scenario)

    # Both keys are written at the JSON root, alongside semantic_labels
    # and alarm_time.
    label_document["sop"] = rules["sop"]
    label_document["root_cause_candidates"] = rules[
        "root_cause_candidates"
    ]
    return scenario


def atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the RCA scenario and append SOP/root-cause candidates "
            "to an existing label JSON."
        )
    )
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--label-file", required=True, type=Path)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Default: overwrite --label-file atomically.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    with args.case.open("r", encoding="utf-8-sig") as file:
        case_document = json.load(file)
    if not isinstance(case_document, Mapping):
        raise TypeError("Case JSON root must be an object.")

    with args.label_file.open("r", encoding="utf-8-sig") as file:
        label_document = json.load(file)
    if not isinstance(label_document, MutableMapping):
        raise TypeError("Label JSON root must be an object.")

    scenario = enrich_label_with_scenario_rules(
        label_document,
        case_path=args.case,
        case_document=case_document,
    )

    output_file = args.output_file or args.label_file
    atomic_write_json(output_file, label_document)

    print(f"scenario={scenario}")
    print(
        "root_cause_candidate_count="
        f"{len(label_document['root_cause_candidates'])}"
    )
    print(f"output_file={output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
