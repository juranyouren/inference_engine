"""Tree 的训练字段目录与特征提取共用的字段映射。"""

import json


KPI_FIELDS = {
    "traffic_in": ("traffic_in",),
    "traffic_out": ("traffic_out",),
    "drop_packet_rate": ("drop_packet_rate",),
    "error_packet_rate": ("error_packet_rate",),
    "offline_loss_rate": ("offline_loss_rate",),
    "cpu_utilization": ("cpu_utilization", "cpu_usage"),
    "memory_utilization": ("memory_utilization", "memory_usage"),
    "temperature": ("temperature", "device_temperature"),
}
SYSLOG_GROUPS = ("syslog_brief", "syslog_message")


def telemetry_dict(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def kpi_segments(kpis, name):
    for field in KPI_FIELDS[name]:
        segments = telemetry_dict(kpis.get(field)).get("abnormal_timing_segments")
        if isinstance(segments, list) and segments:
            return segments
    return []


EXCEPTION_EVENTS = (
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
)


FLASH_EVENTS = (
    "linkDown_active(l)",
    "linkDown_clear(l)",
    "hwPortDown(l)",
    "hwPortUp(l)",
    "hwBgpBackwardTransition_active(l)",
    "hwBgpBackwardTransition_clear(l)",
    "bgpBackwardTransition_active(l)",
    "bgpBackwardTransition_clear(l)",
    "PEER_STATE_CHG(l)",
)
