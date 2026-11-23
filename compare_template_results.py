"""比较同一种推理模式在两套模板下的结果（A 为参照，B 为对照）。

用法：python compare_template_results.py 结果目录A 结果目录B --mode cooperation --output res/compare_coop
模式：cooperation、competition、tree。支持评测总目录、单类别目录、单个结果 JSON。
打开输出目录的 differences.csv 浏览差异；同一行的 detail 指向两侧完整结果。
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


MODES = {"tree", "cooperation", "competition"}


def load_cases(root, mode):
    files = [root] if root.is_file() else root.rglob("*.json")
    cases = {}
    for path in sorted(files):
        if path.name not in {"predictions.json", "results.json", "result.json"} and path != root:
            continue
        if any(part in MODES - {mode} for part in path.relative_to(root).parts[:-1]):
            continue
        if mode == "tree" and path.name in {"result.json", "results.json"}:
            continue
        if mode != "tree" and path.name == "predictions.json":
            continue
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = obj.get("results", [obj])
        if isinstance(rows, dict):
            rows = [row for group in rows.values() for row in group]
        for row in rows:
            if mode == "tree":
                if "pred_rc" not in row:
                    continue
                category = row["alarm_type"]
            else:
                if row.get("mode") != mode:
                    continue
                category = row["incident_type"]
            filename = re.split(r"[/\\]", row.get("label_file_path") or "")[-1]
            match = re.search(r"(?:^|_)(\d+)_label_(.+)\.json$", filename)
            idx = int(match[1]) if match else int(row["case_idx"])
            key = (category, match[2] if match else "", idx)
            if key in cases:
                old, old_path = cases[key]
                if old != row:
                    raise ValueError(f"同一 case 有冲突记录 {key}: {old_path}, {path}。请指定单次运行目录")
                if path.name != "result.json":
                    continue
            cases[key] = row, path
    if not cases:
        raise ValueError(f"{root} 未找到 {mode} 的结果")
    return cases


def ranking(row):
    return row.get("pred_rc", row.get("predicted_list_v1", row.get("predicted_list", []))) or []


def evidence(row, mode):
    if mode == "cooperation":
        return {op["agent"]: op for op in row.get("opinion_pool", [])}
    if mode == "competition":
        return {d["root_cause"]: d for d in row.get("diagnoses", [])}
    return {"决策路径": row.get("cot"), "根因分数": row.get("pred_scores")}


def compare(left, right, mode, output, include_reasoning=False):
    a, b = load_cases(left, mode), load_cases(right, mode)
    # 两版可能重新编号；标签文件尾部 ID 重复时才需要 index 消歧。
    counts = Counter(key[:2] for key in a) | Counter(key[:2] for key in b)

    def align(cases):
        return {(cat, ident, idx if not ident or counts[cat, ident] > 1 else None): (idx, *value)
                for (cat, ident, idx), value in cases.items()}

    a, b = align(a), align(b)
    output.mkdir(parents=True, exist_ok=True)
    details = output / "details"
    details.mkdir(exist_ok=True)
    totals = defaultdict(Counter)
    fields = ["类别", "标签ID", "索引A", "索引B", "差异", "首选A", "首选B", "真实根因A", "真实根因B",
              "排名变化", "推理差异项", "detail"]
    with (output / "differences.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for number, key in enumerate(sorted(a.keys() | b.keys()), 1):
            cat, ident, _ = key
            ia, ra, pa = a.get(key, ("", {}, None))
            ib, rb, pb = b.get(key, ("", {}, None))
            ar, br = ranking(ra), ranking(rb)
            at, bt = ar[:1], br[:1]
            ea, eb = evidence(ra, mode), evidence(rb, mode)
            changes = {k: {"A": ea.get(k), "B": eb.get(k)} for k in sorted(ea.keys() | eb.keys())
                       if ea.get(k) != eb.get(k)}
            if not ra or not rb:
                kind = "仅B存在" if rb else "仅A存在"
            elif (ra.get("status"), ra.get("error")) != (rb.get("status"), rb.get("error")):
                kind = "运行状态或错误变化"
            elif at != bt:
                kind = "首选根因变化"
            elif ar != br:
                kind = "后续排序变化"
            elif ra.get("groundtruth") != rb.get("groundtruth"):
                kind = "真实标签变化"
            elif include_reasoning and changes:
                kind = "排序相同、推理记录变化"
            else:
                kind = "结果相同"
            totals[cat][kind] += 1
            if kind == "结果相同":
                continue
            detail = details / f"{number:06d}.json"
            payload = {"mode": mode, "category": cat, "label_id": ident, "difference": kind,
                       "evidence_changes": changes,
                       "A": {"source": str(pa) if pa else None, "result": ra or None},
                       "B": {"source": str(pb) if pb else None, "result": rb or None}}
            detail.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            shifts = []
            for cause in dict.fromkeys(ar + br):
                x = ar.index(cause) + 1 if cause in ar else "缺失"
                y = br.index(cause) + 1 if cause in br else "缺失"
                if x != y:
                    shifts.append(f"{cause} {x}→{y}")
            writer.writerow([cat, ident, ia, ib, kind, *[next(iter(t), "") for t in (at, bt)],
                             ra.get("groundtruth"), rb.get("groundtruth"), " | ".join(shifts),
                             " | ".join(changes), detail.relative_to(output).as_posix()])
    summary = {"mode": mode, "A": str(left), "B": str(right), "count_A": len(a), "count_B": len(b),
               "by_category": totals}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"A {len(a)} 条，B {len(b)} 条。差异清单：{output / 'differences.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-reasoning", action="store_true", help="纳入排序相同但专家意见、候选诊断或树路径/分数变化的 case")
    args = parser.parse_args()
    compare(args.left.resolve(), args.right.resolve(), args.mode,
            (args.output or Path("res") / f"compare_{args.mode}").resolve(), args.include_reasoning)
