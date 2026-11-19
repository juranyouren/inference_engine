"""延时八卡评测的 CPU 预检，成功后写入四个类别名。"""

import argparse
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryFile

from inference.common import IndexedCaseReader, discover_label_index_map, load_json
from inference.selection import bind_training_selection, build_selection_from_selector, training_feature_catalog
from inference.tree import extract_features_for_cases, train_tree_model
from llm_interfaces.cooperation_telemetry import normalize_telemetry, semantic_labels_template
from llm_interfaces.data_adapter import normalize_case, without_sop
from run_tree_agent_evaluation import _split_scenario, legacy_tree_template


def categories(root):
    return {d.name: indices for d in sorted(root.iterdir()) if d.is_dir()
            and (indices := discover_label_index_map(str(d), d.name))}


def check(args):
    coop = categories(args.coop_root)
    tree = categories(args.tree_root)
    if len(coop) != 4 or coop.keys() != tree.keys():
        raise ValueError(f"两套数据应各有四个同名类别。Cooperation={list(coop)}, Tree={list(tree)}")
    base = load_json(str(args.coop_template))
    for role in ("log", "time_series", "metadata"):
        if not isinstance(base.get(role), dict) or not base[role]:
            raise ValueError(f"{args.coop_template} 缺少 {role} 字段目录")

    for name in coop:
        print(f"检查 {name}: {args.coop_root / name}，{args.tree_root / name}", flush=True)
        _, targets = _split_scenario({"name": name, "label_map": coop[name]}, 50, 150)
        for idx in targets:
            path = coop[name][idx]
            try:
                case = normalize_case(without_sop(load_json(path)), input_index=idx, mode="cooperation", alarm_type=name)
                semantic_labels_template(normalize_telemetry(case.semantic_labels), base)
            except Exception as exc:
                raise ValueError(f"Cooperation 输入异常 {path}: {exc}") from exc
        template = legacy_tree_template(name, args.tree_templates)
        print(f"Tree Selector 模板 {template['_template_path']}", flush=True)
        train, targets = _split_scenario({"name": name, "label_map": tree[name]}, 50, 150)
        reader = IndexedCaseReader(str(args.tree_root / name), name)
        cases = reader.load_cases(train + targets)
        # 检查三轮实际的历史窗口，用训练集可用列试跑 CPU 拟合及预测。
        for end in (50, 100, 150):
            catalog = training_feature_catalog(cases[:end])
            selection = bind_training_selection(build_selection_from_selector({"log": [], "kpi": {}}), catalog)
            clf, _, _ = train_tree_model(cases[:end], selection)
            _, values, _ = extract_features_for_cases(cases[end:end + 50], selection)
            clf.predict(values)

        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        with TemporaryFile(dir=folder) as probe:
            probe.write(b"ok")

    for module in ("torch", "torch_npu", "vllm", "vllm_ascend", "pandas"):
        if find_spec(module) is None:
            raise ImportError(f"当前 Python 缺少 {module}")
    load_json(str(args.model / "config.json"))
    indices = list(args.model.glob("*.index.json"))
    weights = set()
    for index in indices:
        weights.update(args.model / file for file in load_json(str(index)).get("weight_map", {}).values())
    if not weights:
        weights.update(args.model.glob("*.safetensors"))
        weights.update(args.model.glob("*.bin"))
    if not weights or any(not p.is_file() or not p.stat().st_size for p in weights):
        raise ValueError(f"模型权重缺失或为空: {args.model}")
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(str(args.model), local_files_only=True).encode("RCA")
    args.manifest.write_bytes(b"".join(name.encode("utf-8") + b"\0" for name in coop))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("coop-root", "tree-root", "coop-template", "tree-templates", "model", "output", "manifest"):
        parser.add_argument("--" + option, type=Path, required=True)
    check(parser.parse_args())
