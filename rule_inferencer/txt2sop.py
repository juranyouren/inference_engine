# rule_inferencer/txt2sop.py
# -*- coding: utf-8 -*-
"""兼容旧代码的 txt2sop 接口占位。"""


def load_tree_and_class_mapping_from_txt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = content.split("\n\nClass mapping:\n", 1)
    tree_text = parts[0]
    class_mapping_text = parts[1] if len(parts) > 1 else ""
    return tree_text, class_mapping_text


def tree_txt_to_python_function(tree_text: str, class_mapping_text: str, fn_name: str = "predict_root_cause") -> str:
    # 简单兜底：不尝试把 tree 文本转换为完整 SOP，仅返回空预测函数。
    return f'''def {fn_name}(features, rc=None):\n    return {{"cot": [], "rc": rc, "pred_rc": [], "pred_top1_rc": None, "features": features}}\n'''
