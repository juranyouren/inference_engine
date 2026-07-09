# rule_inferencer/tree.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from rule_inferencer.data_process_v3 import extract_all_features


def extract_features_for_cases(cases: List[Dict[str, Any]], selection: Dict[str, Any]):
    X = []
    feature_names = None
    f_list = []
    for case_data in cases:
        fnames, fvalues, f_dict = extract_all_features(case_data["semantic_labels"], selection)
        if feature_names is None:
            feature_names = fnames
        elif fnames != feature_names:
            raise ValueError("不同 case 抽取出的 feature_names 不一致")
        X.append(fvalues)
        f_list.append(f_dict)
    return feature_names, X, f_list


def extract_root_cause_labels(case_list):
    raw_labels = [case_data["root_cause"] for case_data in case_list]
    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_labels)
    return y, encoder


def train_decision_tree(X, y, max_depth=None, random_state=42, min_samples_leaf=10):
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    clf.fit(np.array(X), np.array(y))
    return clf


def save_tree_rules_to_txt(best_clf, feature_names, label_encoder, scenario, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scenario}.txt")
    tree_rules = export_text(best_clf, feature_names=list(feature_names))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tree_rules)
        f.write("\n\nClass mapping:\n")
        for idx, cls in enumerate(label_encoder.classes_):
            f.write(f"class {idx} -> {cls}\n")
    return output_path
