# -*- coding: utf-8 -*-
import os

# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = "/home/sbp/deployment/inference_engine"

RULE_INFERENCER_DIR = os.path.join(PROJECT_ROOT, "rule_inferencer")
LLM_INFERENCE_DIR = os.path.join(PROJECT_ROOT, "llm_inference")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")


# ============================================================
# 模型 / NPU
# ============================================================

MODEL_PATH = "/usr/share/large_language_models/DeepSeek-R1-Distill-Qwen-32B"
ASCEND_RT_VISIBLE_DEVICES = "6,7"
BATCH_SIZE = 16


# ============================================================
# 数据目录
# ============================================================

CASE_POOL_ROOT = "/home/sbp/deployment/case_pool"

# LLM 只读取这个根目录下的类别子目录
AGENTDIGEST_LABEL_ROOT = os.path.join(CASE_POOL_ROOT, "AgentDigest_label")

# Tree 只读取这个根目录下的类别子目录
ANOMALYDETECT_LABEL_ROOT = os.path.join(CASE_POOL_ROOT, "anomalydetect_label")

# 输出目录
BASE_RES_DIR = CASE_POOL_ROOT
PREDICT_RES_DIR = os.path.join(BASE_RES_DIR, "predict_result")
TREE_RULE_DIR = os.path.join(BASE_RES_DIR, "tree_rule")
SUMMARY_DIR = os.path.join(BASE_RES_DIR, "summary")


# ============================================================
# 增量处理状态
# ============================================================

STATE_DIR = os.path.join(PREDICT_RES_DIR, "state")
INFER_STATE_FILE = os.path.join(STATE_DIR, "infer_state.json")

# 首次运行时，last_index 的初值。
# -1 表示首次运行处理目录中的全部已有数据。
# 如果历史 0~59 不需要处理，可改成 59。
STATE_INITIAL_LAST_INDEX = -1

# 每个类别每次最多处理多少条新增数据；None 表示不限制。
MAX_NEW_CASES = None


# ============================================================
# 默认运行模式
# ============================================================

# 默认两种推理都执行：tree_infer + llm_infer
DEFAULT_INFER_TYPE = "all"

# LLM 默认按 semantic_labels["sop"] 自动分流：
# sop 非空 -> competition
# sop 为空 -> cooperation
DEFAULT_LLM_MODE = "auto"

# LLM 使用完整推理流程。
LLM_META_ONLY = False

# Cooperation 最大 Meta/Reasoner 迭代轮数。
LLM_COOPERATION_MAX_ROUNDS = 5


# ============================================================
# Tree 参数
# ============================================================

TRAIN_N = 50
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 10
RANDOM_STATE = 42

# 可选：file / selector / selector_refiner
# 推荐生产环境先用 file；需要在线生成 selection 时改成 selector_refiner。
SELECTION_SOURCE = "file"
SELECTION_FILE_NAME = "selection.json"

# Selector / Refiner
TEMPLATE_DIR = "/home/sbp/huangzeshun/RCA_Units/template"
SELECTOR_EX_NUM = 1
REFINER_EX_NUM = 1
REFINER_ROUNDS = 1


# ============================================================
# 文件匹配模式
# ============================================================

CASE_FILE_PATTERNS = [
    "{alarm_type}_{idx}_case_*.json",
    "{idx}_case_*.json",
    "case_{idx}.json",
    "case-{idx}.json",
    "{idx}.json",
]

LABEL_FILE_PATTERNS = [
    "{alarm_type}_{idx}_label_*.json",
    "{idx}_label_*.json",
    "label_{idx}.json",
    "{idx}.json",
]
