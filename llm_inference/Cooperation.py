# llm_inference/Cooperation.py
# -*- coding: utf-8 -*-
"""
兼容入口：Cooperation 模式。

推荐统一使用：
    python infer_by_index.py --infer-type llm_infer --llm-mode cooperation ...
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from infer_by_index import main  # noqa: E402


if __name__ == "__main__":
    sys.argv = ["infer_by_index.py", "--infer-type", "llm_infer", "--llm-mode", "cooperation"] + sys.argv[1:]
    main()
