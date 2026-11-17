# 在仓库根目录运行，按需替换 --label-root 后的数据集路径。
# python run_agent_evaluation.py --label-root /home/sbp/deploy/case_pool/anomalydetect_label_0 --import-top-level-sop --output-dir res/agent_evaluation
# 每种模式只跑前 N 个符合条件的 case 时，加 --limit-per-mode N。
# 本地样例的 SOP 位于 JSON 顶层，运行时加上 --import-top-level-sop。
# python run_agent_evaluation.py --label-root llm_interfaces/examples --import-top-level-sop --output-dir res/agent_evaluation_samples

from llm_interfaces.evaluate import main


if __name__ == "__main__":
    raise SystemExit(main())
