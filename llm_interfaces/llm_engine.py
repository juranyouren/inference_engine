"""进程内共享的 vLLM 单例。"""

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM


class LLMEngine:
    _instance = None

    def __init__(self, model_path, gpu_memory_utilization=0.9, max_model_len=16384):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=len(os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(',')),
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len
        )
        print(f"LLM 已加载：{model_path}")
        # self.llm = "test llm"
        # print("test none init")

    @classmethod
    def get_instance(cls, model_path=None, gpu_memory_utilization=0.9, max_model_len=16384):
        """仅在第一次调用时加载 LLM"""
        if cls._instance is None:
            assert model_path is not None, "首次调用需要 model_path"
            cls._instance = LLMEngine(model_path, gpu_memory_utilization, max_model_len)
        return cls._instance
