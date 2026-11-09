"""可直接由其他 Python 模块调用的 RCA 大模型推理接口。

模型由 LLMEngine.get_instance() 首次调用时加载，所有推理类共享该实例。
"""

__all__ = ["LLMEngine", "CompetitionInference", "CooperationInference"]


def __getattr__(name):
    if name == "LLMEngine":
        from .llm_engine import LLMEngine
        return LLMEngine
    if name == "CompetitionInference":
        from .competition import CompetitionInference
        return CompetitionInference
    if name == "CooperationInference":
        from .cooperation import CooperationInference
        return CooperationInference
    raise AttributeError(name)
