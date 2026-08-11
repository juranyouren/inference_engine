"""可直接由其他 Python 模块调用的 RCA 大模型推理接口。

导入本包不会加载模型。只有实例化 CompetitionInference 或
CooperationInference 时，所选择的类才会把模型加载进内存。
"""

from .competition import CompetitionInference
from .cooperation import CooperationInference

__all__ = ["CompetitionInference", "CooperationInference"]
