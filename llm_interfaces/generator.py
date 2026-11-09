"""通过共享 vLLM 引擎执行一批独立的 Agent 请求。"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .invoke import vllm_invoke
from .io_utils import save_text


AgentRequest = tuple[str, str, dict[str, Any]]


class RCAGenerator:
    def __init__(
        self, engine, *, batch_size=8, max_tokens=4096, prompt_safety_tokens=512,
        sampling_params_factory=None,
    ):
        if engine is None or not hasattr(engine, "llm"):
            raise TypeError("engine 必须包含共享的 llm")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if sampling_params_factory is None:
            from vllm import SamplingParams

            sampling_params_factory = SamplingParams

        self.llm = engine.llm
        self._lock = getattr(engine, "invoke_lock", None)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.prompt_safety_tokens = prompt_safety_tokens
        self._sampling_params_factory = sampling_params_factory

    def generate(self, requests: list[AgentRequest], output_dir: Path) -> list[str]:
        prompts = [
            instruction + "\n\n输入 JSON\n" + json.dumps(payload, ensure_ascii=False)
            for _stage, instruction, payload in requests
        ]
        stage_dirs = [output_dir / stage for stage, _instruction, _payload in requests]
        params = self._sampling_params_factory(
            temperature=0.1, top_p=0.9, max_tokens=self.max_tokens,
        )
        with self._lock if self._lock is not None else nullcontext():
            responses = vllm_invoke(
                self.llm, inputs=prompts, sampling_params=params,
                batch_size=self.batch_size, prompt_safety_tokens=self.prompt_safety_tokens,
                prompt_output_paths=[folder / "prompt.txt" for folder in stage_dirs],
                require_complete_input=True,
            )
        for response, folder in zip(responses, stage_dirs):
            save_text(response, folder / "raw_responses.txt")
        return responses
