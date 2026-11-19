"""通过共享 vLLM 引擎执行一批独立的 Agent 请求。"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator

from .invoke import vllm_invoke
from .io_utils import save_text


AgentRequest = tuple[str, str, dict[str, Any]]


@dataclass
class AgentStage:
    name: str
    requests: list[AgentRequest]
    final: bool = False


@dataclass
class RequestFailure:
    message: str


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

    def generate_stage(
        self, name: str, requests: Iterable[tuple[Path, AgentRequest]],
    ) -> Iterator[str | RequestFailure]:
        """按输入顺序返回各请求的回答，只对发生异常的批次拆分重试。"""
        pending = iter(requests)
        try:
            params = self._sampling_params_factory(
                temperature=0.1, top_p=0.9, max_tokens=self.max_tokens,
            )
        except Exception as exc:
            for _ in pending:
                yield RequestFailure(str(exc))
            return
        while batch := list(islice(pending, self.batch_size)):
            with self._lock if self._lock is not None else nullcontext():
                answers = self._run_batch(name, batch, params)
            yield from answers

    def _run_batch(self, name, batch, params):
        try:
            return self._invoke(name, batch, params)
        except Exception as exc:
            if len(batch) == 1:
                return [RequestFailure(str(exc))]
            middle = len(batch) // 2
            left = self._run_batch(name, batch[:middle], params)
            right = self._run_batch(name, batch[middle:], params)
            return left + right

    def _invoke(self, name, batch, params):
        prompts = [
            instruction + "\n\n输入 JSON\n" + json.dumps(payload, ensure_ascii=False)
            for _, (_, instruction, payload) in batch
        ]
        stage_dirs = [output_dir / path for output_dir, (path, _, _) in batch]
        responses = vllm_invoke(
            self.llm, inputs=prompts, sampling_params=params,
            batch_size=self.batch_size, prompt_safety_tokens=self.prompt_safety_tokens,
            prompt_output_paths=[folder / "prompt.txt" for folder in stage_dirs],
            require_complete_input=True, allow_empty_responses=True, stage_name=name,
        )
        for index, (response, folder) in enumerate(zip(responses, stage_dirs)):
            try:
                save_text(response, folder / "raw_responses.txt")
            except OSError as exc:
                responses[index] = RequestFailure(str(exc))
        return responses

    def generate(
        self, requests: list[AgentRequest], output_dir: Path, *, allow_empty_responses=False,
    ) -> list[str]:
        """保留单个 case 的直接调用接口。"""
        answers = []
        for response in self.generate_stage("agent", ((output_dir, request) for request in requests)):
            if isinstance(response, RequestFailure):
                raise RuntimeError(response.message)
            if not response and not allow_empty_responses:
                raise RuntimeError("vLLM 返回了空响应")
            answers.append(response)
        return answers
