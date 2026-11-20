"""Lightweight GPU autotuner with TileLang-compatible @autotune decorator."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .compiler import CompiledKernel
from .compiler import compile as compile_kernel
from .ir import CompileError
from .language import PrimFunc


@dataclass(frozen=True)
class TuneResult:
    config: dict
    kernel: CompiledKernel
    latency_ms: float


class AutoTuner:
    """Benchmarks candidate tile/stage configurations on CUDA and caches the fastest kernel."""

    def __init__(
        self,
        fn: Callable,
        configs: Iterable[dict] | Callable[[], Iterable[dict]],
        *,
        warmup: int = 5,
        rep: int = 20,
        target: str = "auto",
        out_idx=None,
    ):
        self.fn = fn
        self._configs = configs
        self.warmup = warmup
        self.rep = rep
        self.target = target
        self.out_idx = out_idx
        self._cache: dict[tuple, TuneResult] = {}
        self.best_config: dict | None = None
        self.best_latency_ms: float | None = None

    def _resolve_configs(self) -> list[dict]:
        raw = self._configs() if callable(self._configs) else self._configs
        configs = [dict(c) for c in raw]
        if not configs:
            raise ValueError("autotune requires at least one configuration")
        return configs

    def _build_candidate(self, args: tuple, kwargs: dict, cfg: dict) -> CompiledKernel:
        merged = {**kwargs, **cfg}
        result = self.fn(*args, **merged)
        if isinstance(result, CompiledKernel):
            return result
        if isinstance(result, PrimFunc):
            return compile_kernel(result, target=self.target, out_idx=self.out_idx)
        raise TypeError("autotune factory must return a CompiledKernel or @T.prim_func")

    @staticmethod
    def _make_dummy_args(kernel: CompiledKernel, torch):
        from .compat import torch_dtype_for
        from .ir import ScalarParameter

        device = torch.device("cuda")
        out_set = set(kernel.out_idx)
        args = []
        for idx, param in enumerate(kernel.user_parameters):
            if idx in out_set:
                continue
            if isinstance(param, ScalarParameter):
                args.append(0)
            else:
                args.append(
                    torch.zeros(
                        param.type.shape, dtype=torch_dtype_for(param.type.dtype, torch), device=device
                    )
                )
        return tuple(args)

    def _measure_ms(self, kernel: CompiledKernel, sample_args: tuple, torch) -> float:
        for _ in range(self.warmup):
            kernel(*sample_args)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(self.rep):
            kernel(*sample_args)
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end)) / max(1, self.rep)

    def tune(self, *args, **kwargs) -> TuneResult:
        key = (args, tuple(sorted(kwargs.items())))
        if key in self._cache:
            return self._cache[key]

        configs = self._resolve_configs()
        torch = sys.modules.get("torch")
        if torch is None:
            try:
                torch = importlib.import_module("torch")
            except ImportError:
                torch = None

        has_cuda = torch is not None and torch.cuda.is_available()
        best: TuneResult | None = None
        last_error: Exception | None = None
        for cfg in configs:
            try:
                kernel = self._build_candidate(args, kwargs, cfg)
                if not has_cuda:
                    best = TuneResult(cfg, kernel, 0.0)
                    break
                sample_args = self._make_dummy_args(kernel, torch)
                ms = self._measure_ms(kernel, sample_args, torch)
                if best is None or ms < best.latency_ms:
                    best = TuneResult(cfg, kernel, ms)
            except (CompileError, RuntimeError, ValueError) as exc:
                last_error = exc
                continue

        if best is None:
            raise CompileError(
                f"All autotune configurations failed; last error: {last_error}"
            ) from last_error

        self._cache[key] = best
        self.best_config = best.config
        self.best_latency_ms = best.latency_ms
        return best

    def __call__(self, *args, **kwargs) -> CompiledKernel:
        return self.tune(*args, **kwargs).kernel


def autotune(
    configs: Iterable[dict] | Callable[[], Iterable[dict]],
    *,
    warmup: int = 5,
    rep: int = 20,
    target: str = "auto",
    out_idx=None,
):
    """TileLang-compatible decorator selecting the fastest kernel config on CUDA."""

    def decorator(fn: Callable) -> AutoTuner:
        return AutoTuner(fn, configs, warmup=warmup, rep=rep, target=target, out_idx=out_idx)

    return decorator
