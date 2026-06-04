"""Pluggable training backends, selected by the job's backend field.

* :class:`CudaBackend` — transformers Trainer on an NVIDIA GPU.
* :class:`HpuBackend`  — optimum-habana GaudiTrainer on an Intel Gaudi VM.
* :class:`StubBackend` — no-ML fast path for local dev (no accelerator).
"""
from .base import TrainingBackend


def make_backend(kind: str) -> TrainingBackend:
    kind = (kind or "STUB").upper()
    if kind == "CUDA":
        from .cuda_backend import CudaBackend
        return CudaBackend()
    if kind == "HPU":
        from .hpu_backend import HpuBackend
        return HpuBackend()
    from .stub_backend import StubBackend
    return StubBackend()
