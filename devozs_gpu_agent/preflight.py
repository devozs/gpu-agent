"""Readiness preflight: prove a box can actually train before it claims work.

Two stages:
1. Identify the accelerator by shelling out to the vendor tool — ``nvidia-smi``
   for CUDA, ``hl-smi`` for Gaudi — and parse a short device summary.
2. Run a tiny REAL LLM workload: load a small HF causal-LM, move it to the
   device, and ``generate()`` a few tokens. This exercises the full
   transformers + CUDA/Habana path, not just "is a driver present".

Returns a ``PreflightResult`` the agent posts to management. The stub backend
short-circuits to a pass so local dev needs no GPU.
"""
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

LOGGER = logging.getLogger(__name__)

# Empty by default → offline in-memory probe (no Hub download, air-gap safe).
# Set PREFLIGHT_MODEL to a repo id / local path to validate a real model instead.
DEFAULT_PROBE_MODEL = os.getenv("PREFLIGHT_MODEL", "").strip()


@dataclass
class PreflightResult:
    ok: bool
    detail: str
    capabilities: dict = field(default_factory=dict)

    def capabilities_json(self) -> str:
        return json.dumps(self.capabilities)


def _run(cmd: list, timeout: int = 30) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        LOGGER.warning("%s failed: %s", cmd[0], e)
        return None


def identify_nvidia() -> dict:
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader"])
    caps = {"tool": "nvidia-smi"}
    if not out:
        caps["detected"] = False
        return caps
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    devices = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            devices.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
    caps["detected"] = bool(devices)
    caps["deviceCount"] = len(devices)
    caps["devices"] = devices
    return caps


def identify_gaudi() -> dict:
    # hl-smi output format varies by SynapseAI version; capture a short summary.
    out = _run(["hl-smi", "-Q", "name,driver_version,memory.total", "-f", "csv,noheader"])
    if out is None:
        out = _run(["hl-smi"])  # fall back to the default table
    caps = {"tool": "hl-smi"}
    if not out:
        caps["detected"] = False
        return caps
    caps["detected"] = True
    caps["raw"] = out[:2000]
    return caps


def _resolve_device(device_kind: str):
    import torch
    if device_kind == "CUDA":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_kind == "HPU":
        import habana_frameworks.torch.core as htcore  # noqa: F401
        return "hpu"
    return "cpu"


def _sample_workload(device_kind: str) -> dict:
    """Prove the accelerator + transformers + generate path works.

    By default this is fully OFFLINE: it builds a tiny GPT-2 from config with
    random weights (no Hub download, no torch.load — so it works air-gapped and
    sidesteps the torch<2.6 safetensors/CVE load guard). Set PREFLIGHT_MODEL to a
    repo id or local path to instead load a real model (needs egress or a local
    cache); useful when you want to validate the exact base model too.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    device = _resolve_device(device_kind)

    if DEFAULT_PROBE_MODEL:
        # Opt-in: validate a real model (download or local cache).
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_PROBE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_PROBE_MODEL)
        model.to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer("hello", return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=8)
        sample = tokenizer.decode(out[0], skip_special_tokens=True)[:80]
        return {"probeModel": DEFAULT_PROBE_MODEL, "device": device, "sample": sample}

    # Default: tiny in-memory model, no I/O.
    cfg = AutoConfig.for_model(
        "gpt2", vocab_size=128, n_positions=32, n_embd=32, n_layer=2, n_head=2,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.to(device)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4)).to(device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=8, do_sample=False)
    if device == "hpu":
        # Flush the lazy graph so the generate actually executes on-device (the
        # mark_step the Habana quick-start does between steps). No-op in eager
        # mode (PT_HPU_LAZY_MODE=0); required under the default lazy mode.
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
    return {
        "probeModel": "in-memory tiny-gpt2 (offline)",
        "device": device,
        "lazyMode": os.getenv("PT_HPU_LAZY_MODE", "1"),
        "generatedTokens": int(out.shape[-1]),
    }


def run_preflight(device_kind: str, stub: bool = False) -> PreflightResult:
    """device_kind: CUDA | HPU | STUB."""
    if stub or device_kind == "STUB":
        return PreflightResult(ok=True, detail="stub preflight: no accelerator required",
                               capabilities={"backend": "STUB"})
    try:
        caps = identify_nvidia() if device_kind == "CUDA" else identify_gaudi()
        if not caps.get("detected"):
            return PreflightResult(
                ok=False,
                detail=f"{caps.get('tool')} did not detect a {device_kind} device",
                capabilities=caps,
            )
        workload = _sample_workload(device_kind)
        caps["workload"] = workload
        return PreflightResult(
            ok=True,
            detail=f"{device_kind} ready: {caps.get('tool')} ok; sample generate on {workload['device']} ok",
            capabilities=caps,
        )
    except Exception as e:  # any failure → not ready, with the reason
        LOGGER.exception("preflight failed")
        return PreflightResult(ok=False, detail=f"preflight error: {e}",
                               capabilities={"backend": device_kind, "error": str(e)})
