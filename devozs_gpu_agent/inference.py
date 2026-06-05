"""Load a trained model and generate against a prompt (INFER jobs).

The agent claims an INFER job, loads the trained model on whatever accelerator it
has (CUDA / HPU, else CPU), generates, and reports the samples back. ``model_ref``
already encodes the location — an HF Hub repo id, or a ``file://`` / ``s3://``
storage URI produced by a training run — so it alone resolves the weights;
``storage_kind`` / ``model_key_prefix`` are accepted for parity with the job but
only used as a fallback hint.

The generation block matches the GPT-Neo family the trainer targets
(``<|endoftext|>`` stop token, ``add_special_tokens=False``).
"""
import logging
import os
import tempfile
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)

STOP_TOKEN = "<|endoftext|>"

# Special tokens the training backends add to the base tokenizer before resizing
# the model embeddings. A model dir saved without its tokenizer is reconstructed
# from the base model with these EXACT tokens so len(tokenizer) matches the
# trained embedding size (see backends.cuda_backend / hpu_backend).
_SPECIAL_TOKENS = {
    "bos_token": "<|startoftext|>",
    "eos_token": "<|endoftext|>",
    "pad_token": "<|pad|>",
}

# Single-entry cache: {model_ref: (tokenizer, model, device)}. Bounded to one so a
# long-lived agent testing several models doesn't accumulate them in memory.
_CACHE: dict = {}


def _looks_like_hub_id(model_ref: str) -> bool:
    """An HF repo id has no URI scheme and isn't an absolute local path."""
    return "://" not in model_ref and not model_ref.startswith("/")


def _resolve_local_dir(model_ref: str) -> str:
    """Return a local dir holding the weights, downloading from storage if needed."""
    if model_ref.startswith("file://"):
        return urlparse(model_ref).path
    if model_ref.startswith("s3://"):
        import boto3  # lazy: only s3 setups need it
        from botocore.config import Config

        p = urlparse(model_ref)
        bucket, prefix = p.netloc, p.path.lstrip("/")
        dest = tempfile.mkdtemp(prefix="devozs-model-")
        client = boto3.client(
            "s3",
            endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
            config=Config(s3={"addressing_style": "path"}),
        )
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = os.path.relpath(key, prefix)
                target = os.path.join(dest, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                client.download_file(bucket, key, target)
        return dest
    # Already a local path.
    return model_ref


def _select_device():
    """Prefer CUDA, then HPU (Gaudi), else CPU — inference is a forward pass only."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import habana_frameworks.torch.hpu as hthpu  # noqa: F401
        if hthpu.is_available():
            return torch.device("hpu")
    except Exception:
        pass
    return torch.device("cpu")


def _load_tokenizer(source: str, base_model: str = None):
    """Load the tokenizer from the model dir, falling back to the base model.

    A trained model published before the tokenizer was saved alongside it has no
    vocab files, so ``AutoTokenizer.from_pretrained(source)`` raises (vocab_file is
    None). In that case we rebuild the tokenizer from ``base_model`` with the same
    special tokens the trainer used — that's what the model's embeddings were
    resized to, so it stays consistent.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(source)
    except Exception:
        if not base_model:
            raise
        LOGGER.warning(
            "no tokenizer in %s; falling back to base model %s with training special tokens",
            source, base_model,
        )
        return AutoTokenizer.from_pretrained(base_model, **_SPECIAL_TOKENS)


def load_model(model_ref: str, storage_kind: str = None, model_key_prefix: str = None,
               base_model: str = None):
    """Load (tokenizer, model, device), caching by model_ref."""
    if model_ref in _CACHE:
        return _CACHE[model_ref]

    from transformers import AutoModelForCausalLM

    if _looks_like_hub_id(model_ref):
        source = model_ref
        LOGGER.info("loading inference model from HF Hub: %s", source)
    else:
        source = _resolve_local_dir(model_ref)
        LOGGER.info("loading inference model from %s", source)

    tokenizer = _load_tokenizer(source, base_model)
    model = AutoModelForCausalLM.from_pretrained(source, pad_token_id=tokenizer.eos_token_id)

    device = _select_device()
    model.to(device)
    model.eval()
    LOGGER.info("model on device: %s", device)

    # Evict any prior entry (single-entry cache) before storing this one.
    _CACHE.clear()
    _CACHE[model_ref] = (tokenizer, model, device)
    return _CACHE[model_ref]


def generate(loaded, prompt: str, params: dict) -> list:
    """Generate samples for prompt. params: {temperature, maxLength, numReturnSequences}."""
    import time

    import torch

    temperature = (params.get("temperature") or 50) / 100
    if temperature < 0.1:
        temperature = 0.5
    max_length = params.get("maxLength") or 512
    num_return = params.get("numReturnSequences") or 3

    tokenizer, model, device = loaded
    encoded = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    input_ids = encoded if encoded.size()[-1] > 0 else None

    eff_max = max_length
    if input_ids is not None:
        eff_max = min(max_length + len(encoded[0]), 2048)

    # Generation is the long pole — on HPU lazy mode the first run also compiles the
    # graph. Log start/finish so a slow run is never mistaken for a hang.
    LOGGER.info(
        "generating on %s: max_length=%d num_return=%d temperature=%.2f (this can take a while on first run)",
        device, eff_max, num_return, temperature,
    )
    started = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            do_sample=True,
            max_length=eff_max,
            temperature=temperature,
            num_return_sequences=num_return,
        )
    LOGGER.info("generate finished in %.1fs (%d sequence(s))", time.time() - started, len(outputs))

    samples = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        cut = text.find(STOP_TOKEN)
        if cut != -1:
            text = text[:cut]
        samples.append(text)
    return samples


def run_inference(model_ref: str, prompt: str, params: dict,
                  storage_kind: str = None, model_key_prefix: str = None,
                  base_model: str = None) -> list:
    """Convenience: load (cached) + generate. Returns the list of sample strings."""
    loaded = load_model(model_ref, storage_kind, model_key_prefix, base_model)
    return generate(loaded, prompt, params)
