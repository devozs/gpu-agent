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


def split_article(text: str) -> dict:
    """Parse a generated sample into ``{title, subTitle, paragraph}``.

    Inverts the training framing (``title. subTitle. content``): split on ``.``,
    the first segment is the title, the second the sub-title, and the rest joined
    back is the paragraph. When there isn't enough structure (a short/odd sample,
    or the ``(no decodable text …)`` note) we keep the whole thing as the paragraph
    so the caller always gets a renderable shape and never crashes on a bad split.
    """
    text = (text or "").strip()
    segments = [s.strip() for s in text.split(".")]
    while segments and not segments[-1]:
        segments.pop()
    if len(segments) >= 3:
        return {
            "title": segments[0],
            "subTitle": segments[1],
            "paragraph": ". ".join(segments[2:]).strip(),
        }
    return {"title": "", "subTitle": "", "paragraph": text}


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


# Files that indicate a real tokenizer was saved in a model dir. Without one of
# these, AutoTokenizer.from_pretrained(dir) does NOT raise — it builds a degenerate
# empty tokenizer (len ~1) from config.json, which can't encode/decode anything.
_TOKENIZER_FILES = (
    "tokenizer.json", "vocab.json", "vocab.txt", "spiece.model",
    "tokenizer.model", "merges.txt", "tokenizer_config.json",
)


def _dir_has_tokenizer(source: str) -> bool:
    """True only if a local dir actually contains tokenizer files (not just config)."""
    if not os.path.isdir(source):
        return True  # an HF hub id / remote ref — let from_pretrained handle it
    return any(os.path.exists(os.path.join(source, f)) for f in _TOKENIZER_FILES)


def _load_tokenizer(source: str, base_model: str = None):
    """Load the tokenizer from the model dir, falling back to the base model.

    A model published before the tokenizer was saved alongside it has no vocab
    files. Crucially, ``AutoTokenizer.from_pretrained(dir)`` does NOT raise in that
    case — it builds a degenerate empty tokenizer (len ~1) from config.json, which
    silently encodes the prompt to one unknown token and decodes everything to ''.
    So we must detect the missing-tokenizer case up front (not just on exception)
    and rebuild from ``base_model`` with the same special tokens the trainer used —
    that's what the model's embeddings were resized to, so it stays consistent.
    """
    from transformers import AutoTokenizer

    if base_model and not _dir_has_tokenizer(source):
        LOGGER.warning(
            "no tokenizer files in %s; falling back to base model %s with training special tokens",
            source, base_model,
        )
        return AutoTokenizer.from_pretrained(base_model, **_SPECIAL_TOKENS)
    try:
        return AutoTokenizer.from_pretrained(source)
    except Exception:
        if not base_model:
            raise
        LOGGER.warning(
            "tokenizer load from %s failed; falling back to base model %s",
            source, base_model,
        )
        return AutoTokenizer.from_pretrained(base_model, **_SPECIAL_TOKENS)


def load_model(model_ref: str, storage_kind: str = None, model_key_prefix: str = None,
               base_model: str = None):
    """Load (tokenizer, model, device), caching by model_ref."""
    if model_ref in _CACHE:
        return _CACHE[model_ref]

    from transformers import AutoModelForCausalLM

    device = _select_device()
    # On Gaudi, apply optimum-habana's adaptations BEFORE from_pretrained so the
    # model is instantiated as its Gaudi variant (e.g. GaudiGPTNeoForCausalLM),
    # whose generate() understands HPU graphs + static shapes. This mirrors the
    # training backend (hpu_backend.load_tokenizer_and_model) — without it,
    # generate() runs with dynamic shapes and recompiles the graph every token,
    # which on a 400+ token run looks like a hang.
    if device.type == "hpu":
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
        adapt_transformers_to_gaudi()
        LOGGER.info("applied optimum-habana Gaudi adaptations for inference")

    if _looks_like_hub_id(model_ref):
        source = model_ref
        LOGGER.info("loading inference model from HF Hub: %s", source)
    else:
        source = _resolve_local_dir(model_ref)
        LOGGER.info("loading inference model from %s", source)

    tokenizer = _load_tokenizer(source, base_model)
    model = AutoModelForCausalLM.from_pretrained(source, pad_token_id=tokenizer.eos_token_id)

    model.to(device)
    model.eval()
    LOGGER.info("model on device: %s", device)

    # Evict any prior entry (single-entry cache) before storing this one.
    _CACHE.clear()
    _CACHE[model_ref] = (tokenizer, model, device)
    return _CACHE[model_ref]


def generate(loaded, prompt: str, params: dict) -> list:
    """Generate samples for prompt. params: {temperature, maxLength, numReturnSequences}.

    Returns a list of structured ``{title, subTitle, paragraph}`` dicts (the raw
    continuation parsed via :func:`split_article`).
    """
    import time

    import torch

    temperature = (params.get("temperature") or 50) / 100
    if temperature < 0.1:
        temperature = 0.5
    max_length = params.get("maxLength") or 512
    num_return = params.get("numReturnSequences") or 3

    tokenizer, model, device = loaded
    # Training wrapped every example as <bos>{text}<eos>, so the model only emits
    # real content after a BOS — prompting with a bare string (no BOS) makes EOS
    # the likeliest first token and generation stops immediately (empty output).
    bos = tokenizer.bos_token or ""
    encoded = tokenizer.encode(bos + prompt, add_special_tokens=False, return_tensors="pt").to(device)
    input_ids = encoded if encoded.size()[-1] > 0 else None
    prompt_len = int(encoded.size()[-1])

    eff_max = max_length
    if input_ids is not None:
        eff_max = min(max_length + len(encoded[0]), 2048)

    gen_kwargs = dict(
        do_sample=True,
        max_length=eff_max,
        temperature=temperature,
        num_return_sequences=num_return,
    )
    # On Gaudi, generate with STATIC shapes so the decode graph compiles once and
    # replays — the default dynamic shapes recompile every token (each new length
    # is a new shape), which on a long generation stalls for minutes per request.
    # These kwargs are recognised by optimum-habana's GaudiGenerationConfig; bucket
    # padding keeps the static length from forcing every run to the full max.
    if device.type == "hpu":
        gen_kwargs.update(
            static_shapes=True,
            hpu_graphs=True,
            bucket_size=128,
            bucket_internal=True,
            ignore_eos=False,
        )

    # Generation is the long pole — on HPU the first run also compiles the graph.
    # Log start/finish so a slow run is never mistaken for a hang.
    LOGGER.info(
        "generating on %s: max_length=%d num_return=%d temperature=%.2f (this can take a while on first run)",
        device, eff_max, num_return, temperature,
    )
    started = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids, **gen_kwargs)
    if device.type == "hpu":
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
    LOGGER.info("generate finished in %.1fs (%d sequence(s))", time.time() - started, len(outputs))

    # Under HPU static shapes the prompt is padded to a bucket length, so slicing
    # by the unpadded prompt_len would misalign. Instead decode the full sequence
    # (skip_special_tokens drops pad/bos/eos) and strip the decoded prompt prefix —
    # padding-agnostic. CPU/CUDA keep the exact, already-tested token-slice path.
    prompt_text = tokenizer.decode(encoded[0], skip_special_tokens=True).strip() if input_ids is not None else ""

    samples = []
    for i, out in enumerate(outputs):
        if device.type == "hpu":
            full = tokenizer.decode(out, skip_special_tokens=True).strip()
            text = full[len(prompt_text):].lstrip() if prompt_text and full.startswith(prompt_text) else full
            gen_count = max(int(out.size()[-1]) - prompt_len, 0)
        else:
            # Decode only the generated continuation (drop the echoed prompt tokens).
            gen_ids = out[prompt_len:] if int(out.size()[-1]) > prompt_len else out
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_count = int(out.size()[-1]) - prompt_len
        cut = text.find(STOP_TOKEN)
        if cut != -1:
            text = text[:cut]
        text = text.strip()
        LOGGER.info("generate: sample %d — %d generated token(s), %d char(s): %.80r",
                    i + 1, gen_count, len(text), text)
        if not text:
            text = (f"(no decodable text — the model generated {gen_count} token(s) that "
                    f"decoded to nothing; the base model is likely undertrained or its "
                    f"tokenizer can't represent this prompt's language)")
        samples.append(split_article(text))
    return samples


def run_inference(model_ref: str, prompt: str, params: dict,
                  storage_kind: str = None, model_key_prefix: str = None,
                  base_model: str = None) -> list:
    """Convenience: load (cached) + generate. Returns the list of sample strings."""
    loaded = load_model(model_ref, storage_kind, model_key_prefix, base_model)
    return generate(loaded, prompt, params)
