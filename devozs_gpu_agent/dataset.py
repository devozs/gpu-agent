"""Turn the exported JSONL into a tokenized dataset for causal-LM fine-tuning.

Each JSONL line is ``{title, subTitle, content}``. We frame one training text per
record as ``<bos> title. subTitle. content <eos>``, then tokenize to a fixed
``context_length`` from the job hyperparams.
"""
import json
import logging
from typing import List

LOGGER = logging.getLogger(__name__)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                LOGGER.warning("skipping malformed jsonl line")
    return rows


def record_to_text(rec: dict) -> str:
    parts = [rec.get("title", ""), rec.get("subTitle", ""), rec.get("content", "")]
    return ". ".join(p.strip() for p in parts if p and p.strip())


def build_dataset(jsonl_path: str, tokenizer, context_length: int):
    """Return a HF Dataset of input_ids/attention_mask/labels for the Trainer."""
    from datasets import Dataset

    rows = read_jsonl(jsonl_path)
    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""
    texts = [f"{bos}{record_to_text(r)}{eos}" for r in rows if record_to_text(r)]
    LOGGER.info("dataset: %d usable records (of %d)", len(texts), len(rows))

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=context_length,
        padding="max_length",
    )

    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        # causal-LM: labels mirror inputs
        "labels": enc["input_ids"],
    })
    return ds
