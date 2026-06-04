"""CUDA backend — transformers Trainer on an NVIDIA GPU.

Parameterized by the job's base model + hyperparams. Standard transformers
Trainer on an NVIDIA GPU (falls back to CPU if CUDA is absent, which is fine for
tiny smoke runs).
"""
import json
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .base import TrainingBackend

LOGGER = logging.getLogger(__name__)


class CudaBackend(TrainingBackend):
    name = "CUDA"

    def load_tokenizer_and_model(self, base_model):
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )
        model = AutoModelForCausalLM.from_pretrained(base_model, pad_token_id=tokenizer.pad_token_id)
        model.resize_token_embeddings(len(tokenizer))
        return tokenizer, model

    def build_trainer(self, job, tokenizer, model, train_ds, eval_ds, output_dir, callbacks):
        import torch

        hp = json.loads(job.hyperparams or "{}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        LOGGER.info("CUDA backend training on %s", device)

        args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=not job.resume,
            num_train_epochs=hp.get("epochs", 3),
            per_device_train_batch_size=hp.get("batchSize", 4),
            per_device_eval_batch_size=hp.get("batchSize", 4),
            learning_rate=hp.get("learningRate", 5e-5),
            warmup_steps=hp.get("warmupSteps", 10),
            weight_decay=hp.get("weightDecay", 0.01),
            save_steps=hp.get("saveSteps", 200),
            save_total_limit=2,
            logging_steps=10,
            do_eval=eval_ds is not None,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            callbacks=callbacks,
        )
        return trainer

    def detect_capabilities(self) -> dict:
        caps = {"backend": "CUDA"}
        try:
            import torch
            caps["cudaAvailable"] = torch.cuda.is_available()
            caps["deviceCount"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if torch.cuda.is_available():
                caps["deviceName"] = torch.cuda.get_device_name(0)
            caps["torchVersion"] = torch.__version__
        except Exception:
            caps["cudaAvailable"] = False
        return caps
