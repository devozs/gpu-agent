"""Intel Gaudi (HPU) backend via optimum-habana.

Mirrors the CUDA backend but swaps in GaudiTrainer/GaudiTrainingArguments and the
Habana runtime. These imports only work inside a Habana environment (SynapseAI +
habana_frameworks must match the driver on the VM), so they are deferred to
build_trainer — the agent process itself imports fine on a laptop.
"""
import json
import logging

from .base import TrainingBackend

LOGGER = logging.getLogger(__name__)


class HpuBackend(TrainingBackend):
    name = "HPU"

    def load_tokenizer_and_model(self, base_model):
        # Apply optimum-habana's Gaudi patches BEFORE from_pretrained. This swaps
        # the model classes (e.g. GPTNeoForCausalLM -> GaudiGPTNeoForCausalLM,
        # whose forward accepts cache_position) AND the inner *.forward methods.
        # GaudiTrainer would call this too, but only at trainer-construction time
        # — by then the model instance already exists, so the *class* swap misses
        # it and you get a stock CausalLM calling a Gaudi model-forward that lacks
        # cache_position -> TypeError. Patching here keeps the model fully Gaudi,
        # and it's model-agnostic (every model OH supports gets its Gaudi classes).
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
        adapt_transformers_to_gaudi()

        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        # Deferred imports: present only inside a Habana environment.
        import habana_frameworks.torch.core as htcore  # noqa: F401
        from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments

        hp = json.loads(job.hyperparams or "{}")
        # Gaudi is a bf16-first accelerator: every optimum-habana training example
        # (and our own run_glue.py smoke test) trains with --bf16. Plain fp32
        # training trips the lazy-mode graph compiler on ops like layer-norm
        # ("synStatus 26 ... layer_norm_fwd_f32 ... Graph compile failed"). Default
        # bf16 ON for HPU; allow a job to force it off via hyperparams.bf16=false.
        use_bf16 = bool(hp.get("bf16", True))
        LOGGER.info("HPU backend training on Gaudi (bf16=%s)", use_bf16)

        gaudi_config = GaudiConfig(use_fused_adam=True, use_fused_clip_norm=True)
        args = GaudiTrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=not job.resume,
            use_habana=True,
            use_lazy_mode=True,
            gaudi_config_name=None,
            bf16=use_bf16,
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
        trainer = GaudiTrainer(
            model=model,
            gaudi_config=gaudi_config,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            callbacks=callbacks,
        )
        return trainer

    def detect_capabilities(self) -> dict:
        caps = {"backend": "HPU"}
        try:
            import habana_frameworks.torch.hpu as hthpu
            caps["hpuAvailable"] = hthpu.is_available()
            caps["deviceCount"] = hthpu.device_count()
        except Exception:
            caps["hpuAvailable"] = False
        return caps
