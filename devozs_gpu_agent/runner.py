"""Run one claimed job end to end.

Wires together: backend selection, dataset download + tokenization, checkpoint
download (resume), training with the progress callback, model publish (storage +
optional HF Hub), and terminal reporting (complete / stopped / error).

The stub backend short-circuits the model/dataset steps so the full protocol can
be exercised with no accelerator.
"""
import json
import logging
import os

from .backends import make_backend
from .backends.stub_backend import StubBackend
from .dataset import build_dataset
from .progress_callback import ManagementProgressCallback
from .storage_client import StorageClient

LOGGER = logging.getLogger(__name__)


def run_job(job, client, work_dir):
    # INFER jobs are a forward pass only — no dataset, no trainer, no checkpoints.
    # They report to a different endpoint, so branch before any training setup.
    if getattr(job, "kind", "TRAIN") == "INFER":
        return run_inference(job, client)
    session_dir = os.path.join(work_dir, job.session_id)
    output_dir = os.path.join(session_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    storage = StorageClient(job, client)
    backend = make_backend(job.backend)
    is_stub = isinstance(backend, StubBackend)

    client.report_log(job.session_id, "INFO", f"starting job on {backend.name} backend")
    callback = ManagementProgressCallback(client, storage, job.session_id)

    tokenizer = None
    if is_stub:
        trainer = backend.build_trainer(job, None, None, None, None, output_dir, [callback])
    else:
        hp = json.loads(job.hyperparams or "{}")
        context_length = hp.get("contextLength", 128)

        tokenizer, model = backend.load_tokenizer_and_model(job.base_model)

        dataset_path = os.path.join(session_dir, "dataset.jsonl")
        storage.download_dataset(dataset_path)
        train_ds = build_dataset(dataset_path, tokenizer, context_length)
        eval_ds = None

        trainer = backend.build_trainer(job, tokenizer, model, train_ds, eval_ds, output_dir, [callback])
        callback.bind_trainer(trainer)

    # Resume from a downloaded checkpoint when the job is a RESUMING pick.
    resume_path = None
    if job.resume and job.checkpoint_uri:
        resume_path = os.path.join(session_dir, "resume-checkpoint")
        storage.download_checkpoint(job.checkpoint_uri, resume_path)
        client.report_log(job.session_id, "INFO", f"resuming from {job.checkpoint_uri}")

    # Train. The callback flips state.agent_stopped on a cooperative stop.
    if is_stub:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=resume_path)

    if getattr(trainer.state, "agent_stopped", False):
        client.report_stopped(job.session_id)
        client.report_log(job.session_id, "INFO", "stopped cooperatively; checkpoint retained")
        return

    # Persist the final model and report completion.
    output_ref = _publish_model(job, trainer, storage, output_dir, is_stub, client, tokenizer)
    client.complete(job.session_id, output_ref, None)


def run_inference(job, client):
    """Run one INFER job: load the trained model, generate, and report the samples.

    Errors are reported via the inference result endpoint (NOT the training
    /error path, whose id is a training_session). We swallow the exception after
    reporting so the agent's outer handler doesn't also POST to /error with an
    inference_run id — that endpoint would 500 on the ownership lookup.
    """
    params = json.loads(job.inference_params or "{}")
    is_stub = (job.backend or "").upper() == "STUB"
    try:
        if is_stub:
            num_return = params.get("numReturnSequences") or 3
            outputs = [f"[stub] generated sample {i + 1} for: {job.prompt}" for i in range(num_return)]
        else:
            from .inference import run_inference as _generate
            outputs = _generate(
                job.model_ref, job.prompt, params,
                storage_kind=job.storage_kind, model_key_prefix=job.model_key_prefix,
                base_model=job.base_model,
            )
        client.report_inference_result(job.session_id, outputs)
        LOGGER.info("inference run %s reported %d sample(s)", job.session_id, len(outputs))
    except Exception as e:
        LOGGER.exception("inference run %s failed", job.session_id)
        try:
            client.report_inference_result(job.session_id, None, "INTERNAL", str(e))
        except Exception:
            LOGGER.warning("could not report inference error", exc_info=True)


def _publish_model(job, trainer, storage, output_dir, is_stub, client, tokenizer=None):
    trainer.save_model(output_dir)
    # Save the tokenizer next to the weights so the published model dir is
    # self-contained — inference loads AutoTokenizer.from_pretrained(model_dir)
    # and would fail (vocab_file=None) on a weights-only directory.
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    storage_ref = storage.upload_model(output_dir)
    client.report_log(job.session_id, "INFO", f"model uploaded to {storage_ref}")

    hub_ref = None
    if job.push_to_hub and not is_stub:
        try:
            # HF Hub namespace comes from HF_HUB_NAMESPACE (e.g. your org/user);
            # the repo name is the job's model key. Falls back to the bare name.
            namespace = os.getenv("HF_HUB_NAMESPACE", "").strip()
            name = os.path.basename(job.model_key_prefix)
            repo_id = f"{namespace}/{name}" if namespace else name
            trainer.model.push_to_hub(repo_id)
            hub_ref = repo_id
            client.report_log(job.session_id, "INFO", f"model pushed to HF Hub: {repo_id}")
        except Exception as e:
            client.report_log(job.session_id, "WARN", f"HF Hub push failed: {e}")

    return hub_ref or storage_ref
