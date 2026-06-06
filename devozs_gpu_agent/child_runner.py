"""The compute job runs here, in a spawned child process.

Why a separate process (not a thread): a training/inference run loads native
CUDA/HPU runtimes that can OOM or segfault. Isolating each job means a crash
takes down only that job, never the agent's heartbeat loop, and a hard cancel is
a clean SIGTERM. We use the "spawn" start method (see agent.py), so this module
is re-imported in a fresh interpreter — every heavy import and the HTTP session
are built here, never inherited across the fork.

The child reports the terminal outcome two ways: it POSTs the real terminal call
(complete/stopped/error/inference result) over its OWN ManagementClient, and it
also drops a (kind, session_id) tuple on the result queue so the parent knows the
job ended cleanly and need not synthesize an error. A child that dies without
putting anything on the queue is exactly what the parent's crash-safety catches.
"""
import logging
import os
import signal
import sys

LOGGER = logging.getLogger(__name__)


def _apply_device_env(env: dict) -> None:
    """Set accelerator env BEFORE any backend (torch/habana) import in this child.

    spawn already inherits os.environ, so this mostly documents the contract and
    leaves a per-job override seam (the parent may someday pin specific devices).
    setdefault so an explicitly-set value from the environment still wins.
    """
    for key, value in (env or {}).items():
        if value is not None:
            os.environ.setdefault(key, str(value))


def _install_sigterm_handler() -> None:
    """Turn SIGTERM (the parent's hard-cancel) into SystemExit so an in-flight
    trainer.train() unwinds through normal exception handling instead of dying
    where it stands. The parent is the backstop reporter after a SIGTERM, so we
    don't try to POST from inside the handler (the run may be deep in C code)."""
    def _handler(signum, frame):
        raise SystemExit(f"received signal {signum}")
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):
        # Not on the main thread of the child, or platform without SIGTERM — the
        # parent's terminate()->kill() escalation still bounds the cancel.
        LOGGER.debug("could not install SIGTERM handler", exc_info=True)


def run_job_child(job, token: str, device_env: dict, result_queue) -> None:
    """multiprocessing entrypoint. Runs one job to a terminal state.

    job          an AgentJob (picklable dataclass) handed over from the parent.
    token        the agent bearer token; a fresh ManagementClient is built HERE.
    device_env   accelerator env to apply before importing the ML stack.
    result_queue a multiprocessing SimpleQueue; we put (kind, session_id) on it.
    """
    _apply_device_env(device_env)
    _install_sigterm_handler()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # Imports deferred until after device env is set, and kept in the child so the
    # parent never loads the ML stack.
    from .config import AgentConfig
    from .management_client import ManagementClient
    from .runner import run_job

    cfg = AgentConfig()
    # The requests.Session is created INSIDE the child — never inherited across spawn.
    client = ManagementClient(cfg.mgmt_url, token=token)
    is_infer = getattr(job, "kind", "TRAIN") == "INFER"

    try:
        kind = run_job(job, client, cfg.work_dir)
        # run_inference swallows its own errors and returns "inference_error"; that
        # is still a clean, fully-reported terminal state from the parent's view.
        result_queue.put((kind or "complete", job.session_id))
    except SystemExit:
        # Hard cancel (SIGTERM) — the parent reports stopped/error as the backstop.
        # Put nothing on the queue; let the parent see "died without reporting".
        LOGGER.info("job %s cancelled (SIGTERM)", job.session_id)
        raise
    except Exception as e:
        LOGGER.exception("job %s failed in child", job.session_id)
        # Report the terminal failure ourselves so the session fails immediately
        # rather than waiting for the reaper. INFER uses its own endpoint (the
        # training /error path 500s on an inference run id — see runner.py).
        try:
            if is_infer:
                client.report_inference_result(job.session_id, None, "INTERNAL", str(e))
                result_queue.put(("inference_error", job.session_id))
            else:
                client.report_error(job.session_id, "INTERNAL", str(e))
                result_queue.put(("error", job.session_id))
        except Exception:
            # Couldn't report — leave the queue empty so the parent synthesizes it.
            LOGGER.warning("could not report job failure from child", exc_info=True)
        sys.exit(1)
