"""The agent poll loop.

Enroll once (redeem the admin's code for a bearer token, cached to disk), then
forever: heartbeat → claim → run. Every connection is OUTBOUND. Transient
management outages just back off and retry; a job that throws is reported as an
error so the session fails cleanly rather than hanging until the reaper.
"""
import json
import logging
import multiprocessing
import os
import threading
import time
from urllib.parse import urlparse

import requests

from .backends import make_backend
from .child_runner import run_job_child
from .config import AgentConfig
from .management_client import ManagementClient
from .preflight import run_preflight

LOGGER = logging.getLogger(__name__)

# Cap for the transient-outage backoff so a long management downtime settles into a
# steady ~1/min retry instead of growing unbounded — and recovers promptly once it's
# back (the next success resets the backoff).
_MAX_BACKOFF_SECONDS = 60.0

# Grace period after SIGTERM before we SIGKILL a child that ignores it.
_TERMINATE_GRACE_SECONDS = 30.0

# "spawn" (not the default "fork" on Linux): a fork would inherit the parent's
# requests.Session sockets and any imported accelerator runtime state — both unsafe.
# spawn gives the child a clean interpreter that builds its own client and imports
# torch/habana fresh. One shared context for every job we launch.
_MP = multiprocessing.get_context("spawn")


def _device_env_for(cfg: AgentConfig) -> dict:
    """Accelerator env to hand the child, applied before it imports the ML stack.

    HPU needs lazy mode on explicitly (SynapseAI defaults to eager); both backends
    pass through any device-visibility pin and HF credentials. Values default to the
    parent's environment, so a box-wide setting still flows through unless overridden.
    """
    env = {}
    if (cfg.type or "").upper() == "HPU":
        env["PT_HPU_LAZY_MODE"] = os.getenv("PT_HPU_LAZY_MODE", "1")
    for key in ("HABANA_VISIBLE_MODULES", "CUDA_VISIBLE_DEVICES", "HF_TOKEN", "HF_HUB_NAMESPACE"):
        if os.getenv(key) is not None:
            env[key] = os.getenv(key)
    return env


class JobProcess:
    """A single in-flight compute job running in a spawned child process.

    Wraps the multiprocessing.Process plus the result queue the child reports its
    terminal outcome on. The parent never waits on it: it polls is_alive() each loop
    and finalizes once it exits (see _finalize_job)."""

    def __init__(self, proc, session_id, kind, result_queue):
        self.proc = proc
        self.session_id = session_id
        self.kind = kind
        self.result_queue = result_queue

    @classmethod
    def spawn(cls, job, token: str, cfg: AgentConfig) -> "JobProcess":
        result_queue = _MP.SimpleQueue()
        proc = _MP.Process(
            target=run_job_child,
            args=(job, token, _device_env_for(cfg), result_queue),
            name=f"job-{job.session_id}",
            daemon=False,
        )
        proc.start()
        return cls(proc, job.session_id, getattr(job, "kind", "TRAIN"), result_queue)

    def is_alive(self) -> bool:
        return self.proc.is_alive()

    @property
    def exitcode(self):
        return self.proc.exitcode

    def reported_terminal(self):
        """The terminal kind the child put on the queue, or None if it died silent."""
        if not self.result_queue.empty():
            try:
                kind, _sid = self.result_queue.get()
                return kind
            except Exception:
                return None
        return None

    def terminate(self, grace: float = _TERMINATE_GRACE_SECONDS) -> None:
        """Hard cancel: SIGTERM, wait out the grace period, then SIGKILL."""
        if not self.proc.is_alive():
            return
        self.proc.terminate()  # SIGTERM
        self.proc.join(grace)
        if self.proc.is_alive():
            LOGGER.warning("job %s ignored SIGTERM; killing", self.session_id)
            self.proc.kill()  # SIGKILL
            self.proc.join(5.0)


def _root_cause(exc: Exception) -> str:
    """Innermost exception message — the useful line, not the requests/urllib3 wrapper."""
    cur = exc
    while cur.__cause__ or cur.__context__:
        cur = cur.__cause__ or cur.__context__
    msg = str(cur).strip()
    return msg or type(exc).__name__


def _detect_capabilities(cfg: AgentConfig) -> str:
    backend = make_backend("STUB" if cfg.stub else cfg.type)
    caps = backend.detect_capabilities()
    caps["hostname"] = cfg.name
    caps["agentType"] = cfg.type
    return json.dumps(caps)


def _ensure_enrolled(cfg: AgentConfig, client: ManagementClient) -> None:
    if client.token:
        return
    token = cfg.load_token()
    if token:
        client.token = token
        return
    if not cfg.enroll_code:
        raise SystemExit(
            "no bearer token and no ENROLL_CODE set. Create the resource in the "
            "admin UI, copy its enrollment code, and set ENROLL_CODE."
        )
    LOGGER.info("enrolling with management at %s", cfg.mgmt_url)
    data = client.enroll(cfg.enroll_code, _detect_capabilities(cfg))
    cfg.save_token(data["token"])
    LOGGER.info("enrolled as resource %s (%s)", data.get("name"), data.get("resourceId"))


def _device_kind(cfg: AgentConfig) -> str:
    return "STUB" if cfg.stub else cfg.type


def _do_preflight(cfg: AgentConfig, client: ManagementClient) -> bool:
    """Run the readiness preflight and report the verdict. Returns ok."""
    LOGGER.info("running readiness preflight (%s)…", _device_kind(cfg))
    result = run_preflight(_device_kind(cfg), stub=cfg.stub)
    try:
        client.report_preflight(result.ok, result.detail, result.capabilities_json())
    except Exception:
        LOGGER.warning("could not report preflight verdict", exc_info=True)
    LOGGER.info("preflight %s: %s", "OK" if result.ok else "FAILED", result.detail)
    return result.ok


def _upload_with_retries(client, session_id, rel, fpath, files_total, bytes_total, attempts=4):
    """Upload one model file, retrying transient network failures with backoff.

    The push is agent→management over a possibly slow/flaky link; a single file
    timing out (TimeoutError/Connection aborted) shouldn't abort the whole fetch.
    Re-sending is safe — management overwrites the file — and the next file's diff
    still skips anything that did land. Re-raises after the last attempt so the
    caller reports the push failed (and the UI offers resume)."""
    for attempt in range(1, attempts + 1):
        try:
            client.upload_model_file(session_id, rel, fpath, files_total, bytes_total)
            return
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == attempts:
                raise
            delay = min(2 ** (attempt - 1), 15)
            LOGGER.warning("upload of %s failed (%s); retry %d/%d in %ds",
                           rel, _root_cause(e), attempt, attempts - 1, delay)
            time.sleep(delay)


def _push_model(client: ManagementClient, session_id: str, source_uri: str) -> None:
    """Push a trained model dir up to management (fetch-to-local). OUTBOUND-only:
    the agent reads its local file:// model dir and streams each file. Reports
    completion either way so management clears the request and stops re-asking."""
    try:
        if not source_uri or not source_uri.startswith("file://"):
            raise ValueError(f"model source is not a local path: {source_uri!r}")
        root = urlparse(source_uri).path
        if not os.path.isdir(root):
            raise FileNotFoundError(f"model dir not found on this box: {root}")
        # The HF Trainer writes intermediate checkpoint-N/ subdirs into the same
        # output_dir as the final model, but those hold optimizer/scheduler/RNG state
        # (optimizer.pt alone is ~2x the model) that inference never loads. Push ONLY
        # the final model files — skip any checkpoint-* dir — so we don't ship GBs of
        # useless state (which is also what kept timing out).
        def _is_checkpoint(path: str) -> bool:
            rel = os.path.relpath(path, root)
            return any(part.startswith("checkpoint-") for part in rel.split(os.sep))

        # Pre-walk so management knows the totals up front and can show a live
        # fraction ("3/8 files, 40/210 MB") instead of an opaque "uploading".
        paths = [os.path.join(dp, n) for dp, _d, fs in os.walk(root) for n in fs
                 if not _is_checkpoint(os.path.join(dp, n))]
        files_total = len(paths)
        bytes_total = sum(os.path.getsize(p) for p in paths)

        # Resume support: ask management what it already holds and skip files that
        # already landed at the exact size. A fetch interrupted mid-push then only
        # re-sends the missing/partial files instead of starting over. ("From scratch"
        # is handled server-side by wiping the dir first, so the manifest comes back
        # empty and every file is re-sent.) Best-effort: if the manifest call fails we
        # just upload everything.
        try:
            have = client.model_manifest(session_id)
        except Exception:
            LOGGER.warning("could not read model manifest; uploading all files", exc_info=True)
            have = {}

        LOGGER.info("pushing model for session %s: %d file(s), %d byte(s) from %s (%d already present)",
                    session_id, files_total, bytes_total, root, len(have))
        count = 0
        skipped = 0
        for fpath in paths:
            rel = os.path.relpath(fpath, root)
            # os.walk yields OS-separated paths; the manifest keys are '/'-joined.
            rel_key = rel.replace(os.sep, "/")
            if have.get(rel_key) == os.path.getsize(fpath):
                skipped += 1
                LOGGER.info("skip %s (already present, %d bytes)", rel_key, os.path.getsize(fpath))
                continue
            # Retry a transient upload failure (timeout / dropped connection) a few
            # times before giving up on the whole push — one slow file shouldn't
            # force a manual re-fetch. Management overwrites a partial file on the
            # retry, so re-sending is safe.
            _upload_with_retries(client, session_id, rel, fpath, files_total, bytes_total)
            count += 1
            LOGGER.info("pushed %d/%d: %s", count + skipped, files_total, rel)
        client.report_model_upload_complete(session_id, True)
        LOGGER.info("pushed model for session %s (%d uploaded, %d skipped, %d total) to management",
                    session_id, count, skipped, files_total)
    except Exception as e:
        LOGGER.exception("model push for session %s failed", session_id)
        try:
            client.report_model_upload_complete(session_id, False, str(e))
        except Exception:
            LOGGER.warning("could not report model upload failure", exc_info=True)


def _finalize_job(job_proc: "JobProcess", client: ManagementClient) -> None:
    """Reap a finished compute child and guarantee its session reaches a terminal
    state. The happy path is the child already POSTed complete/stopped/error and
    told us via the queue — then there's nothing to do. The crash path is a child
    that OOMed/segfaulted/was killed without reporting: we synthesize the terminal
    report here so the session fails immediately instead of waiting for the reaper."""
    kind = job_proc.reported_terminal()
    code = job_proc.exitcode
    if kind is not None:
        LOGGER.info("session %s finished (%s, exit=%s)", job_proc.session_id, kind, code)
        return
    # Child ended without reporting. exit==0 with no report shouldn't happen, but we
    # still terminate the session defensively rather than leave it hanging.
    msg = f"job process exited {code} without reporting a terminal state"
    LOGGER.warning("session %s: %s — reporting failure", job_proc.session_id, msg)
    try:
        if job_proc.kind == "INFER":
            client.report_inference_result(job_proc.session_id, None, "INTERNAL", msg)
        else:
            client.report_error(job_proc.session_id, "INTERNAL", msg)
    except Exception:
        # Couldn't reach management; the next heartbeat restores liveness and the
        # server-side reaper remains the last-resort net.
        LOGGER.warning("could not report job %s failure", job_proc.session_id, exc_info=True)


def _report_cancelled(job_proc: "JobProcess", client: ManagementClient) -> None:
    """Backstop-report a hard-cancelled (SIGTERM'd) job. The child may not have
    POSTed anything, so the parent reports for it — UNLESS the child already
    reached a terminal state on the queue (the completion race: the server freed
    the box right after the child POSTed complete). In that case the session is
    already terminal server-side; reporting stopped would wrongly override it."""
    kind = job_proc.reported_terminal()
    if kind is not None:
        LOGGER.info("job %s already reported %s before cancel; not overriding",
                    job_proc.session_id, kind)
        return
    try:
        if job_proc.kind == "INFER":
            client.report_inference_result(job_proc.session_id, None, "INTERNAL",
                                           "cancelled (run deleted)")
        else:
            client.report_stopped(job_proc.session_id)
    except Exception:
        # Best-effort: the run was likely deleted, so this may 4xx; that's fine.
        LOGGER.debug("cancel report for %s failed (non-fatal)", job_proc.session_id, exc_info=True)


def run(cfg: AgentConfig = None) -> None:
    cfg = cfg or AgentConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    client = ManagementClient(cfg.mgmt_url)
    _ensure_enrolled(cfg, client)

    # Preflight runs before the poll loop; an unready box keeps heartbeating
    # (so the admin sees it) but will not be handed jobs by management.
    ready = _do_preflight(cfg, client)

    LOGGER.info("agent up; polling every %.1fs (stub=%s, ready=%s)", cfg.poll_interval, cfg.stub, ready)
    # Backoff state for transient management outages: grows on consecutive network
    # failures (so we don't hammer a down server), resets on the first success.
    backoff_strikes = 0
    # The supervisor owns two concurrent activity classes, both decoupled from the
    # heartbeat so the box never flaps OFFLINE:
    #  - job_proc: the ONE in-flight compute job (TRAIN/INFER), in its own process.
    #    BUSY is derived purely from this — a fetch does NOT make the box BUSY.
    #  - push_threads: fetch-to-local model pushes, one daemon thread per session id.
    #    A push streams GBs over minutes; running it off the loop keeps heartbeats
    #    flowing AND lets a fetch run concurrently with a compute job.
    job_proc: JobProcess = None
    push_threads: dict[str, threading.Thread] = {}
    # Hard-cancel detection: the server frees a deleted job's box immediately
    # (markIdle clears currentSessionId), so the heartbeat ack stops naming our
    # session. Confirm over two consecutive heartbeats before killing, so we don't
    # mistake the brief window where a child is still alive right after it POSTed
    # complete (the server already freed the box, but the child is exiting cleanly).
    cancel_strikes = 0
    while True:
        try:
            # BUSY iff a compute job occupies the accelerator. Heartbeat every loop
            # regardless of fetch/job state — this is what keeps the box online
            # through the long pre-train phase (model load + dataset download) that
            # used to block the loop and trip the reaper.
            status = "BUSY" if (job_proc is not None and job_proc.is_alive()) else "IDLE"
            ack = client.heartbeat(status)
            backoff_strikes = 0  # management answered — clear any backoff

            # Admin asked for a fresh check: re-run preflight. Only when idle — a
            # running job must not be disturbed (and the server rejects reverify on BUSY).
            if status == "IDLE" and isinstance(ack, dict) and ack.get("reverifyRequested"):
                ready = _do_preflight(cfg, client)

            # Hard cancel (delete-as-kill): a running job whose box the server has
            # reclaimed. A cooperative stop keeps the box bound (assignedSessionId
            # unchanged) and is handled in-step by the callback; a DELETE frees it,
            # so assignedSessionId no longer matches our live job. This is the only
            # way to abort the pre-train phase or an INFER run (no step callback to
            # carry stopRequested). After SIGTERM the parent reports the terminal
            # state as backstop, since a killed child can't POST cleanly.
            if job_proc is not None and job_proc.is_alive() and isinstance(ack, dict):
                assigned = ack.get("assignedSessionId")
                if assigned != job_proc.session_id:
                    cancel_strikes += 1
                    if cancel_strikes >= 2:
                        LOGGER.info("job %s no longer assigned to this box; cancelling",
                                    job_proc.session_id)
                        job_proc.terminate()
                        _report_cancelled(job_proc, client)
                        job_proc = None
                        cancel_strikes = 0
                else:
                    cancel_strikes = 0
            else:
                cancel_strikes = 0

            # Prune finished pushes (each self-reports completion; this is bookkeeping).
            for sid in [s for s, t in push_threads.items() if not t.is_alive()]:
                push_threads.pop(sid, None)

            # Admin asked to fetch a trained model to local: push it up in a background
            # thread, concurrently with whatever else is running. Dedup per session id
            # (management keeps sending modelUploadSessionId until the push completes).
            if isinstance(ack, dict) and ack.get("modelUploadSessionId"):
                sid = ack["modelUploadSessionId"]
                if sid not in push_threads:
                    src = ack.get("modelUploadSourceUri")
                    # Dedicated client: requests.Session isn't safe for concurrent use,
                    # so each push gets its own, never sharing the loop's heartbeat session.
                    push_client = ManagementClient(cfg.mgmt_url, token=client.token)
                    t = threading.Thread(
                        target=_push_model, args=(push_client, sid, src),
                        name=f"push-{sid}", daemon=True)
                    t.start()
                    push_threads[sid] = t

            # Reap a finished compute job and make sure its session terminated.
            if job_proc is not None and not job_proc.is_alive():
                _finalize_job(job_proc, client)
                job_proc = None

            # Claim only when ready and no compute job is in flight. A fetch in flight
            # must NOT block claiming — that's the whole point of decoupling them.
            if ready and job_proc is None:
                job = client.claim()
                if job is not None:
                    LOGGER.info("claimed session %s (backend=%s, resume=%s)",
                                job.session_id, job.backend, job.resume)
                    # Flip the UI to BUSY immediately instead of waiting a full poll.
                    try:
                        client.heartbeat("BUSY")
                    except (requests.ConnectionError, requests.Timeout):
                        pass  # the next loop's heartbeat will catch up
                    job_proc = JobProcess.spawn(job, client.token, cfg)

            time.sleep(cfg.poll_interval)
        except KeyboardInterrupt:
            LOGGER.info("agent shutting down")
            if job_proc is not None and job_proc.is_alive():
                LOGGER.info("terminating in-flight job %s", job_proc.session_id)
                job_proc.terminate()
            return
        except (requests.ConnectionError, requests.Timeout) as e:
            # Management briefly unreachable (restart/redeploy/blip): "Connection
            # refused", a connect timeout, or DNS failure. Expected and self-healing
            # — log a concise one-liner (not a 25-line stack trace that reads like a
            # crash) and back off with a cap so we don't hammer a down server.
            backoff_strikes += 1
            delay = min(cfg.poll_interval * (2 ** (backoff_strikes - 1)), _MAX_BACKOFF_SECONDS)
            LOGGER.warning("management unreachable (%s); retry in %.0fs", _root_cause(e), delay)
            time.sleep(delay)
        except Exception:
            # An UNEXPECTED error (bug, bad response, etc.) — keep the full traceback.
            LOGGER.warning("poll loop error; backing off", exc_info=True)
            time.sleep(cfg.poll_interval)
