"""The agent poll loop.

Enroll once (redeem the admin's code for a bearer token, cached to disk), then
forever: heartbeat → claim → run. Every connection is OUTBOUND. Transient
management outages just back off and retry; a job that throws is reported as an
error so the session fails cleanly rather than hanging until the reaper.
"""
import json
import logging
import os
import threading
import time
from urllib.parse import urlparse

import requests

from .backends import make_backend
from .config import AgentConfig
from .management_client import ManagementClient
from .preflight import run_preflight
from .runner import run_job

LOGGER = logging.getLogger(__name__)

# Cap for the transient-outage backoff so a long management downtime settles into a
# steady ~1/min retry instead of growing unbounded — and recovers promptly once it's
# back (the next success resets the backoff).
_MAX_BACKOFF_SECONDS = 60.0


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
    # A model push streams GBs over minutes. Run it in a BACKGROUND thread so the
    # poll loop keeps heartbeating — otherwise management's 60s heartbeat timeout
    # marks the box OFFLINE for the whole push (the "flapping" the admin saw). The
    # holder tracks the in-flight push so we neither launch a second one (management
    # keeps sending modelUploadSessionId until it completes) nor claim a job mid-push.
    push_thread: threading.Thread = None
    while True:
        try:
            ack = client.heartbeat("IDLE")
            backoff_strikes = 0  # management answered — clear any backoff
            # Admin asked for a fresh check: re-run preflight and report.
            if isinstance(ack, dict) and ack.get("reverifyRequested"):
                ready = _do_preflight(cfg, client)
            pushing = push_thread is not None and push_thread.is_alive()
            # Admin asked to fetch a trained model to local: push it up in the
            # background so heartbeats keep flowing. Skip if one is already running.
            if not pushing and isinstance(ack, dict) and ack.get("modelUploadSessionId"):
                sid = ack["modelUploadSessionId"]
                src = ack.get("modelUploadSourceUri")
                # Dedicated client: requests.Session isn't safe for concurrent use, so
                # the background push must not share the loop's session for heartbeats.
                push_client = ManagementClient(cfg.mgmt_url, token=client.token)
                push_thread = threading.Thread(
                    target=_push_model, args=(push_client, sid, src),
                    name=f"push-{sid}", daemon=True)
                push_thread.start()
                pushing = True
            # Don't claim a training/inference job while a push occupies the box.
            if not ready or pushing:
                time.sleep(cfg.poll_interval)
                continue
            job = client.claim()
            if job is None:
                time.sleep(cfg.poll_interval)
                continue

            LOGGER.info("claimed session %s (backend=%s, resume=%s)", job.session_id, job.backend, job.resume)
            try:
                client.heartbeat("BUSY")
                run_job(job, client, cfg.work_dir)
                LOGGER.info("session %s finished", job.session_id)
            except Exception as e:  # job-level failure → report and keep the agent alive
                LOGGER.exception("session %s failed", job.session_id)
                try:
                    client.report_error(job.session_id, "INTERNAL", str(e))
                except Exception:
                    LOGGER.warning("could not report job error", exc_info=True)
        except KeyboardInterrupt:
            LOGGER.info("agent shutting down")
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
