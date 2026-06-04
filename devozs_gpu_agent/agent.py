"""The agent poll loop.

Enroll once (redeem the admin's code for a bearer token, cached to disk), then
forever: heartbeat → claim → run. Every connection is OUTBOUND. Transient
management outages just back off and retry; a job that throws is reported as an
error so the session fails cleanly rather than hanging until the reaper.
"""
import json
import logging
import time

from .backends import make_backend
from .config import AgentConfig
from .management_client import ManagementClient
from .preflight import run_preflight
from .runner import run_job

LOGGER = logging.getLogger(__name__)


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


def run(cfg: AgentConfig = None) -> None:
    cfg = cfg or AgentConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    client = ManagementClient(cfg.mgmt_url)
    _ensure_enrolled(cfg, client)

    # Preflight runs before the poll loop; an unready box keeps heartbeating
    # (so the admin sees it) but will not be handed jobs by management.
    ready = _do_preflight(cfg, client)

    LOGGER.info("agent up; polling every %.1fs (stub=%s, ready=%s)", cfg.poll_interval, cfg.stub, ready)
    while True:
        try:
            ack = client.heartbeat("IDLE")
            # Admin asked for a fresh check: re-run preflight and report.
            if isinstance(ack, dict) and ack.get("reverifyRequested"):
                ready = _do_preflight(cfg, client)
            if not ready:
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
        except Exception:
            # Transient management/network problem; back off and keep polling.
            LOGGER.warning("poll loop error; backing off", exc_info=True)
            time.sleep(cfg.poll_interval)
