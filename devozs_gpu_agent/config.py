"""Agent configuration, read from the environment.

Nothing here is secret-by-default except the enrollment code / token, which the
agent persists to a small local file after enrolling so a restart reuses it.
"""
import json
import logging
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)

DEFAULT_TOKEN_FILE = "~/.devozs_gpu_agent/agent-token.json"


@dataclass
class AgentConfig:
    # Management base URL, including the /api context path, e.g. http://localhost:8080/api
    mgmt_url: str = field(default_factory=lambda: os.getenv("MGMT_URL", "http://localhost:8080/api"))
    # Human name for this box; must match the resource the admin created.
    name: str = field(default_factory=lambda: os.getenv("AGENT_NAME", platform.node() or "agent"))
    # CUDA | HPU — informational on enroll; the server already knows the type.
    type: str = field(default_factory=lambda: os.getenv("AGENT_TYPE", "CUDA"))
    # One-time enrollment code issued by the admin (only needed until we have a token).
    enroll_code: Optional[str] = field(default_factory=lambda: os.getenv("ENROLL_CODE"))
    # Seconds between heartbeat/claim polls.
    poll_interval: float = field(default_factory=lambda: float(os.getenv("POLL_INTERVAL", "5")))
    # Where the bearer token is cached between runs.
    token_file: str = field(default_factory=lambda: os.getenv("AGENT_TOKEN_FILE", DEFAULT_TOKEN_FILE))
    # Force the no-ML stub backend regardless of what the job says (local dev).
    stub: bool = field(default_factory=lambda: os.getenv("GPU_AGENT_STUB", "").strip() == "1")
    # Local scratch dir for datasets/checkpoints the agent downloads.
    work_dir: str = field(default_factory=lambda: os.getenv("AGENT_WORK_DIR", "/tmp/devozs-gpu-agent"))
    # Optional HuggingFace token for push_to_hub.
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))

    def token_path(self) -> Path:
        return Path(os.path.expanduser(self.token_file))

    def load_token(self) -> Optional[str]:
        p = self.token_path()
        if p.exists():
            try:
                return json.loads(p.read_text()).get("token")
            except Exception:
                LOGGER.warning("could not read token file %s", p)
        return None

    def save_token(self, token: str) -> None:
        p = self.token_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"token": token}))
        try:
            p.chmod(0o600)
        except OSError:
            pass

    def clear_token(self) -> None:
        self.token_path().unlink(missing_ok=True)
