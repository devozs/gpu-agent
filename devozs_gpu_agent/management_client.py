"""HTTP client for the training-agent protocol.

Thin wrapper over ``requests``. Every method maps to one endpoint in the
management service's training-agent controller. All calls are OUTBOUND; the
bearer token is attached automatically once enrolled.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import requests

LOGGER = logging.getLogger(__name__)


class StopRequested(Exception):
    """Raised internally when the server signals a cooperative stop."""


@dataclass
class AgentJob:
    session_id: str
    status: str
    backend: str
    base_model: str
    hyperparams: str          # JSON string
    dataset_spec: str         # JSON string
    dataset_uri: Optional[str]
    dataset_download_url: Optional[str]
    resume: bool
    checkpoint_uri: Optional[str]
    storage_kind: str
    checkpoint_key_prefix: str
    model_key_prefix: str
    push_to_hub: bool

    # job kind: "TRAIN" (default, back-compat) or "INFER". An INFER job ignores the
    # dataset/hyperparams/checkpoint fields and uses model_ref/prompt/inference_params.
    kind: str = "TRAIN"
    model_ref: Optional[str] = None        # trained model: HF Hub id or file://-/s3:// URI
    prompt: Optional[str] = None
    inference_params: str = "{}"           # JSON: {temperature, maxLength, numReturnSequences}

    @classmethod
    def from_json(cls, j: dict) -> "AgentJob":
        return cls(
            session_id=j["sessionId"],
            status=j.get("status"),
            backend=j.get("backend"),
            base_model=j.get("baseModel"),
            hyperparams=j.get("hyperparams") or "{}",
            dataset_spec=j.get("datasetSpec") or "{}",
            dataset_uri=j.get("datasetUri"),
            dataset_download_url=j.get("datasetDownloadUrl"),
            resume=bool(j.get("resume")),
            checkpoint_uri=j.get("checkpointUri"),
            storage_kind=j.get("storageKind") or "local",
            checkpoint_key_prefix=j.get("checkpointKeyPrefix") or "",
            model_key_prefix=j.get("modelKeyPrefix") or "",
            push_to_hub=bool(j.get("pushToHub")),
            kind=j.get("kind") or "TRAIN",
            model_ref=j.get("modelRef"),
            prompt=j.get("prompt"),
            inference_params=j.get("inferenceParams") or "{}",
        )


class ManagementClient:
    def __init__(self, base_url: str, token: Optional[str] = None, timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            # Custom header, NOT Authorization: the server's OAuth2 resource-server
            # filter would 403 an opaque (non-JWT) token sent as a bearer.
            h["X-Agent-Token"] = self.token
        return h

    def _agent(self, path: str) -> str:
        return f"{self.base}/training/agent{path}"

    # --- enrollment ------------------------------------------------------
    def enroll(self, code: str, capabilities: str) -> dict:
        resp = self.session.post(
            self._agent("/enroll"),
            json={"code": code, "capabilities": capabilities},
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self.token = data["token"]
        return data

    # --- liveness + claim ------------------------------------------------
    def heartbeat(self, status: str) -> dict:
        resp = self.session.post(
            self._agent("/heartbeat"), json={"status": status},
            headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def report_preflight(self, ok: bool, detail: str, capabilities: str) -> None:
        resp = self.session.post(
            self._agent("/preflight"),
            json={"ok": ok, "detail": detail, "capabilities": capabilities},
            headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()

    def claim(self) -> Optional[AgentJob]:
        resp = self.session.post(self._agent("/claim"), headers=self._headers(), timeout=self.timeout)
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        return AgentJob.from_json(resp.json())

    # --- session reports -------------------------------------------------
    def report_progress(self, session_id: str, report: dict) -> bool:
        """Returns True if the server has requested a cooperative stop."""
        resp = self.session.post(
            self._agent(f"/sessions/{session_id}/progress"), json=report,
            headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()
        return bool(resp.json().get("stopRequested"))

    def report_log(self, session_id: str, level: str, message: str) -> None:
        try:
            self.session.post(
                self._agent(f"/sessions/{session_id}/log"),
                json={"level": level, "message": message},
                headers=self._headers(), timeout=self.timeout,
            )
        except Exception:
            LOGGER.debug("log post failed (non-fatal)", exc_info=True)

    def report_checkpoint(self, session_id: str, checkpoint_uri: str) -> None:
        resp = self.session.post(
            self._agent(f"/sessions/{session_id}/checkpoint"),
            json={"checkpointUri": checkpoint_uri},
            headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()

    def complete(self, session_id: str, output_model_ref: str, final_checkpoint_uri: Optional[str]) -> None:
        resp = self.session.post(
            self._agent(f"/sessions/{session_id}/complete"),
            json={"outputModelRef": output_model_ref, "finalCheckpointUri": final_checkpoint_uri},
            headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()

    def report_stopped(self, session_id: str) -> None:
        resp = self.session.post(
            self._agent(f"/sessions/{session_id}/stopped"),
            headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()

    def report_error(self, session_id: str, error_type: str, message: str) -> None:
        resp = self.session.post(
            self._agent(f"/sessions/{session_id}/error"),
            json={"errorType": error_type, "message": message},
            headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()

    # --- inference result (INFER jobs) -----------------------------------
    def report_inference_result(self, run_id: str, outputs: Optional[list],
                                error_type: Optional[str] = None,
                                message: Optional[str] = None) -> None:
        """POST the generated samples (or an error) for an INFER job back to management."""
        resp = self.session.post(
            self._agent(f"/inference/{run_id}/result"),
            json={"outputs": outputs, "errorType": error_type, "message": message},
            headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()

    # --- model push (fetch-to-local) -------------------------------------
    def upload_model_file(self, session_id: str, rel_path: str, file_path: str,
                          files_total: int = None, bytes_total: int = None) -> None:
        """Stream one model file up to management under models/{session_id}/{rel_path}.

        files_total/bytes_total are the whole-dir pre-walk totals; sent with each
        file so management can render a live fetch fraction in the monitor."""
        with open(file_path, "rb") as f:
            headers = self._headers()
            # raw bytes, not JSON; rel path travels in a header
            headers["Content-Type"] = "application/octet-stream"
            headers["X-Rel-Path"] = rel_path
            if files_total is not None:
                headers["X-Files-Total"] = str(files_total)
            if bytes_total is not None:
                headers["X-Bytes-Total"] = str(bytes_total)
            resp = self.session.post(
                self._agent(f"/sessions/{session_id}/model-file"),
                data=f, headers=headers, timeout=self.timeout,
            )
            resp.raise_for_status()

    def report_model_upload_complete(self, session_id: str, ok: bool, message: Optional[str] = None) -> None:
        """Tell management the model push finished (or failed); clears the request flag."""
        resp = self.session.post(
            self._agent(f"/sessions/{session_id}/model-upload-complete"),
            json={"ok": ok, "message": message},
            headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()

    def download_dataset_fallback(self, session_id: str, dest_path: str) -> None:
        """GET the JSONL via the authenticated endpoint (no direct storage access)."""
        with self.session.get(
            self._agent(f"/sessions/{session_id}/dataset"),
            headers=self._headers(), timeout=self.timeout, stream=True,
        ) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
