"""Agent-side artifact transfer.

Downloads the dataset and uploads/downloads checkpoints + the final model. Three
transfer modes, chosen by the job's ``storage_kind`` and what's reachable:

* presigned URL (S3): stream directly from/to the bucket — large artifacts never
  touch management.
* shared mount (local-fs): the ``file://`` URI is a path both sides can see.
* HTTPS fallback: the authenticated management dataset endpoint.

Checkpoint upload to S3 walks the local checkpoint dir and puts each file under
the job's ``checkpoint_key_prefix``.
"""
import logging
import os
import shutil
from typing import Optional
from urllib.parse import urlparse

import requests

LOGGER = logging.getLogger(__name__)


def _boto3_client(endpoint: Optional[str] = None):
    import boto3  # imported lazily so non-S3 setups don't need it
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL") or endpoint,
        config=Config(s3={"addressing_style": "path"}),
    )


def _parse_s3_uri(uri: str):
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


class StorageClient:
    def __init__(self, job, mgmt_client):
        self.kind = job.storage_kind
        self.job = job
        self.mgmt = mgmt_client

    # --- dataset ---------------------------------------------------------
    def download_dataset(self, dest_path: str) -> None:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        url = self.job.dataset_download_url
        uri = self.job.dataset_uri

        # 1) Shared local mount: file:// path is directly readable.
        if uri and uri.startswith("file://"):
            src = urlparse(uri).path
            if os.path.exists(src):
                shutil.copyfile(src, dest_path)
                LOGGER.info("dataset copied from shared mount %s", src)
                return

        # 2) Presigned / direct HTTP(S) URL (S3, or local-fs public base url).
        if url and url.startswith(("http://", "https://")):
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            LOGGER.info("dataset downloaded from presigned url")
            return

        # 3) S3 URI without a presigned url: use boto3 with ambient creds.
        if uri and uri.startswith("s3://"):
            bucket, key = _parse_s3_uri(uri)
            _boto3_client().download_file(bucket, key, dest_path)
            LOGGER.info("dataset downloaded from s3://%s/%s", bucket, key)
            return

        # 4) Authenticated management fallback.
        LOGGER.info("dataset: falling back to management HTTPS endpoint")
        self.mgmt.download_dataset_fallback(self.job.session_id, dest_path)

    # --- checkpoints -----------------------------------------------------
    def upload_checkpoint(self, local_dir: str, step) -> str:
        """Persist a local checkpoint dir to storage; returns its canonical URI."""
        rel_prefix = f"{self.job.checkpoint_key_prefix}/checkpoint-{step}"
        if self.kind == "s3":
            bucket, base_key = self._s3_prefix(rel_prefix)
            client = _boto3_client()
            for root, _dirs, files in os.walk(local_dir):
                for name in files:
                    fpath = os.path.join(root, name)
                    rel = os.path.relpath(fpath, local_dir)
                    client.upload_file(fpath, bucket, f"{base_key}/{rel}")
            return f"s3://{bucket}/{base_key}"
        # local-fs shared mount: copy into the storage root under the key.
        dest = os.path.join(self._local_root(), rel_prefix)
        if os.path.abspath(dest) != os.path.abspath(local_dir):
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(local_dir, dest)
        return f"file://{os.path.abspath(dest)}"

    def download_checkpoint(self, checkpoint_uri: str, dest_dir: str) -> str:
        os.makedirs(dest_dir, exist_ok=True)
        if checkpoint_uri.startswith("file://"):
            src = urlparse(checkpoint_uri).path
            if os.path.abspath(src) != os.path.abspath(dest_dir):
                shutil.rmtree(dest_dir, ignore_errors=True)
                shutil.copytree(src, dest_dir)
            return dest_dir
        if checkpoint_uri.startswith("s3://"):
            bucket, prefix = _parse_s3_uri(checkpoint_uri)
            client = _boto3_client()
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel = os.path.relpath(key, prefix)
                    target = os.path.join(dest_dir, rel)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    client.download_file(bucket, key, target)
            return dest_dir
        raise ValueError(f"unsupported checkpoint uri: {checkpoint_uri}")

    # --- final model -----------------------------------------------------
    def upload_model(self, local_dir: str) -> str:
        rel_prefix = self.job.model_key_prefix
        if self.kind == "s3":
            bucket, base_key = self._s3_prefix(rel_prefix)
            client = _boto3_client()
            for root, _dirs, files in os.walk(local_dir):
                for name in files:
                    fpath = os.path.join(root, name)
                    rel = os.path.relpath(fpath, local_dir)
                    client.upload_file(fpath, bucket, f"{base_key}/{rel}")
            return f"s3://{bucket}/{base_key}"
        dest = os.path.join(self._local_root(), rel_prefix)
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(local_dir, dest)
        return f"file://{os.path.abspath(dest)}"

    # --- helpers ---------------------------------------------------------
    def _s3_prefix(self, rel_prefix: str):
        # Bucket comes from the dataset URI (same bucket for all session artifacts).
        if self.job.dataset_uri and self.job.dataset_uri.startswith("s3://"):
            bucket, _ = _parse_s3_uri(self.job.dataset_uri)
        else:
            bucket = os.getenv("TRAINING_S3_BUCKET", "")
        return bucket, rel_prefix

    def _local_root(self) -> str:
        # Derive the storage root from the dataset file:// uri (.../datasets/x.jsonl).
        if self.job.dataset_uri and self.job.dataset_uri.startswith("file://"):
            ds_path = urlparse(self.job.dataset_uri).path
            # strip ".../datasets/<id>.jsonl" -> root
            return os.path.dirname(os.path.dirname(ds_path))
        return os.getenv("TRAINING_STORAGE_ROOT", "/data/devozs-gpu-agent")
