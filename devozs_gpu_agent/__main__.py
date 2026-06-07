"""Entrypoint: ``python -m devozs_gpu_agent``.

With ``--enroll-only`` the agent enrolls (caching its bearer token) and runs a
single readiness preflight, then exits — handy for an unattended setup script.
Without it, the agent enrolls and then polls for jobs forever (the service mode).
"""
import sys

from .agent import run

if __name__ == "__main__":
    run(enroll_only="--enroll-only" in sys.argv[1:])
