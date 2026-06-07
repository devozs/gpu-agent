#!/usr/bin/env bash
#
# enroll.sh — enroll this box once (step 4 of the bare-metal setup). Reads the
# config bootstrap.sh wrote to /etc/devozs-gpu-agent.env (MGMT_URL / AGENT_TYPE /
# ENROLL_CODE / PT_HPU_LAZY_MODE), redeems the enrollment code for a bearer token
# (cached to ~/.devozs_gpu_agent/agent-token.json), runs one readiness preflight,
# then EXITS — no manual Ctrl+C needed. The systemd service (step 5) reuses the
# cached token.
#
#   ./deploy/enroll.sh
#
# Options / env:
#   --venv PATH      Habana venv to run from (default: $VIRTUAL_ENV, else
#                    ~/habanalabs-venv)
#   --env-file PATH  config to source (default: /etc/devozs-gpu-agent.env)
#
# Idempotent: once a token is cached the enrollment code is ignored, so re-running
# just re-runs preflight against the cached token.

set -euo pipefail

ENV_FILE="${ENV_FILE:-/etc/devozs-gpu-agent.env}"
VENV="${VENV:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --venv)     VENV="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    -h|--help)  grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
done

die() { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }
log() { printf '\033[1;36m== %s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m== %s\033[0m\n' "$*"; }

# Distinguish "missing" from "exists but unreadable" — the latter is usually a
# root-owned file from running bootstrap.sh under sudo on an older version.
if [ ! -e "$ENV_FILE" ]; then
  die "$ENV_FILE not found — run ./deploy/bootstrap.sh first"
elif [ ! -r "$ENV_FILE" ]; then
  die "$ENV_FILE exists but is not readable by $(id -un) (owner $(stat -c '%U:%G mode %a' "$ENV_FILE" 2>/dev/null)). Fix with: sudo chown $(id -un) $ENV_FILE"
fi

if [ -z "$VENV" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then VENV="$VIRTUAL_ENV"; else VENV="$HOME/habanalabs-venv"; fi
fi
[ -x "$VENV/bin/python" ] || die "no python at $VENV/bin/python — run ./deploy/install-agent.sh first (or pass --venv)"

log "Loading config from $ENV_FILE"
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

log "Enrolling (one preflight, then exit)…"
# --enroll-only: enroll + cache token + one preflight, then exit 0/1 — no poll loop.
"$VENV/bin/python" -m devozs_gpu_agent --enroll-only

ok "Enrolled and verified. Next: ./deploy/install-service.sh"
