#!/usr/bin/env bash
#
# install-agent.sh — install the devozs GPU/HPU agent INTO the Habana venv
# (step 3 of the bare-metal setup). Installs the agent package without disturbing
# the SynapseAI-matched torch (`--no-deps`), plus the extra deps a real training
# job needs. Run after ./setup-agent.sh has created the venv.
#
#   ./deploy/install-agent.sh
#
# Options / env:
#   --venv PATH      Habana venv to install into (default: $VIRTUAL_ENV, else
#                    ~/habanalabs-venv)
#   --no-job-deps    install only the agent (skip datasets/boto3/optimum-habana —
#                    enough to enroll + pass preflight, not to run a real job)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV="${VENV:-}"
JOB_DEPS=1

while [ $# -gt 0 ]; do
  case "$1" in
    --venv)        VENV="$2"; shift 2 ;;
    --no-job-deps) JOB_DEPS=0; shift ;;
    -h|--help)     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
done

die() { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }
log() { printf '\033[1;36m== %s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m== %s\033[0m\n' "$*"; }

# Resolve the Habana venv (same precedence as install-service.sh): an active
# venv, else ~/habanalabs-venv that setup-agent.sh creates.
if [ -z "$VENV" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    VENV="$VIRTUAL_ENV"
  else
    VENV="$HOME/habanalabs-venv"
  fi
fi
[ -x "$VENV/bin/python" ] || die "no python at $VENV/bin/python — run ./setup-agent.sh first (or pass --venv)"

log "Installing the agent into $VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --no-deps: do NOT let pip pull a generic torch over the SynapseAI-matched one
# the venv already has. The agent's only hard runtime dep is requests, present in
# the Habana venv; preflight also needs transformers + habana_frameworks (both
# already installed by setup-agent.sh).
pip install --no-deps -e "$REPO_DIR"

if [ "$JOB_DEPS" -eq 1 ]; then
  log "Installing job deps (datasets, boto3, optimum-habana)"
  # Imported lazily and only when a real job runs — a box can enroll + pass
  # preflight without these, so --no-job-deps is fine for a verify-only box.
  pip install 'datasets>=2.18' 'boto3>=1.34' optimum-habana
fi

# Fail early (here, not at enroll) if the agent isn't importable by this python.
python -c 'import devozs_gpu_agent' || die "devozs_gpu_agent not importable after install"
ok "agent installed. Next: ./deploy/enroll.sh"
