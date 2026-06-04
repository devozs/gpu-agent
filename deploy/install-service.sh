#!/usr/bin/env bash
#
# install-service.sh — install the devozs GPU/HPU agent as a systemd SYSTEM
# service (starts on boot, auto-restarts on crash, survives logout).
#
# Run from a checkout of this repo, as the user the agent should run as, with
# sudo available:
#
#   ./deploy/install-service.sh
#
# It auto-detects the venv python, the repo dir, and the current user, fills in
# the unit template, installs an env file (without clobbering an existing one),
# then enables + starts the service.
#
# Options / env overrides:
#   --python PATH    python to run the agent (default: $VIRTUAL_ENV/bin/python,
#                    else ~/habanalabs-venv/bin/python, else `command -v python`)
#   --env-file PATH  source env file to install from (default:
#                    deploy/devozs-gpu-agent.env.example)
#   --user NAME      run the service as this user   (default: current user)
#   --no-start       install + enable but don't start now
#
# Prerequisite: the agent must already be importable by that python
#   (`pip install --no-deps -e .` into the venv) and, ideally, already enrolled
#   once so the token is cached. See README.md.

set -euo pipefail

SERVICE_NAME="devozs-gpu-agent"
ENV_DEST="/etc/${SERVICE_NAME}.env"
UNIT_DEST="/etc/systemd/system/${SERVICE_NAME}.service"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UNIT_SRC="$SCRIPT_DIR/${SERVICE_NAME}.service"

PYTHON=""
ENV_SRC="$SCRIPT_DIR/${SERVICE_NAME}.env.example"
RUN_USER="$(id -un)"
DO_START=1

while [ $# -gt 0 ]; do
  case "$1" in
    --python)   PYTHON="$2"; shift 2 ;;
    --env-file) ENV_SRC="$2"; shift 2 ;;
    --user)     RUN_USER="$2"; shift 2 ;;
    --no-start) DO_START=0; shift ;;
    -h|--help)  grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
done

die() { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }
log() { printf '\033[1;36m== %s\033[0m\n' "$*"; }

# --- resolve the python that can run the agent ------------------------------
if [ -z "$PYTHON" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
  elif [ -x "$HOME/habanalabs-venv/bin/python" ]; then
    PYTHON="$HOME/habanalabs-venv/bin/python"
  else
    PYTHON="$(command -v python3 || command -v python || true)"
  fi
fi
[ -n "$PYTHON" ] && [ -x "$PYTHON" ] || die "could not resolve a python (pass --python PATH)"

# Verify the agent is importable by that python — fail early, not at boot.
"$PYTHON" -c "import devozs_gpu_agent" 2>/dev/null \
  || die "'$PYTHON' can't import devozs_gpu_agent — run 'pip install --no-deps -e .' into that env first"

RUN_GROUP="$(id -gn "$RUN_USER")"

# The unit puts the venv on PATH so optimum-habana's `pip list | grep` version
# probe resolves the venv pip (it runs python by absolute path, not via PATH).
VENV_BIN="$(dirname "$PYTHON")"
VENV_ROOT="$(dirname "$VENV_BIN")"

log "Service config"
echo "  service : $SERVICE_NAME"
echo "  user    : $RUN_USER ($RUN_GROUP)"
echo "  python  : $PYTHON"
echo "  workdir : $REPO_DIR"
echo "  envfile : $ENV_DEST"

# --- install the env file (don't clobber an edited one) ---------------------
if [ -f "$ENV_DEST" ]; then
  log "env file $ENV_DEST already exists — leaving it untouched"
else
  log "installing env file → $ENV_DEST (edit it to set MGMT_URL / AGENT_TYPE)"
  sudo cp "$ENV_SRC" "$ENV_DEST"
  sudo chmod 600 "$ENV_DEST"
  sudo chown "$RUN_USER:$RUN_GROUP" "$ENV_DEST"
fi

# --- render + install the unit ----------------------------------------------
log "installing unit → $UNIT_DEST"
tmp_unit="$(mktemp)"
sed -e "s|__USER__|$RUN_USER|g" \
    -e "s|__GROUP__|$RUN_GROUP|g" \
    -e "s|__WORKDIR__|$REPO_DIR|g" \
    -e "s|__ENVFILE__|$ENV_DEST|g" \
    -e "s|__PYTHON__|$PYTHON|g" \
    -e "s|__VENV_BIN__|$VENV_BIN|g" \
    -e "s|__VENV__|$VENV_ROOT|g" \
    "$UNIT_SRC" > "$tmp_unit"
sudo cp "$tmp_unit" "$UNIT_DEST"
rm -f "$tmp_unit"

log "reloading systemd + enabling service"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

if [ "$DO_START" -eq 1 ]; then
  log "starting service"
  sudo systemctl restart "$SERVICE_NAME"
  sleep 2
  sudo systemctl --no-pager --full status "$SERVICE_NAME" || true
fi

cat <<EOF

Done. Useful commands:
  sudo systemctl status   $SERVICE_NAME
  sudo systemctl restart  $SERVICE_NAME
  sudo systemctl stop     $SERVICE_NAME      # stops claiming jobs; enrollment stays cached
  journalctl -u $SERVICE_NAME -f             # live logs
  sudoedit $ENV_DEST                         # change MGMT_URL / AGENT_TYPE, then restart

First run on a never-enrolled box: set ENROLL_CODE in $ENV_DEST, start once, then
remove it (the token is cached and reused on every restart).
EOF
