#!/usr/bin/env bash
#
# uninstall-service.sh — stop, disable, and remove the devozs-gpu-agent systemd
# service. Leaves the cached enrollment token in place (so a later reinstall
# does not need a new code); pass --purge-env to also remove the env file.
#
#   ./deploy/uninstall-service.sh [--purge-env]

set -euo pipefail

SERVICE_NAME="devozs-gpu-agent"
ENV_DEST="/etc/${SERVICE_NAME}.env"
UNIT_DEST="/etc/systemd/system/${SERVICE_NAME}.service"

PURGE_ENV=0
[ "${1:-}" = "--purge-env" ] && PURGE_ENV=1

log() { printf '\033[1;36m== %s\033[0m\n' "$*"; }

log "stopping + disabling $SERVICE_NAME"
sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true
sudo systemctl disable "$SERVICE_NAME" 2>/dev/null || true

log "removing unit $UNIT_DEST"
sudo rm -f "$UNIT_DEST"
sudo systemctl daemon-reload

if [ "$PURGE_ENV" -eq 1 ]; then
  log "removing env file $ENV_DEST"
  sudo rm -f "$ENV_DEST"
else
  log "leaving env file $ENV_DEST (use --purge-env to remove)"
fi

log "done"
