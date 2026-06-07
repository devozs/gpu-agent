#!/usr/bin/env bash
#
# bootstrap.sh — pre-setup for a fresh GPU/HPU box. Writes the agent config to
# /etc/devozs-gpu-agent.env FIRST, then sanity-checks that the management URL is
# reachable from this box. Run this before ./setup-agent.sh so the env file is
# the single source of truth for MGMT_URL / AGENT_TYPE / ENROLL_CODE that every
# later step (foreground enroll, systemd service) reads.
#
#   sudo ./deploy/bootstrap.sh --mgmt-url http://10.111.56.26/api \
#                              --type HPU --enroll-code HRT-xxxx
#
# The sanity check posts an empty body to /training/agent/heartbeat:
#   * HTTP 400/401/2xx  → REACHABLE (server answered; empty body is expected to
#                         be rejected) — the URL and port are good.
#   * connection refused / timeout → wrong URL or a blocked port (e.g. a high
#                         port the lab firewall drops; the mgmt host commonly
#                         redirects :80 → its app port, so prefer the :80 URL).
#
# Options:
#   --mgmt-url URL    management base URL INCLUDING /api   (required)
#   --type   CUDA|HPU accelerator family on this box       (required)
#   --enroll-code C   one-time enrollment code (omit if the box is already
#                     enrolled and the token is cached)
#   --env-file PATH   source template to start from
#                     (default: deploy/devozs-gpu-agent.env.example)
#   --user NAME       own the env file as this user        (default: current user)
#   --force           overwrite an existing /etc/devozs-gpu-agent.env
#   --skip-check      write the env file but don't run the sanity check

set -euo pipefail

SERVICE_NAME="devozs-gpu-agent"
ENV_DEST="/etc/${SERVICE_NAME}.env"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SRC="$SCRIPT_DIR/${SERVICE_NAME}.env.example"

MGMT_URL=""
AGENT_TYPE=""
ENROLL_CODE=""
RUN_USER="$(id -un)"
FORCE=0
DO_CHECK=1

while [ $# -gt 0 ]; do
  case "$1" in
    --mgmt-url)    MGMT_URL="$2"; shift 2 ;;
    --type)        AGENT_TYPE="$2"; shift 2 ;;
    --enroll-code) ENROLL_CODE="$2"; shift 2 ;;
    --env-file)    ENV_SRC="$2"; shift 2 ;;
    --user)        RUN_USER="$2"; shift 2 ;;
    --force)       FORCE=1; shift ;;
    --skip-check)  DO_CHECK=0; shift ;;
    -h|--help)     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
done

die()  { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }
log()  { printf '\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }

[ -n "$MGMT_URL" ]   || die "--mgmt-url is required (e.g. http://10.111.56.26/api)"
[ -n "$AGENT_TYPE" ] || die "--type is required (CUDA or HPU)"
case "$AGENT_TYPE" in CUDA|HPU) ;; *) die "--type must be CUDA or HPU (got '$AGENT_TYPE')" ;; esac
[ -f "$ENV_SRC" ]    || die "env template not found: $ENV_SRC"

RUN_GROUP="$(id -gn "$RUN_USER")"

log "Bootstrap config"
echo "  envfile : $ENV_DEST"
echo "  mgmt    : $MGMT_URL"
echo "  type    : $AGENT_TYPE"
echo "  enroll  : $([ -n "$ENROLL_CODE" ] && echo '<set>' || echo '<none — using cached token>')"
echo "  owner   : $RUN_USER ($RUN_GROUP)"

# --- write the env file ------------------------------------------------------
if [ -f "$ENV_DEST" ] && [ "$FORCE" -ne 1 ]; then
  die "$ENV_DEST already exists — re-run with --force to overwrite it"
fi

# Start from the example, then set MGMT_URL / AGENT_TYPE (and ENROLL_CODE if
# given, uncommenting the line). Anything else in the template (proxy block,
# GC_KERNEL_PATH, PT_HPU_LAZY_MODE) is preserved for the admin to tweak.
tmp_env="$(mktemp)"
sed -e "s#^MGMT_URL=.*#MGMT_URL=${MGMT_URL}#" \
    -e "s#^AGENT_TYPE=.*#AGENT_TYPE=${AGENT_TYPE}#" \
    "$ENV_SRC" > "$tmp_env"

if [ -n "$ENROLL_CODE" ]; then
  if grep -qE '^#?ENROLL_CODE=' "$tmp_env"; then
    sed -i -E "s@^#?ENROLL_CODE=.*@ENROLL_CODE=${ENROLL_CODE}@" "$tmp_env"
  else
    printf '\nENROLL_CODE=%s\n' "$ENROLL_CODE" >> "$tmp_env"
  fi
fi

sudo cp "$tmp_env" "$ENV_DEST"
sudo chmod 600 "$ENV_DEST"
sudo chown "$RUN_USER:$RUN_GROUP" "$ENV_DEST"
rm -f "$tmp_env"
ok "wrote $ENV_DEST"

# --- sanity check the management URL -----------------------------------------
if [ "$DO_CHECK" -eq 1 ]; then
  base="${MGMT_URL%/}"
  url="${base}/training/agent/heartbeat"
  log "Sanity check → POST $url"
  # curl's -w always prints a code (000 on connection failure) and exits non-zero
  # then; `|| true` keeps set -e happy WITHOUT appending a second 000 to $http_code.
  http_code="$(curl -sS -m 5 -o /dev/null -w '%{http_code}' \
                 "$url" -X POST -H 'Content-Type: application/json' -d '{}' \
                 2>/dev/null || true)"
  if [ -z "$http_code" ] || [ "$http_code" = "000" ]; then
    warn "UNREACHABLE (connection refused / timeout) — code=$http_code"
    echo "   The management URL is wrong or the port is blocked from this box."
    echo "   Common cause: a high app port (e.g. :18080) is dropped by the lab"
    echo "   firewall while :80 is redirected to it. Try the :80 form, e.g."
    echo "       --mgmt-url http://${base#http://}" | sed -E 's#:[0-9]+/#/#'
    exit 1
  fi
  ok "REACHABLE — HTTP $http_code (an empty body is expected to be rejected with 400/401)"
fi

cat <<EOF

Done. Config is written and the management URL is reachable.
Next:
  ./setup-agent.sh                  # driver + Habana venv + MNIST verify (HPU)
  # then install the agent into the venv, enroll once, and:
  ./deploy/install-service.sh       # reuses $ENV_DEST as-is
EOF
