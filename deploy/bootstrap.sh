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
# The sanity check posts an empty body to /training/agent/heartbeat and honours
# NO_PROXY (so an internal host bypasses any corporate proxy, like the agent will):
#   * HTTP 2xx/400/401 → REACHABLE — the management APP answered (the empty body
#                        is expected to be rejected with 400/401).
#   * HTTP 5xx (502/503/504) → a PROXY/GATEWAY answered, not the app. On the Intel
#                        lab a bare hostname is sent through the DMZ proxy because
#                        NO_PROXY only covers 10.0.0.0/8 and .intel.com — use the
#                        LAN IP (http://10.111.56.26/api) or an .intel.com FQDN.
#   * 000 (refused/timeout/DNS) → wrong URL or a blocked port; a high app port
#                        (:18080) is often firewalled while :80 redirects to it.
#
# Re-run the check alone any time (reads MGMT_URL from the env file):
#   ./deploy/bootstrap.sh --check-only
#
# Options:
#   --mgmt-url URL    management base URL INCLUDING /api   (required unless --check-only)
#   --type   CUDA|HPU accelerator family on this box       (required unless --check-only)
#   --enroll-code C   one-time enrollment code (omit if the box is already
#                     enrolled and the token is cached)
#   --env-file PATH   source template to start from
#                     (default: deploy/devozs-gpu-agent.env.example)
#   --user NAME       own the env file as this user        (default: current user)
#   --force           overwrite an existing /etc/devozs-gpu-agent.env
#   --skip-check      write the env file but don't run the sanity check
#   --check-only      only re-run the reachability check (no writing); reads
#                     MGMT_URL from /etc/devozs-gpu-agent.env unless --mgmt-url given

set -euo pipefail

SERVICE_NAME="devozs-gpu-agent"
ENV_DEST="/etc/${SERVICE_NAME}.env"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SRC="$SCRIPT_DIR/${SERVICE_NAME}.env.example"

MGMT_URL=""
AGENT_TYPE=""
ENROLL_CODE=""
# This script is run WITH sudo (it writes /etc), so `id -un` would be root and the
# env file would land root-owned and unreadable by the human user who later runs
# enroll.sh. Default the owner to the invoking user ($SUDO_USER) when under sudo.
RUN_USER="${SUDO_USER:-$(id -un)}"
FORCE=0
DO_CHECK=1
CHECK_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --mgmt-url)    MGMT_URL="$2"; shift 2 ;;
    --type)        AGENT_TYPE="$2"; shift 2 ;;
    --enroll-code) ENROLL_CODE="$2"; shift 2 ;;
    --env-file)    ENV_SRC="$2"; shift 2 ;;
    --user)        RUN_USER="$2"; shift 2 ;;
    --force)       FORCE=1; shift ;;
    --skip-check)  DO_CHECK=0; shift ;;
    --check-only)  CHECK_ONLY=1; shift ;;
    -h|--help)     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
done

die()  { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }
log()  { printf '\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }

# Re-runnable connectivity check: read MGMT_URL from the env file (or --mgmt-url)
# and only run the sanity check — no writing, no other required args. Use this to
# re-verify reachability any time before moving on to ./setup-agent.sh.
if [ "$CHECK_ONLY" -eq 1 ]; then
  if [ -z "$MGMT_URL" ]; then
    [ -f "$ENV_DEST" ] || die "--check-only needs $ENV_DEST (run bootstrap first) or an explicit --mgmt-url"
    MGMT_URL="$(sed -n 's/^MGMT_URL=//p' "$ENV_DEST" | head -n1)"
    [ -n "$MGMT_URL" ] || die "no MGMT_URL= found in $ENV_DEST — pass --mgmt-url"
  fi
else
  [ -n "$MGMT_URL" ]   || die "--mgmt-url is required (e.g. http://10.111.56.26/api)"
  [ -n "$AGENT_TYPE" ] || die "--type is required (CUDA or HPU)"
  case "$AGENT_TYPE" in CUDA|HPU) ;; *) die "--type must be CUDA or HPU (got '$AGENT_TYPE')" ;; esac
  [ -f "$ENV_SRC" ]    || die "env template not found: $ENV_SRC"
fi

if [ "$CHECK_ONLY" -ne 1 ]; then
  RUN_GROUP="$(id -gn "$RUN_USER")"

  log "Bootstrap config"
  echo "  envfile : $ENV_DEST"
  echo "  mgmt    : $MGMT_URL"
  echo "  type    : $AGENT_TYPE"
  echo "  enroll  : $([ -n "$ENROLL_CODE" ] && echo '<set>' || echo '<none — using cached token>')"
  echo "  owner   : $RUN_USER ($RUN_GROUP)"

  # --- write the env file ----------------------------------------------------
  if [ -f "$ENV_DEST" ] && [ "$FORCE" -ne 1 ]; then
    die "$ENV_DEST already exists — re-run with --force to overwrite it, or --check-only to just re-test connectivity"
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
fi

# --- sanity check the management URL -----------------------------------------
if [ "$DO_CHECK" -eq 1 ]; then
  base="${MGMT_URL%/}"
  url="${base}/training/agent/heartbeat"
  log "Sanity check → POST $url"

  # Mirror how the agent (python requests) will reach the mgmt host: honour
  # NO_PROXY so an internal host bypasses any corporate proxy. If the admin
  # hasn't exported NO_PROXY in this shell, fall back to the same defaults the
  # env template ships so a 10.x / .intel.com host isn't sent through the proxy.
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12,.intel.com}"
  export no_proxy="$NO_PROXY"

  # curl's -w always prints a code (000 on connection failure) and exits non-zero
  # then; `|| true` keeps set -e happy WITHOUT appending a second 000 to $http_code.
  http_code="$(curl -sS -m 5 -o /dev/null -w '%{http_code}' \
                 "$url" -X POST -H 'Content-Type: application/json' -d '{}' \
                 2>/dev/null || true)"
  [ -z "$http_code" ] && http_code=000

  # Only the management APP answering counts as reachable: 2xx, or the expected
  # rejection of the empty body (400/401). A 5xx — especially 502/503/504 — means
  # a proxy/gateway answered, NOT the app; 000 means refused/timeout/DNS.
  case "$http_code" in
    2[0-9][0-9]|400|401)
      ok "REACHABLE — HTTP $http_code (the app answered; an empty body is expected to be rejected with 400/401)"
      ;;
    000)
      warn "UNREACHABLE (connection refused / timeout / DNS) — code=000"
      echo "   The management URL is wrong or the port is blocked from this box."
      echo "   A high app port (e.g. :18080) is often dropped by the lab firewall"
      echo "   while :80 is redirected to it — prefer the :80 form."
      exit 1
      ;;
    50[0-9]|5[0-9][0-9])
      warn "A GATEWAY/PROXY answered, NOT the management app — HTTP $http_code"
      echo "   A $http_code (e.g. 502/503/504) means a proxy sat in the path and could"
      echo "   not reach the backend. On the Intel lab a bare hostname like"
      echo "   '${base#http://}' is sent through the DMZ proxy because NO_PROXY only"
      echo "   covers 10.0.0.0/8 and .intel.com. Fixes:"
      echo "     • use the management host's LAN IP   (e.g. --mgmt-url http://10.111.56.26/api)"
      echo "     • or an FQDN under .intel.com"
      echo "     • or add the hostname to NO_PROXY in /etc/devozs-gpu-agent.env"
      exit 1
      ;;
    *)
      warn "Unexpected response — HTTP $http_code"
      echo "   Not the heartbeat's expected 400/401. Check the URL path (must end in /api)"
      echo "   and that it points at the management service, not another server."
      exit 1
      ;;
  esac
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  ok "Connectivity OK. Safe to continue with ./setup-agent.sh"
  exit 0
fi

cat <<EOF

Done. Config is written and the management URL is reachable.
Next:
  ./setup-agent.sh                  # driver + Habana venv + MNIST verify (HPU)
  # then install the agent into the venv, enroll once, and:
  ./deploy/install-service.sh       # reuses $ENV_DEST as-is

Re-check connectivity any time (reads MGMT_URL from $ENV_DEST):
  ./deploy/bootstrap.sh --check-only
EOF
