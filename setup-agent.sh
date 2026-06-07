#!/usr/bin/env bash
#
# setup-agent.sh — bring a fresh Intel Gaudi VM up to a verified PyTorch + HPU
# environment for the devozs GPU agent, then (optionally) verify the Hugging Face
# fine-tune path.
#
# Mirrors gaudi-vm-setup.md:
#   Sections 1 + 2  ALWAYS  — driver + PyTorch stack + bare-metal verification
#                             (MNIST). This is the gate that must pass before an
#                             admin allows training on the resource in the UI.
#   Section 3       OPTIONAL — Hugging Face / optimum-habana single-card
#                             fine-tune (run_glue.py on MRPC). Use --with-hf when
#                             you want to confirm the exact HF training path the
#                             agent uses before pointing real jobs at the box.
#
# Usage:
#   ./setup-agent.sh                 # sections 1 + 2 (driver, venv, MNIST)
#   ./setup-agent.sh --with-hf       # also section 3 (HF run_glue.py verify)
#   ./setup-agent.sh --hf-only       # only section 3 (env already set up)
#   ./setup-agent.sh --skip-driver   # skip 1.1/1.2 install, still verify + MNIST
#   ./setup-agent.sh -y              # quiet/non-interactive: no approval prompts
#
# -y / --yes / --quiet: answer yes to every prompt so the run is unattended —
#   passes -y to the Habana installer, sets DEBIAN_FRONTEND=noninteractive for the
#   apt step, and pre-authenticates sudo once up front (it still asks for the sudo
#   password ONCE at the start unless the user has NOPASSWD; everything after that
#   runs without stopping to ask).
#
# Env overrides:
#   SYNAPSE_VERSION   SynapseAI version to install      (default 1.24.0)
#   VENV              Habana venv path                  (default ~/habanalabs-venv)
#   OPTIMUM_HABANA    optimum-habana version for §3     (default 1.21.0)
#   TRANSFORMERS      transformers pin (matches OH)      (default >=4.55,<4.56)
#
# This script is idempotent-ish: re-running skips clones/installs that already
# exist. It is intentionally bare-metal (no Docker) — on Gaudi the SynapseAI
# userspace must match the host driver exactly, which a container can break.

set -euo pipefail

# --- config -----------------------------------------------------------------
SYNAPSE_VERSION="${SYNAPSE_VERSION:-1.24.0}"
VENV="${VENV:-$HOME/habanalabs-venv}"
# optimum-habana must match the SynapseAI/Gaudi release: v1.21.0 <-> Gaudi 1.24,
# v1.20.0 <-> 1.23, v1.19.x <-> 1.22. optimum-habana monkey-patches transformers
# per-model against ONE pinned transformers window — so transformers must also be
# pinned to what that OH release declares, or a model's Gaudi forward override
# gets kwargs it never declared (e.g. cache_position) and dies with a TypeError.
OPTIMUM_HABANA="${OPTIMUM_HABANA:-1.21.0}"
# transformers window OH 1.21.0 pins in its setup.py.
TRANSFORMERS="${TRANSFORMERS:-transformers>=4.55.0,<4.56.0}"
INSTALLER="habanalabs-installer.sh"

WITH_HF=0
HF_ONLY=0
SKIP_DRIVER=0
YES=0

for arg in "$@"; do
  case "$arg" in
    --with-hf)        WITH_HF=1 ;;
    --hf-only)        HF_ONLY=1; WITH_HF=1 ;;
    --skip-driver)    SKIP_DRIVER=1 ;;
    -y|--yes|--quiet) YES=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# Non-interactive plumbing. The Habana installer takes -y; apt honours
# DEBIAN_FRONTEND=noninteractive (passed through to the sudo'd dependencies step).
INSTALLER_YES=""
if [ "$YES" -eq 1 ]; then
  INSTALLER_YES="-y"
  export DEBIAN_FRONTEND=noninteractive
fi

log()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }

# ===========================================================================
# SECTION 1 — Install the driver + software
# ===========================================================================
section_1_install() {
  log "1.1 Install driver and base software (SynapseAI ${SYNAPSE_VERSION})"
  if [ ! -f "$INSTALLER" ]; then
    wget -nv "https://vault.habana.ai/artifactory/gaudi-installer/${SYNAPSE_VERSION}/${INSTALLER}"
    chmod +x "$INSTALLER"
  fi
  ./"$INSTALLER" install --type base $INSTALLER_YES

  log "1.2 Install OS dependencies, then the PyTorch stack (torch + habana_frameworks)"
  # The `base` type installs only the driver/firmware — NOT PyTorch. The
  # `pytorch` install ABORTS (and does not create the venv) if system libs are
  # missing, so install `dependencies` first. Needs sudo, and runs apt under the
  # hood — DEBIAN_FRONTEND keeps apt from prompting in quiet mode.
  sudo DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-}" ./"$INSTALLER" install --type dependencies $INSTALLER_YES
  ./"$INSTALLER" install --type pytorch --venv $INSTALLER_YES   # creates $VENV with torch + habana_frameworks
}

# ===========================================================================
# SECTION 1.3 — Verify driver + torch
# ===========================================================================
section_1_verify() {
  log "1.3 Verify driver + torch"
  [ -d "$VENV" ] || die "venv $VENV not found — did §1.2 (pytorch --venv) run?"
  command -v hl-smi >/dev/null || die "hl-smi not found — driver install (§1.1) incomplete"
  hl-smi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -c 'import torch, habana_frameworks.torch.core as h; print("OK torch", torch.__version__)' \
    || die "torch / habana_frameworks import failed inside the venv"
}

# ===========================================================================
# SECTION 2 — Quick start on bare metal (hello_world + MNIST)
# ===========================================================================
section_2_quickstart() {
  log "2.1 Download PyTorch Model-References + run hello_world example"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  [ -d "$HOME/Model-References" ] || git clone https://github.com/HabanaAI/Model-References.git "$HOME/Model-References"

  local HW="$HOME/Model-References/PyTorch/examples/computer_vision/hello_world"
  cd "$HW"
  export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
  export PYTHONPATH="${PYTHONPATH:-}:$HOME/Model-References"
  export PYTHON="$VENV/bin/python"
  "$PYTHON" --version

  # example.py saves a checkpoint to ./checkpoints after training — create it
  # first or the run ends with "Parent directory ./checkpoints does not exist".
  mkdir -p "$HW/checkpoints"
  "$PYTHON" example.py

  log "2.2 Training example — single Gaudi card (MNIST)"
  # MNIST quick-start runs in eager mode.
  PT_HPU_LAZY_MODE=0 "$PYTHON" mnist.py \
    --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu --autocast --use-torch-compile
  log "Sections 1 + 2 PASSED — the box is ready to be marked trainable in the UI."
}

# ===========================================================================
# SECTION 3 — OPTIONAL: Hugging Face fine-tune verification (single card)
# ===========================================================================
section_3_hf() {
  log "3.1 Setup optimum-habana ${OPTIMUM_HABANA} (HF fine-tune verification)"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  # PYTHONNOUSERSITE=1 keeps installs inside the venv; otherwise they leak into
  # ~/.local (a generic CUDA torch) and run_glue.py crashes in
  # check_synapse_version() with "IndexError: list index out of range".
  # Version match: optimum-habana v1.21.0 <-> SynapseAI 1.24, v1.20.0 <-> 1.23,
  # v1.19.x <-> 1.22 (per the OH release notes). A bare `git clone` lands the
  # default branch (currently 1.19.x, built for the PREVIOUS Gaudi release), so
  # we MUST checkout the matching tag — hence the explicit v${OPTIMUM_HABANA}.
  [ -d "$HOME/optimum-habana" ] || git clone https://github.com/huggingface/optimum-habana.git "$HOME/optimum-habana"
  ( cd "$HOME/optimum-habana" && git fetch --tags --quiet && git checkout "v${OPTIMUM_HABANA}" )

  # Pin transformers to OH's declared window in the SAME pip call so its resolver
  # can't pull a newer transformers that breaks the per-model Gaudi forward patch.
  PYTHONNOUSERSITE=1 pip install "optimum-habana==${OPTIMUM_HABANA}" "${TRANSFORMERS}"
  PYTHONNOUSERSITE=1 pip install -r "$HOME/optimum-habana/examples/text-classification/requirements.txt"

  log "3.2 Single-card training (run_glue.py, BERT-large on MRPC)"
  # SynapseAI 1.24 defaults to EAGER. The env var — not --use_lazy_mode — picks
  # the real mode, and --use_hpu_graphs_for_inference needs lazy mode at eval or
  # it dies with "HPUGraph class is available in lazy mode only." Force lazy:
  export PT_HPU_LAZY_MODE=1
  cd "$HOME/optimum-habana/examples/text-classification"
  PYTHONNOUSERSITE=1 python run_glue.py \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --gaudi_config_name Habana/bert-large-uncased-whole-word-masking \
    --task_name mrpc \
    --do_train --do_eval \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length 128 \
    --output_dir ./output/mrpc/ \
    --use_habana --use_lazy_mode --bf16 \
    --use_hpu_graphs_for_inference --throughput_warmup_steps 3
  log "Section 3 PASSED — the HF / optimum-habana fine-tune path works on this box."
}

# --- run --------------------------------------------------------------------
if [ "$HF_ONLY" -eq 1 ]; then
  section_3_hf
  exit 0
fi

if [ "$SKIP_DRIVER" -eq 0 ]; then
  # Pre-authenticate sudo up front so quiet runs don't stall on the password
  # prompt mid-install (the dependencies step needs sudo). Asks once now, or not
  # at all under NOPASSWD; harmless when sudo is already cached.
  if [ "$YES" -eq 1 ]; then
    log "Caching sudo credentials up front (needed for the dependencies step)"
    sudo -v || die "sudo authentication failed — required for --type dependencies"
  fi
  section_1_install
else
  warn "--skip-driver: assuming the SynapseAI driver + venv are already installed"
fi
section_1_verify
section_2_quickstart

if [ "$WITH_HF" -eq 1 ]; then
  section_3_hf
else
  log "Skipping section 3 (HF verify). Re-run with --with-hf to verify the Hugging Face path."
fi

cat <<EOF

Next: install the agent into this same Habana venv and run it.

  source "$VENV/bin/activate"
  pip install --no-deps -e /path/to/gpu-agent            # the agent package
  pip install 'datasets>=2.18' 'boto3>=1.34' optimum-habana   # only for a real job

  export PT_HPU_LAZY_MODE=1
  ENROLL_CODE=... MGMT_URL=http://<mgmt-host>/api AGENT_TYPE=HPU \\
    python -m devozs_gpu_agent

See README.md for details.
EOF
