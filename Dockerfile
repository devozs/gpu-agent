# devozs GPU/HPU training agent image.
#
# The agent connects OUTBOUND to management (enroll/heartbeat/claim/report), so a
# container just needs Python + this package + the right accelerator stack.
# Parameterized by base image so one file builds both targets:
#
#   CUDA laptop / GPU host:
#     docker build -f Dockerfile \
#       --build-arg BASE_IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime \
#       --build-arg EXTRAS=training,cuda -t devozs-gpu-agent:cuda .
#
#   Intel Gaudi VM (use Habana's base image so habana_frameworks + the matched
#   SynapseAI driver are present; pick the tag matching your driver — browse
#   https://vault.habana.ai/ui/native/gaudi-docker):
#     docker build -f Dockerfile \
#       --build-arg BASE_IMAGE=vault.habana.ai/gaudi-docker/<synapse>/ubuntu22.04/habanalabs/pytorch-installer-<pt>:latest \
#       --build-arg EXTRAS=training,gaudi -t devozs-gpu-agent:gaudi .
#
# Run (env supplies the one-time enrollment code + management URL):
#   docker run --rm --gpus all \                 # CUDA
#     -e ENROLL_CODE=... -e MGMT_URL=http://<mgmt-host>:8080/api -e AGENT_TYPE=CUDA \
#     devozs-gpu-agent:cuda
#   docker run --rm --runtime=habana \           # Gaudi
#     -e HABANA_VISIBLE_DEVICES=all -e ENROLL_CODE=... -e MGMT_URL=... -e AGENT_TYPE=HPU \
#     devozs-gpu-agent:gaudi
#
# NOTE: bare-metal install (setup-agent.sh) is recommended on Gaudi over a
# container, because the SynapseAI userspace must match the host driver exactly.

ARG BASE_IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

# Which setup.py extras to install: training[,cuda] or training[,gaudi].
ARG EXTRAS=training

WORKDIR /app

# Install only what the agent needs (the package + selected extras). Copy the
# minimum so a code change doesn't bust the dependency layer unnecessarily.
COPY setup.py README.md ./
COPY devozs_gpu_agent ./devozs_gpu_agent

# -v + default progress bar so `docker build --progress=plain` streams what pip
# is collecting/downloading (the Gaudi deps are large and slow over a proxy, so
# the build can look stuck for many minutes without this).
RUN pip install --no-cache-dir -v --progress-bar on ".[${EXTRAS}]"

# Token cache + agent scratch live here; mount a volume to persist the token
# across restarts so the box doesn't need re-enrolling.
ENV AGENT_TOKEN_FILE=/agent/agent-token.json \
    AGENT_WORK_DIR=/agent/work
VOLUME ["/agent"]

ENTRYPOINT ["python", "-m", "devozs_gpu_agent"]
