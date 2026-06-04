# devozs GPU Agent

A standalone GPU/HPU **training agent**. It runs on a registered compute box
(NVIDIA CUDA or Intel Gaudi/HPU), connects **outbound** to a management service
(enroll → heartbeat → claim → report), and fine-tunes an admin-selected
HuggingFace model on a project's dataset — reporting live progress with
cooperative stop and checkpoint-based resume.

Because every connection is outbound, the box needs **no inbound access** and
works behind NAT / a corporate proxy. The agent is **project-neutral**: the
management URL, resource type, dataset, and storage are all supplied by
configuration, so the same agent serves any project that speaks the training
agent protocol.

```
compute box (this agent)  ──HTTPS, outbound, X-Agent-Token──▶  management service
  CUDA | HPU backend                                            (enroll/heartbeat/claim/
  preflight → poll → train → report                              progress/log/checkpoint/
                                                                  complete/error/dataset)
```

## Install

```bash
# from git (recommended — pin a tag/branch as needed)
pip install 'devozs-gpu-agent[training] @ git+ssh://git@github.com/devozs/gpu-agent.git'

# or from a local checkout, editable
pip install -e '/path/to/gpu-agent[training]'
```

Extras:
- `training` — base ML stack: `transformers`, `datasets`, `boto3` (needed for a real job; imported lazily, so a box can enroll + pass preflight without them).
- `cuda` — adds a CUDA-capable `torch` (use the build matching your CUDA version).
- `gaudi` — adds `optimum-habana`. On Gaudi, `habana_frameworks` + a matched `torch` come from the **SynapseAI install on the box**, not pip — see below.

## Run

The agent reads everything from the environment, enrolls once (caching a bearer
token to disk), then polls.

```bash
ENROLL_CODE=<one-time-code> \
MGMT_URL=http://<mgmt-host>/api \
AGENT_TYPE=CUDA \
python -m devozs_gpu_agent
```

`AGENT_TYPE` is `CUDA` or `HPU`. The admin creates the resource in the UI, which
issues the one-time `ENROLL_CODE`; the agent redeems it for a per-agent token
(cached at `~/.devozs_gpu_agent/agent-token.json`) and never needs the code
again.

### Intel Gaudi VM (bare metal — recommended)

On Gaudi, the SynapseAI userspace (`habana_frameworks` + `torch`) must match the
host driver **exactly**, so run the agent in the same venv that runs Habana's
quick-start — not a generic container.

1. **Set up + verify the box** with the bundled script (mirrors
   [`gaudi-vm-setup.md`](gaudi-vm-setup.md)):

   ```bash
   ./setup-agent.sh             # sections 1 + 2: driver, venv, MNIST verify
   ./setup-agent.sh --with-hf   # also section 3: HF fine-tune verify (optional)
   ```

   Sections 1 + 2 are the **gate**: once they pass, the box is ready to be marked
   trainable in the UI. Section 3 (`--with-hf`) additionally proves the exact
   Hugging Face / optimum-habana path the agent's HPU backend uses.

2. **Install the agent into the Habana venv** without disturbing the matched
   torch, then run it:

   ```bash
   source ~/habanalabs-venv/bin/activate
   pip install --no-deps -e /path/to/gpu-agent                 # the agent + requests
   pip install 'datasets>=2.18' 'boto3>=1.34' optimum-habana   # only needed for a real job

   export PT_HPU_LAZY_MODE=1                                    # see gaudi-vm-setup.md §3.2
   ENROLL_CODE=<code> MGMT_URL=http://<mgmt-host>/api AGENT_TYPE=HPU \
     python -m devozs_gpu_agent
   ```

Preflight needs only `requests` + `transformers` + `habana_frameworks` (all
present after `setup-agent.sh`); `datasets`/`boto3`/`optimum-habana` are imported
lazily and only when a job actually runs, so the box can reach **READY** before
they're installed.

### Container (only if a published image matches your driver)

`Dockerfile` builds a CUDA or Gaudi image; see its header for build/run commands.
On Gaudi prefer bare metal — a base-image SynapseAI version that doesn't match
the host driver fails at device init (`synStatus=26 ... Device acquire failed`).

## Configuration (env vars)

| var | meaning | default |
|-----|---------|---------|
| `MGMT_URL` | management base URL incl. `/api`, e.g. `http://host:8080/api` | `http://localhost:8080/api` |
| `AGENT_TYPE` | `CUDA` or `HPU` | `CUDA` |
| `ENROLL_CODE` | one-time enrollment code (only needed until a token is cached) | — |
| `AGENT_NAME` | human name for the box; must match the resource the admin created | hostname |
| `POLL_INTERVAL` | seconds between heartbeat/claim polls | `5` |
| `AGENT_TOKEN_FILE` | where the bearer token is cached | `~/.devozs_gpu_agent/agent-token.json` |
| `AGENT_WORK_DIR` | local scratch for datasets/checkpoints | `/tmp/devozs-gpu-agent` |
| `GPU_AGENT_STUB=1` | no-ML dev mode: fake training + a pass-through preflight | off |
| `PREFLIGHT_MODEL` | validate a real model in preflight instead of the offline tiny-gpt2 | offline probe |
| `PT_HPU_LAZY_MODE` | Gaudi lazy/eager mode (`1`=lazy). Set `1` when using HPU graphs | `1` |
| `HF_TOKEN` | HuggingFace token, required only when a session has *push to Hub* on | — |
| `HF_HUB_NAMESPACE` | org/user prefix for the pushed model repo (used with push-to-Hub) | — |
| `TRAINING_S3_BUCKET` | S3 bucket for artifacts when not derivable from the dataset URI | — |
| `TRAINING_STORAGE_ROOT` | local-fs storage root for `file://` artifacts | `/data/devozs-gpu-agent` |
| `AWS_ENDPOINT_URL` | custom S3 endpoint (e.g. Nebius/MinIO) | — |

## Local dev (no GPU)

Stub mode exercises the whole protocol — enroll → preflight → claim → progress →
checkpoint → complete, plus cooperative stop and resume — without an accelerator:

```bash
GPU_AGENT_STUB=1 ENROLL_CODE=<code> MGMT_URL=http://localhost:8080/api \
  python -m devozs_gpu_agent
```

## Layout

```
devozs_gpu_agent/
  __main__.py            python -m devozs_gpu_agent → agent.run()
  agent.py               poll loop: enroll → preflight → heartbeat → claim → run
  config.py              env-driven AgentConfig + token cache
  management_client.py   HTTP client for the training-agent protocol (X-Agent-Token)
  preflight.py           readiness check: device identify + tiny real generate()
  runner.py              one job end-to-end: dataset → train/resume → publish → report
  dataset.py             JSONL → tokenized HF Dataset
  progress_callback.py   TrainerCallback → progress reports + cooperative stop + checkpoint
  storage_client.py      dataset/checkpoint/model transfer (S3 presigned | file:// | HTTPS)
  backends/
    base.py              TrainingBackend contract
    cuda_backend.py      transformers Trainer on NVIDIA
    hpu_backend.py       optimum-habana GaudiTrainer on Intel Gaudi
    stub_backend.py      no-ML fake trainer for dev
setup-agent.sh           Gaudi bare-metal setup + verify (gaudi-vm-setup.md §1–3)
gaudi-vm-setup.md        manual reference for the setup/verify steps
Dockerfile               optional CUDA/Gaudi container image
```

## Protocol compatibility

The agent speaks a fixed contract the management service defines. Do not change
these when adapting the agent:
- token header **`X-Agent-Token`** (not `Authorization: Bearer`);
- endpoints under **`/training/agent/...`** (`enroll`, `heartbeat`, `preflight`, `claim`, `sessions/{id}/{progress,log,checkpoint,complete,stopped,error,dataset}`);
- DTO field names as sent/parsed in [`management_client.py`](devozs_gpu_agent/management_client.py);
- only claim when readiness is `READY`; claim only jobs matching the resource type.
