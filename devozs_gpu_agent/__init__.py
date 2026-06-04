"""devozs GPU/HPU training agent.

A lightweight worker that runs on a registered GPU (CUDA) or Intel Gaudi (HPU)
box. It connects OUTBOUND to a management service (enroll/heartbeat/claim/
report), so it works behind NAT or a corporate proxy without management ever
dialing the box. It fine-tunes an admin-selected HuggingFace model on a project's
dataset and reports live progress, supporting cooperative stop and
checkpoint-based resume.

Project-neutral: the management URL, resource type, and storage are all supplied
by configuration, so the same agent serves any project that speaks the training
agent protocol.

Run it with:  ``python -m devozs_gpu_agent``
"""

__version__ = "0.1.0"
