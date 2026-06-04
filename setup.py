from setuptools import find_packages, setup

setup(
    name="devozs-gpu-agent",
    version="0.1.0",
    description="Standalone GPU/HPU training agent (CUDA + Intel Gaudi) for the training-agent protocol",
    packages=find_packages(include=["devozs_gpu_agent", "devozs_gpu_agent.*"]),
    python_requires=">=3.8",
    install_requires=[
        "requests",
    ],
    extras_require={
        # Base ML stack for a real (non-stub) job. transformers + datasets are
        # imported only when a job runs, so a box can enroll + pass preflight
        # before these are installed.
        "training": [
            "transformers>=4.40",
            "datasets>=2.18",
            "boto3>=1.34",
        ],
        # NVIDIA GPU: a CUDA-capable torch (pick the build matching your CUDA).
        "cuda": [
            "torch>=2.1",
        ],
        # Intel Gaudi: optimum-habana. habana_frameworks + a matched torch come
        # from the SynapseAI install on the box (NOT pip-installed here) — see
        # gaudi-vm-setup.md / setup-agent.sh.
        "gaudi": [
            "optimum-habana>=1.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "devozs-gpu-agent=devozs_gpu_agent.agent:run",
        ],
    },
    url="https://github.com/devozs/gpu-agent",
    author="Devozs Ltd.",
    author_email="ozishemesh@gmail.com",
)
