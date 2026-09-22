# syntax=docker/dockerfile:1
#
# Build context: repository root.
# This is a SageMaker-compatible BYOC image built from CUDA/Ubuntu, not from a
# SageMaker DLC. It installs runtime dependencies only. ModelTrainer.SourceCode
# supplies scripts/train.py at job submission time; args.yaml and the dataset
# remain input channels.

ARG BASE_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:828c4d878adcaa4265d80c95d8ec877149b49bb2419a4cf3bb6aa889bbb7ca2e
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ARG PYTORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCHAUDIO_VERSION=2.8.0
ARG SAGEMAKER_TRAINING_VERSION=5.1.1
ARG SAGEMAKER_PYTORCH_TRAINING_VERSION=3.0.0

ENV VIRTUAL_ENV=/opt/venv \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_CONSTRAINT=/opt/smolvla-build/image-constraints.txt

RUN . /etc/os-release \
    && test "$ID" = ubuntu \
    && test "$VERSION_ID" = "24.04" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && mkdir -p /opt/smolvla-build /opt/ml/code \
    && touch /opt/smolvla-build/image-constraints.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==${PYTORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"

# Protect the CUDA-enabled torch stack from being replaced while resolving
# LeRobot and the SageMaker toolkits.
RUN python - <<'PY'
import importlib.metadata as md
from pathlib import Path

packages = ("torch", "torchvision", "torchaudio", "triton")
pins = [f"{name}=={md.version(name)}" for name in packages]
Path("/opt/smolvla-build/image-constraints.txt").write_text(
    "\n".join(pins) + "\n"
)
print("\n".join(pins))
PY

# Dependency-only copy. Training source, YAML, data, and model weights are not
# part of the image and cannot become stale when those artifacts change.
COPY scripts/requirements.txt /opt/smolvla-build/requirements.txt

RUN python -m pip install \
    "sagemaker-training==${SAGEMAKER_TRAINING_VERSION}" \
    "sagemaker-pytorch-training==${SAGEMAKER_PYTORCH_TRAINING_VERSION}" \
    && python -m pip install -r /opt/smolvla-build/requirements.txt \
    && python -m pip check \
    && python -m pip freeze --all > /opt/smolvla-build/resolved-environment.txt

# CPU-safe build validation. CUDA kernels and distributed collectives still
# require a separate smoke test on the target GPU instance.
RUN python - <<'PY'
import importlib.metadata as md
import sys

import accelerate
import av
import lerobot
import torch
import transformers
import yaml

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.startswith("2.8.0"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert md.version("lerobot") == "0.6.1"
entry = next(
    item
    for item in md.distribution("sagemaker-training").entry_points
    if item.name == "train"
)
assert entry.value == "sagemaker_training.cli.train:main", entry
assert callable(entry.load())
print(
    "Validated",
    {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "lerobot": md.version("lerobot"),
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "pyav": av.__version__,
    },
)
PY

ENV SAGEMAKER_TRAINING_MODULE=sagemaker_pytorch_container.training:main
WORKDIR /opt/ml/code
ENTRYPOINT ["/opt/venv/bin/train"]
CMD []
