# SmolVLA custom training image

The default SageMaker path in `03_finetune_smolvla.ipynb` uses the native
PyTorch DLC returned by `image_uris.retrieve`. This image is the optional BYOC
path for preinstalling dependencies.

It is built from a digest-pinned NVIDIA CUDA 12.8 runtime + Ubuntu 24.04 image,
not from a SageMaker image. It
installs:

- Python 3.12 in `/opt/venv`;
- PyTorch 2.8, torchvision 0.23, and torchaudio 2.8 from the cu128 index;
- LeRobot/SmolVLA dependencies from `scripts/requirements.txt`;
- `sagemaker-training` and `sagemaker-pytorch-training`;
- the SageMaker-compatible `/opt/venv/bin/train` entrypoint.

It does not copy `scripts/train.py`, `scripts/args.yaml`, the dataset, or model
weights. SageMaker `SourceCode` and input channels provide those at runtime.

## Build and push

Run from the repository root so the build context contains
`scripts/requirements.txt`:

```bash
./container/create-image.sh \
  smolvla-training latest container/Dockerfile .
```

The image still requires a GPU smoke test after building. The Docker build
checks CPU-safe imports and package metadata but does not execute CUDA kernels,
NCCL collectives, a training step, or checkpoint reload.

With the custom image, keep `SourceCode` decoupled and omit its `requirements`
field because dependencies are already installed:

```python
source_code = SourceCode(
    source_dir="./scripts",
    entry_script="train.py",
)
```
