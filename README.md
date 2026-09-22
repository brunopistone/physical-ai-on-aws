# Physical AI on AWS: SO100 + SmolVLA Workshop

This workshop implements a complete Physical AI data loop:

```text
scripted teacher
      │
      ▼
simulation demonstrations ──▶ zero-shot VLA baseline ──▶ fine-tuning ──▶ before/after evaluation
        Notebook 1                 Notebook 2              Notebook 3          Notebook 4
```

The reference task is:

> Pick up the cube and place it in the box.

The workshop starts in MuJoCo, records a LeRobot dataset, demonstrates why a
checkpoint trained on real-camera data can fail under simulation domain shift,
fine-tunes SmolVLA with a SageMaker Training Job, and evaluates the updated
checkpoint under the same physical conditions.

## What You Will Learn

- The difference between a **model** and a **policy**.
- How a VLA maps images, language, and robot state to actions.
- Why embodiment, camera geometry, units, and task distribution matter.
- How to generate successful demonstrations with a privileged scripted teacher.
- How LeRobot stores video, state, action, task, and episode boundaries.
- How to launch the same training entrypoint locally, on SageMaker, EKS, or HyperPod.
- How to compare two policies using a physical task metric rather than API status.

## Workshop Notebooks

| Notebook                                                 | Purpose                                                                              | Main output                                  |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------- |
| [`01_scripted_pick.ipynb`](01_scripted_pick.ipynb)       | Build the SO100 simulation, run a scripted teacher, and record demonstrations        | `datasets/so100_sim_pickplace`               |
| [`02_smolvla_pick.ipynb`](02_smolvla_pick.ipynb)         | Run the original real-data SmolVLA checkpoint in MuJoCo                              | Zero-shot video and `placed_in_box` baseline |
| [`03_finetune_smolvla.ipynb`](03_finetune_smolvla.ipynb) | Generate `args.yaml`, upload inputs, and launch a SageMaker Training Job             | Fine-tuned SmolVLA checkpoint in S3          |
| [`04_evaluate_smolvla.ipynb`](04_evaluate_smolvla.ipynb) | Find the latest completed job, download its model, and compare before/after rollouts | Comparison table, metrics, and two videos    |

Run the notebooks in order. Notebook 4 deliberately reruns the baseline instead
of assuming that a successful inference call means task success.

## Reference Configuration

| Component           | Workshop value                                       |
| ------------------- | ---------------------------------------------------- |
| Robot               | SO100                                                |
| Runtime             | MuJoCo simulation through `strands-robots`           |
| Learned model       | SmolVLA, approximately 450M parameters               |
| Starting checkpoint | `lucarrr/smolvla_so100_pickplace_finetuned_v2`       |
| Checkpoint revision | `4fde42badae91b5c88bfe1a399a3f4b994fc0698`           |
| Instruction         | `Pick up the cube and place it in the box.`          |
| Cameras             | `observation.images.top`, `observation.images.wrist` |
| State               | Six SO100 joint values                               |
| Action              | Six SO100 joint commands                             |
| Simulation units    | Radians                                              |
| Dataset frequency   | 30 FPS                                               |
| Dataset format      | LeRobotDataset: Parquet + H.264 MP4 + metadata       |
| Success metric      | `placed_in_box`                                      |

The real-data checkpoint originally expects `camera1`, `camera2`, and
`camera3`. The training entrypoint preserves the semantic order
`camera1=wrist`, `camera2=top`, removes the unused third camera, and exports a
checkpoint with the simulation-native `wrist` and `top` feature names.

## Quick Start

Requirements:

- Python 3.12.
- Apple Silicon or Linux for local simulation/inference.
- AWS credentials and SageMaker permissions for Notebooks 3 and 4.
- Network access to Hugging Face for the public SmolVLA checkpoints.
- Sufficient AWS quota for the selected training instance.

Start Jupyter from the repository root:

```bash
cd /path/to/physical-ai-on-aws
jupyter lab
```

The notebooks install their pinned dependencies. To prepare the VLA runtime
manually:

```bash
python -m pip install -r requirements.txt
```

Validate the training entrypoint without downloading a model or starting
training:

```bash
python scripts/train.py \
  --config scripts/args.yaml \
  --dry-run
```

## The Data Contract

The current trainer intentionally fails when the dataset does not match this
contract:

```text
robot_type: so100
fps: 30

observation.state
  shape: [6]
  names: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw

action
  shape: [6]
  names: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw

observation.images.top
observation.images.wrist
```

A finalized dataset must contain:

```text
dataset-root/
├── data/
│   └── ... parquet frame shards
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/
└── videos/
    ├── observation.images.top/
    └── observation.images.wrist/
```

For multiple demonstrations, the recording lifecycle is:

```text
start_recording
├── run_policy → save_episode
├── run_policy → save_episode
└── run_policy → save_episode
stop_recording
```

`save_episode()` is essential. Without it, multiple rollouts become one long
episode.

## Which Models Are Supported?

### Supported as implemented

The current training entrypoint supports:

- a **pretrained SmolVLA checkpoint**;
- LeRobot demonstration data;
- the SO100 6D state/action embodiment;
- `top` and `wrist` image streams;
- supervised imitation learning through SmolVLA's flow-matching objective.

It does not train the entire 450M-parameter model. The default configuration:

- freezes the vision encoder;
- trains the action expert;
- trains the state projection;
- leaves the main VLM backbone frozen.

This corresponds to approximately 100M trainable parameters for the reference
checkpoint. It is neither LoRA nor text SFT.

### Other SmolVLA checkpoints

Another SmolVLA checkpoint can be used by changing:

```yaml
model:
  id: organization/checkpoint
  revision: pinned-commit
  camera_mapping:
    observation.images.wrist: checkpoint-camera-key
    observation.images.top: checkpoint-camera-key
```

This works only when:

- the checkpoint is a LeRobot SmolVLA checkpoint with `config.json` and
  `model.safetensors`;
- its state/action dimensions are compatible with the dataset;
- the camera mapping is explicit and semantically correct;
- the processor and normalization contract can be regenerated from the new
  dataset.

The script does not initialize SmolVLA from scratch.

### Models not supported by the current trainer

| Model or policy family                    | Current support | What would be required                                                  |
| ----------------------------------------- | --------------- | ----------------------------------------------------------------------- |
| SmolVLA + SO100 matching schema           | Yes             | New data/config may be sufficient                                       |
| SmolVLA + another task, same SO100 schema | Yes             | New demonstrations, instruction, and metric                             |
| SmolVLA + SO101 or another arm            | Not drop-in     | New embodiment, state/action schema, scene, and dataset                 |
| SmolVLA + ALOHA                           | No              | 14D bimanual dataset, ALOHA adapter, new teacher/evaluation             |
| ACT                                       | No              | ACT policy configuration and training path                              |
| Diffusion Policy                          | No              | Diffusion-specific configuration and action semantics                   |
| π0 / π0.5                                 | No              | Different checkpoint, processor, action head, and hardware requirements |
| OpenVLA / GR00T                           | No              | Different model runtime and fine-tuning implementation                  |
| Text LLM SFT, DPO, or GRPO                | No              | Use a text/TRL training entrypoint instead                              |
| Reinforcement learning                    | No              | Environment interaction, reward, and RL training loop                   |

The infrastructure interface is generic; the model adapter and training
objective are SmolVLA-specific.

## Which Use Cases Can Be Covered?

The current stack can support other instruction-conditioned SO100 tasks when
the state/action and camera contract stays unchanged.

| Use case                            | Required changes                                                          |
| ----------------------------------- | ------------------------------------------------------------------------- |
| Pick objects at different positions | Add successful demonstrations covering the new workspace                  |
| Pick different colors or shapes     | Vary object appearance, instruction, and positions                        |
| Place objects in different bins     | Add multiple targets and task-specific success metrics                    |
| Color-based sorting                 | Record multiple instructions and balanced examples per class/target       |
| Object relocation                   | Change start/goal distributions and verify collision-free teacher motions |
| Push or slide                       | Replace the teacher trajectory and define a planar goal metric            |
| Stacking                            | Add a second object, longer trajectories, and a stable-stack metric       |
| Simple insertion                    | Add tighter pose coverage and contact-sensitive success checks            |
| Multi-task manipulation             | Store multiple task strings in one schema-compatible dataset              |
| Sim-to-real adaptation              | Collect real SO100 demonstrations with calibrated cameras and units       |

Tasks such as insertion, stacking, tool use, deformable objects, or long-horizon
sequences generally need substantially more data and a more capable teacher.
The fact that the model accepts an instruction does not guarantee that the
dataset contains enough evidence to learn it.

## Which Datasets Can Be Used?

### Workshop-generated simulation datasets

The easiest compatible dataset is one produced by Notebook 1 or the same
`strands-robots` recorder:

- SO100;
- 30 FPS;
- `top` and `wrist`;
- six named state/action dimensions;
- actions and state recorded in MuJoCo radians;
- one or more language tasks.

The current eight-episode dataset validates the engineering path. It is too
small for strong model-quality conclusions. Use at least 50 successful
demonstrations before treating the result as a meaningful adaptation.

### Real SO100 datasets

The starting checkpoint was trained on:

```text
lerobot/svla_so100_pickplace
```

That dataset is useful as a real-world reference, but it is not automatically
compatible with the workshop trainer: its camera names and motor units differ
from the simulation schema. A real dataset must be converted deliberately, or
the validator, camera adapter, unit conversion, and exported manifest must be
updated together.

Do not label degree-valued data as radians or rename cameras without preserving
their physical meaning.

### Other LeRobot datasets

A local or Hugging Face LeRobot dataset can be used after it is materialized
into a complete local dataset tree or SageMaker input channel. Check:

1. robot and embodiment;
2. state and action dimensions;
3. joint ordering and names;
4. action semantics: absolute, relative, velocity, or delta;
5. numeric units and gripper representation;
6. camera count, names, viewpoint, resolution, and ordering;
7. FPS and action chunk timing;
8. language task fields;
9. episode boundaries;
10. normalization statistics.

Matching tensor shapes is not sufficient. Two datasets may both contain six
numbers while assigning different joints, units, or control semantics to them.

### Incompatible examples

- ALOHA datasets with 14 bimanual actions.
- Datasets without the required camera streams.
- SO100 data stored in degrees while the exported policy is configured for
  radians.
- Relative-action datasets used as absolute commands.
- Datasets whose cameras are renamed by position rather than physical role.
- Pushing or navigation datasets with a different action meaning.

These datasets can still be useful, but require a corresponding embodiment and
model/training adapter.

## Training Configuration

Notebook 3 generates `args.yaml` explicitly and uploads it through the
SageMaker `config` channel. The sample configuration is also available at
[`scripts/args.yaml`](scripts/args.yaml).

Main sections:

```yaml
model: # checkpoint, revision, frozen/trainable components, camera mapping
dataset: # LeRobot repository identity, decoder, evaluation split
training: # steps, batch, optimizer schedule, checkpoint and evaluation cadence
tracking: # optional experiment tracking
paths: # local defaults or mounted/container paths
```

Resolution order is:

```text
CLI override → environment/SageMaker channel → args.yaml
```

For the SageMaker PyTorch 2.8 DLC, use:

```yaml
training:
  use_amp: false
```

The model already uses BF16 parameters. Enabling the current AMP/GradScaler
path causes BF16 gradient unscale to fail on that runtime.

## Training Environments

| Environment            | How the script is launched                    | Storage contract                                       |
| ---------------------- | --------------------------------------------- | ------------------------------------------------------ |
| Local                  | `python` or `torchrun scripts/train.py`       | Local dataset/work/model directories                   |
| SageMaker Training Job | `ModelTrainer` + `SourceCode` + `Torchrun`    | Input channels, `/opt/ml/checkpoints`, `/opt/ml/model` |
| EKS                    | Kubernetes Job or PyTorch operator            | PVC, FSx, or Mountpoint for Amazon S3                  |
| HyperPod EKS           | HyperPod PyTorch job or Kubernetes scheduling | Shared mounted dataset/checkpoints                     |
| HyperPod Slurm         | `srun`/`torchrun`                             | Shared FSx or another shared filesystem                |

The training logic is the same. Launch configuration, networking, storage, and
distributed rendezvous remain platform-specific.

## SageMaker Training

Notebook 3 uses the native SageMaker PyTorch DLC by default:

```python
image_uri = image_uris.retrieve(
    framework="pytorch",
    version="2.8.0",
    instance_type=instance_type,
    image_scope="training",
)
```

`SourceCode` packages `scripts/train.py` independently from the image. Dataset
and YAML configuration are separate input channels.

The optional custom image is built from a digest-pinned NVIDIA CUDA/Ubuntu
runtime rather than from a SageMaker image. It installs Python, PyTorch,
LeRobot, and the SageMaker training toolkits, but does not copy:

- `train.py`;
- `args.yaml`;
- datasets;
- model weights;
- credentials.

Build and push it from the repository root:

```bash
./container/create-image.sh \
  smolvla-training latest container/Dockerfile .
```

See [`container/README.md`](container/README.md) for details and validation
limits.

## Evaluation

Notebook 4:

1. finds the latest completed SageMaker Training Job by base-job prefix;
2. obtains the model root from
   `DescribeTrainingJob.ModelArtifacts.S3ModelArtifacts`;
3. downloads the `smolvla/` export while ignoring SageMaker marker objects;
4. validates the checkpoint schema and training manifest;
5. reruns the zero-shot baseline;
6. releases the first model from memory;
7. runs the fine-tuned checkpoint in a fresh scene;
8. compares videos, final cube positions, and `placed_in_box`.

`run_policy status="success"` means the software loop completed. It does not
mean that the robot completed the task.

## Repository Layout

```text
.
├── 01_scripted_pick.ipynb
├── 02_smolvla_pick.ipynb
├── 03_finetune_smolvla.ipynb
├── 04_evaluate_smolvla.ipynb
├── requirements.txt
├── code/
│   ├── so100_teacher.py
│   └── vla_pick.py
├── scripts/
│   ├── train.py
│   ├── args.yaml
│   └── requirements.txt
├── container/
│   ├── Dockerfile
│   ├── create-image.sh
│   └── README.md
└── datasets/
    └── so100_sim_pickplace/
```

Generated datasets, MP4 files, downloaded model weights, checkpoints, and
temporary `args.yaml` files should not be committed.

## Common Warnings and Failures

### `IProgress not found`

This only changes the notebook progress-bar presentation. Install
`ipywidgets` if desired.

### TorchCodec cannot load FFmpeg

LeRobot can fall back to PyAV. The fallback is acceptable when:

- `video_backend` is `pyav`;
- MP4 shards exist;
- frames decode during training.

### NCCL/OFI warnings on a single GPU

The runtime may fail to initialize the EFA plugin and fall back to NCCL Socket.
If NCCL reaches `Init COMPLETE`, this warning is not the training failure.

### BF16 AMP unscale failure

```text
_amp_foreach_non_finite_check_and_unscale_cuda
not implemented for 'BFloat16'
```

Set `training.use_amp: false` for the current PyTorch 2.8 SageMaker DLC.

### Inference runs but the task fails

Inspect:

- `placed_in_box`;
- final cube position;
- both camera streams;
- state/action unit conversion;
- camera mapping;
- the rollout video.

Model loading is not evidence of learned task success.

## Extending the Workshop

For a new task or dataset:

1. define an observable physical success metric;
2. keep robot, cameras, units, and action semantics explicit;
3. implement or teleoperate a teacher that succeeds reliably;
4. collect enough successful and varied episodes;
5. validate the finalized dataset artifact;
6. run a zero-shot baseline;
7. fine-tune from a pinned checkpoint;
8. evaluate before and after in fresh, identical scenes;
9. inspect failures before adding model complexity;
10. add hardware safety, limits, and supervision before any real-robot run.

## Limitations

- The scripted teacher reads ground-truth MuJoCo object pose. This is privileged
  simulation supervision, not real-robot perception.
- Eight episodes validate the pipeline, not generalization.
- The current trainer is SmolVLA/SO100-specific.
- The workshop does not implement reinforcement learning, DPO, GRPO, or text
  SFT.
- The default evaluation is simulation-only.
- Nothing in this repository is a real-robot safety controller.
