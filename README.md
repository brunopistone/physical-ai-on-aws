# Physical AI on AWS: SO100 + SmolVLA Workshop

This workshop implements a complete Physical AI data loop:

```text
scripted teacher
      │
      ▼
simulation demonstrations ─▶ zero-shot baseline ─▶ fine-tuning ─▶ evaluation ─▶ agentic supervision
        Notebook 1              Notebook 2          Notebook 3     Notebook 4        Notebook 5
```

The reference task is:

> Pick up the cube and place it in the box.

The workshop starts in MuJoCo, records a LeRobot dataset, demonstrates why a
checkpoint trained on real-camera data can fail under simulation domain shift,
fine-tunes SmolVLA with a SageMaker Training Job, evaluates the updated
checkpoint under the same physical conditions, and adds a bounded Strands Agent
above the learned policy.

## Table of Contents

- [What You Will Learn](#what-you-will-learn)
- [Workshop Notebooks](#workshop-notebooks)
- [Simulation Scenario Architecture](#simulation-scenario-architecture)
  - [The `SimulationScenario` contract](#the-simulationscenario-contract)
  - [`code/scenarios/pick_place.py`](#codescenariospick_placepy)
  - [`code/so100_teacher.py`](#codeso100_teacherpy)
  - [`code/vla_pick.py`](#codevla_pickpy)
  - [`code/agentic_pick.py`](#codeagentic_pickpy)
  - [Add another scenario](#add-another-scenario)
- [Reference Configuration](#reference-configuration)
- [Quick Start](#quick-start)
- [The Data Contract](#the-data-contract)
- [Which Models Are Supported?](#which-models-are-supported)
- [Which Use Cases Can Be Covered?](#which-use-cases-can-be-covered)
- [Which Datasets Can Be Used?](#which-datasets-can-be-used)
- [Training Configuration](#training-configuration)
- [Training Environments](#training-environments)
- [SageMaker Training](#sagemaker-training)
- [Evaluation](#evaluation)
- [Agentic Supervision](#agentic-supervision)
- [Repository Layout](#repository-layout)
- [Common Warnings and Failures](#common-warnings-and-failures)
- [Extending the Workshop](#extending-the-workshop)
- [Limitations](#limitations)

## What You Will Learn

- The difference between a **model** and a **policy**.
- How a VLA maps images, language, and robot state to actions.
- Why embodiment, camera geometry, units, and task distribution matter.
- How to generate successful demonstrations with a privileged scripted teacher.
- How LeRobot stores video, state, action, task, and episode boundaries.
- How to launch the same training entrypoint locally, on SageMaker, EKS, or HyperPod.
- How to compare two policies using a physical task metric rather than API status.
- How a local Strands Agent maps a free-form goal to validated robot skills
  without becoming the joint controller.

## Workshop Notebooks

| Notebook                                                 | Purpose                                                                              | Main output                                  |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------- |
| [`01_scripted_pick.ipynb`](01_scripted_pick.ipynb)       | Build the SO100 simulation, run a scripted teacher, and record demonstrations        | `datasets/so100_sim_pickplace`               |
| [`02_smolvla_pick.ipynb`](02_smolvla_pick.ipynb)         | Run the original real-data SmolVLA checkpoint in MuJoCo                              | Zero-shot video and `placed_in_box` baseline |
| [`03_finetune_smolvla.ipynb`](03_finetune_smolvla.ipynb) | Generate `args.yaml`, upload inputs, and launch a SageMaker Training Job             | Fine-tuned SmolVLA checkpoint in S3          |
| [`04_evaluate_smolvla.ipynb`](04_evaluate_smolvla.ipynb) | Find the latest completed job, download its model, and compare before/after rollouts | Comparison table, metrics, and two videos    |
| [`05_agentic_pick.ipynb`](05_agentic_pick.ipynb)         | Run an offline local agent that selects bounded robot skills                         | Tool trace, task verdict, segment videos     |

Run the notebooks in order. Notebook 4 deliberately reruns the baseline instead
of assuming that a successful inference call means task success. Notebook 5
reuses the downloaded fine-tuned checkpoint.

Notebooks 2, 4, and 5 intentionally show their complete runtime logic in the
cells: scene construction, embodiment mapping, policy loading, rollout,
physical metrics, and agent tools. The equivalent modules under `code/` remain
available for automation and command-line reuse, but these notebooks do not
import them.

## Simulation Scenario Architecture

Simulation tasks are explicit modules under `code/scenarios/`; they are not
inferred from filenames:

```text
code/
├── scenarios/
│   ├── base.py          # SimulationScenario contract
│   ├── pick_place.py    # one concrete scenario
│   ├── __init__.py      # explicit registry
│   └── README.md        # template and extension guide
├── so100_teacher.py     # teacher for the pick-place scenario
├── vla_pick.py          # learned-policy inference over registered scenarios
└── agentic_pick.py      # bounded task-level tools for a Strands Agent
```

### The `SimulationScenario` contract

Every scenario declares:

- a unique public `name`;
- robot name and ordered joint keys;
- language instruction;
- camera names and video camera;
- control FPS and default inference length;
- dataset repository ID and local path;
- `build_scene()`;
- `prepare_episode()`, including reset and randomization;
- `make_teacher()`;
- `diagnostics()` and the key that represents task success.

Notebook 1 selects one scenario explicitly:

```python
from scenarios import get_scenario, list_scenarios

print(list_scenarios())
scenario = get_scenario("so100_pick_place")
```

The recorder then reads all scenario-specific values from that object rather
than hardcoding SO100, camera names, FPS, dataset path, teacher, or metric.

### `code/scenarios/pick_place.py`

This is the current concrete scenario. It owns:

- SO100 joint ordering;
- task instruction;
- cube and target-box geometry;
- `top` and `wrist` camera placement;
- the original-checkpoint-aligned robot start pose;
- the eight tested cube start positions;
- episode reset/randomization;
- `placed_in_box` diagnostics;
- the lazy factory for `SO100PickPlaceTeacher`.

Notebook 1 uses this module through the registry, and `vla_pick.py` uses it for
the reusable CLI path. Notebooks 2, 4, and 5 reproduce the same scene and metric
inline so participants can inspect every step.

### `code/so100_teacher.py`

This module contains the privileged teacher for `so100_pick_place`. It reads
the live MuJoCo cube pose, performs numerical IK on the jaw pads, and generates
the smooth grasp/carry/release action trajectory.

The scenario owns reset, randomization, task metadata, and success evaluation;
the teacher owns only action generation. It is used by Notebook 1 and is not
used by the training job or learned-policy inference.

### `code/vla_pick.py`

This module contains the learned-policy side:

- pinned zero-shot SmolVLA checkpoint and original embodiment conversion;
- scenario selection through the registry;
- baseline policy loading;
- fine-tuned policy loading with simulation-native units and camera keys;
- common rollout logic;
- a reusable command-line path equivalent to the steps shown in Notebooks 2
  and 4.

It can run any registered scenario that remains compatible with the current
SO100 checkpoint:

```bash
python code/vla_pick.py --scenario so100_pick_place
```

### `code/agentic_pick.py`

This module packages the same three least-privilege tools demonstrated in
Notebook 5:

- `create_agent_model(provider)` selects Ollama or Amazon Bedrock without
  changing the robot loop;
- `inspect_robot_workspace()` discovers capabilities and deterministic state;
- `execute_robot_skill(...)` validates a skill, target, and destination before
  calling the native `sim.run_policy(...)` loop;
- `request_human_help(...)` disables further motion and records an escalation.

It also owns the shared step budget, attempt history, full-task rollout
horizon, and the agent system prompt. The language model never receives a
joint-action interface. Notebook 5 defines the same pieces inline for teaching;
this module is the reusable packaged form.

### Add another scenario

There is no hidden naming convention. A new scenario is one explicit Python
module, one optional teacher module, and one registry entry.

#### Step 0 — Decide whether this is only a new task

The current training and inference stack can be reused without structural
changes when all of these remain true:

```text
robot:          SO100
state/action:   six values in the existing joint order
cameras:        top + wrist
units:          MuJoCo radians
control rate:   30 FPS
policy family:  SmolVLA
```

Changing object layout, instructions, randomization, teacher motion, and
success metric is a **new scenario**.

Changing robot, number/order of joints, camera contract, units, action
semantics, or policy family is also an **embodiment/trainer change**. Complete
the scenario steps below, then follow Step 10 for the additional files.

#### Step 1 — Create the scenario file

Create:

```text
code/scenarios/my_task.py
```

Import the common contract:

```python
from typing import Any

from .base import SimulationScenario
```

Define task-level constants in this file:

```python
JOINT_KEYS = (
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
)

INSTRUCTION = "Move the red cube to the green target."
CAMERA_NAMES = ("top", "wrist")
```

This file owns the simulation task contract. Do not put scene geometry or task
metrics in the notebook.

#### Step 2 — Implement scene construction

In `code/scenarios/my_task.py`, implement:

```python
def build_scene() -> Any:
    sim = Robot("so100", mesh=False)

    sim.add_object(...)
    sim.add_camera(name="top", ...)
    sim.add_camera(name="wrist", ...)

    # Move the robot and movable objects to the common episode start state.
    ...
    return sim
```

`build_scene()` must return a completely fresh simulation. Include:

- robot;
- task objects and targets;
- static obstacles;
- model-facing cameras;
- deterministic initial robot pose;
- any settling steps required before the first observation.

Notebook 1 calls this through `scenario.build_scene()`. Learned-policy
evaluation can call the same function through the registry.

#### Step 3 — Implement episode reset and randomization

In the same scenario file, implement:

```python
def prepare_episode(sim: Any, episode_index: int) -> dict[str, Any]:
    sim.reset()

    # Select a deterministic or seeded variation.
    object_xy = ...
    sim.move_object("object", position=[*object_xy, object_z])

    # Restore the robot's expected start pose after reset.
    ...

    return {
        "object_xy": object_xy,
        "variation": episode_index,
    }
```

This function owns the distribution from which demonstrations are collected.
Vary only conditions the teacher can complete reliably. Useful variation
includes:

- object and target pose;
- object appearance or shape;
- task instruction;
- obstacles;
- lighting or texture;
- small camera perturbations.

The returned dictionary is metadata printed by Notebook 1; it is useful when a
specific variation fails.

#### Step 4 — Define diagnostics and physical success

In the scenario file, implement:

```python
def diagnostics(sim: Any) -> dict[str, Any]:
    object_position = ...
    task_success = ...

    return {
        "object_position_m": object_position,
        "task_success": bool(task_success),
    }
```

The success value must be physical and independently measurable. Examples:

| Task           | Suggested metric                                           |
| -------------- | ---------------------------------------------------------- |
| Pick and place | Object center inside the target volume                     |
| Pushing        | Final XY distance below a tolerance                        |
| Stacking       | Relative XY alignment and expected object height           |
| Insertion      | Depth, orientation, and contact constraints                |
| Sorting        | Every object in the target associated with its instruction |

Do not use `run_policy status` as task success. That status only reports whether
the software loop completed.

The name of the boolean returned here becomes `success_key` in Step 6.

#### Step 5 — Implement the scripted teacher

Create a dedicated file:

```text
code/my_task_teacher.py
```

The teacher should implement the standard `Policy` interface:

```python
from strands_robots.policies.base import Policy


class MyTaskTeacher(Policy):
    @property
    def provider_name(self) -> str:
        return "scripted-my-task"

    @property
    def requires_images(self) -> bool:
        return False

    @property
    def execution_horizon(self) -> int:
        return 20

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    def set_robot_state_keys(self, keys: list[str]) -> None:
        return None

    async def get_actions(self, observation, instruction, **kwargs):
        ...
        return actions
```

The teacher may use privileged simulator information such as exact object
poses. That is acceptable for demonstration generation: the recorded VLA input
still contains only images, robot state, and instruction.

Before collecting data, test the teacher across every planned variation. Do
not widen randomization until the teacher succeeds consistently.

#### Step 6 — Connect the teacher to the scenario

Back in `code/scenarios/my_task.py`, add a lazy factory:

```python
def make_teacher(sim: Any) -> Any:
    from my_task_teacher import MyTaskTeacher

    return MyTaskTeacher(sim)
```

The import is intentionally lazy. It avoids a circular import while allowing
the teacher to import task constants from `scenarios.my_task`.

#### Step 7 — Export the `SCENARIO` object

At the bottom of `code/scenarios/my_task.py`, construct:

```python
SCENARIO = SimulationScenario(
    name="my_task",
    description="Move a red cube to a green target.",
    robot_name="so100",
    instruction=INSTRUCTION,
    joint_keys=JOINT_KEYS,
    camera_names=CAMERA_NAMES,
    video_camera="top",
    fps=30,
    inference_steps=400,
    dataset_repo_id="local/so100_my_task",
    dataset_relative_path="datasets/so100_my_task",
    success_key="task_success",
    build_scene_fn=build_scene,
    prepare_episode_fn=prepare_episode,
    diagnostics_fn=diagnostics,
    teacher_factory=make_teacher,
)
```

Field ownership:

| Field                                      | Used by                                        |
| ------------------------------------------ | ---------------------------------------------- |
| `robot_name`, `instruction`, `joint_keys`  | Recorder and policy runner                     |
| `camera_names`, `video_camera`, `fps`      | Dataset recording and videos                   |
| `dataset_repo_id`, `dataset_relative_path` | Notebook 1 output                              |
| `inference_steps`                          | Learned-policy evaluation                      |
| `success_key`                              | `scenario.is_success()`                        |
| function fields                            | Scene, randomization, teacher, and diagnostics |

#### Step 8 — Register the scenario explicitly

Edit:

```text
code/scenarios/__init__.py
```

Import and register the new object:

```python
from .my_task import SCENARIO as MY_TASK

register_scenario(MY_TASK)
```

There is no automatic file discovery. A missing registry entry means
`list_scenarios()` will not show the scenario and `get_scenario("my_task")`
will fail with the available names.

#### Step 9 — Select it in Notebook 1

Edit the selection cell in:

```text
01_scripted_pick.ipynb
```

Change only:

```python
SCENARIO_NAME = "my_task"
```

Notebook 1 will then obtain scene, teacher, cameras, FPS, dataset location,
episode metadata, dimensions, diagnostics, and success from the new scenario.

Start with a small smoke dataset. After validating videos, schema, episode
boundaries, and teacher success, collect at least 50 varied successful
demonstrations for a meaningful fine-tuning experiment.

#### Step 10 — Update training only when required

For a new task with the same SO100/top+wrist/6D/radians contract:

1. In `03_finetune_smolvla.ipynb`, update the generated `args.yaml` values:
   - `dataset.repo_id`;
   - job name;
   - dataset input path if it is not obtained from the scenario output.
2. Keep `scripts/train.py` unchanged.
3. Keep the SmolVLA camera mapping unchanged.

If state/action dimensions, names, cameras, FPS, or units change, edit:

```text
scripts/train.py
```

Review:

- `EXPECTED_STATE_NAMES`;
- `EXPECTED_CAMERA_KEYS`;
- `validate_dataset()`;
- `_stage_model_for_dataset()`;
- `state_action_units` written to the training manifest.

Also edit the generated `model.camera_mapping` in
`03_finetune_smolvla.ipynb` so dataset camera roles map to the correct source
checkpoint camera roles.

If the policy family changes from SmolVLA to ACT, Diffusion Policy, π0, GR00T,
or another architecture, create a model-specific training adapter rather than
adding conditionals to the SmolVLA entrypoint.

#### Step 11 — Update learned-policy evaluation

For another compatible scenario, select the registered scenario when calling
the helpers in:

```text
code/vla_pick.py
```

The standalone form is:

```bash
python code/vla_pick.py --scenario my_task
```

For Notebook 4, adapt the inline scene-construction, rollout, and metric cells
so both baseline and fine-tuned policies use the new scenario.

The before/after comparison is valid only when both checkpoints see the same
fresh scene, instruction, cameras, rollout length, and physical success metric.

#### Step 12 — Validate before training

Run these checks in order:

1. `list_scenarios()` includes the new public name.
2. `scenario.build_scene()` returns a fresh valid simulation.
3. Every declared camera produces a useful image.
4. The teacher succeeds across the complete randomization set.
5. Every saved rollout is a separate episode.
6. State/action names and dimensions match the selected robot.
7. Recorded units and action semantics are documented.
8. The zero-shot baseline is measured.
9. The fine-tuned checkpoint is evaluated with the same scenario metric.

See [`code/scenarios/README.md`](code/scenarios/README.md) for a shorter
copyable scenario skeleton.

### Module usage by notebook

| Notebook                          | Scenario registry | `so100_teacher.py` | `vla_pick.py` | `agentic_pick.py` | `scripts/train.py` |
| --------------------------------- | :---------------: | :----------------: | :-----------: | :---------------: | :----------------: |
| 1. Simulation and data collection |        Yes        |    Via scenario    |      No       |        No         |         No         |
| 2. Zero-shot VLA baseline         |   Logic inline    |         No         |  No (inline)  |        No         |         No         |
| 3. SageMaker fine-tuning          |        No         |         No         |      No       |        No         |        Yes         |
| 4. Fine-tuned evaluation          |   Logic inline    |         No         |  No (inline)  |        No         |         No         |
| 5. Agentic supervision            |   Logic inline    |         No         |  No (inline)  |    No (inline)    |         No         |

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
| Agent runtime       | Strands Agents                                       |
| Default provider    | Ollama with `qwen3:4b`                               |
| Optional provider   | Amazon Bedrock                                       |

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
- Ollama plus a tool-capable local model for Notebook 5:

  ```bash
  ollama serve
  ollama pull qwen3:4b
  ```

Notebook 5 defaults to `AGENT_MODEL_PROVIDER=OLLAMA`. To use Bedrock instead:

```bash
export AGENT_MODEL_PROVIDER=BEDROCK
export STRANDS_BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-6
export AWS_REGION=us-east-1
```

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

## Agentic Supervision

Notebook 5 uses the official `strands-robots` control boundary:

```text
free-form user goal
  ↓
Strands Agent
  ├─ model: Ollama (on-device) or Bedrock (connected)
  ├─ inspect_robot_workspace()
  ├─ execute_robot_skill(skill, target, destination)
  │    └─ canonical instruction → sim.run_policy()
  │         └─ SmolVLA observation → action loop at 30 Hz
  └─ request_human_help(reason)
```

`Robot("so100")` is itself a Strands `AgentTool` whose JSON schema publishes 77
simulation actions. A general operator can therefore use
`Agent(tools=[sim])`. This workshop deliberately wraps that broad interface in
three narrower tools so the checkpoint is loaded once, the local LLM cannot
mutate the scene arbitrarily, unsupported capabilities are refused, and the
total control-step budget is enforced in code. Notebook 5 shows these tools,
the capability catalog, and the budget state directly in its cells;
`code/agentic_pick.py` is the reusable equivalent.

The selected agent model interprets a free-form goal, discovers available
entities and skills, chooses structured tool arguments, reads `placed_in_box`,
retries once, and escalates when recovery is exhausted. It does not replace the
VLA, emit joint commands, or repair weak model weights.

## Repository Layout

```text
.
├── 01_scripted_pick.ipynb
├── 02_smolvla_pick.ipynb
├── 03_finetune_smolvla.ipynb
├── 04_evaluate_smolvla.ipynb
├── 05_agentic_pick.ipynb
├── requirements.txt
├── code/
│   ├── scenarios/
│   │   ├── base.py
│   │   ├── pick_place.py
│   │   ├── __init__.py
│   │   └── README.md
│   ├── agentic_pick.py
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
- The agent can retry or stop a policy; it cannot create a grasp skill absent
  from the checkpoint and demonstrations.
- Nothing in this repository is a real-robot safety controller.
