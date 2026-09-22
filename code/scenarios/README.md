# Simulation scenarios

Each concrete scenario lives in its own Python file and exposes exactly one
`SCENARIO` object implementing the `SimulationScenario` contract from
`base.py`. Registration is explicit; filenames are not discovered by magic.

The current scenario is:

```text
pick_place.py → SCENARIO.name == "so100_pick_place"
```

Select it with:

```python
from scenarios import get_scenario, list_scenarios

print(list_scenarios())
scenario = get_scenario("so100_pick_place")
sim = scenario.build_scene()
teacher = scenario.make_teacher(sim)
```

## Add a new scenario

1. Create `code/scenarios/my_task.py`.
2. Implement scene construction, episode preparation/randomization, diagnostics,
   and an optional teacher factory.
3. Construct one `SimulationScenario` named `SCENARIO`.
4. Import and register it explicitly in `code/scenarios/__init__.py`.
5. Select its public name in the data-collection notebook.

Minimal structure:

```python
from typing import Any

from .base import SimulationScenario


def build_scene() -> Any:
    ...


def prepare_episode(sim: Any, episode_index: int) -> dict[str, Any]:
    ...
    return {"variation": episode_index}


def diagnostics(sim: Any) -> dict[str, Any]:
    return {"task_success": ...}


def make_teacher(sim: Any) -> Any:
    from my_task_teacher import MyTaskTeacher
    return MyTaskTeacher(sim)


SCENARIO = SimulationScenario(
    name="my_task",
    description="One sentence describing the task.",
    robot_name="so100",
    instruction="The instruction received by the VLA.",
    joint_keys=("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"),
    camera_names=("top", "wrist"),
    video_camera="top",
    fps=30,
    inference_steps=400,
    dataset_repo_id="local/my_task",
    dataset_relative_path="datasets/my_task",
    success_key="task_success",
    build_scene_fn=build_scene,
    prepare_episode_fn=prepare_episode,
    diagnostics_fn=diagnostics,
    teacher_factory=make_teacher,
)
```

Then register it:

```python
from .my_task import SCENARIO as MY_TASK

register_scenario(MY_TASK)
```

If the new scenario changes robot, joint dimensions, camera contract, units, or
action semantics, the training validator and inference embodiment must change
as well. A new scene alone is reusable only while those contracts stay
compatible.
