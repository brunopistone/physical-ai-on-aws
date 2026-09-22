"""SO100 cube pick-and-place scenario used by the four workshop notebooks."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from strands_robots import Robot

from .base import SimulationScenario

JOINT_KEYS = (
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
)
INSTRUCTION = "Pick up the cube and place it in the box."
BOX_CENTER = np.array([0.16, -0.30], dtype=np.float64)

# Small variations that remain inside the tested top-down grasp workspace.
DEMO_CUBE_POSITIONS = (
    (0.000, -0.350),
    (-0.020, -0.350),
    (0.020, -0.350),
    (0.000, -0.340),
    (0.000, -0.360),
    (-0.015, -0.340),
    (0.015, -0.360),
    (0.025, -0.345),
)

# Mean observation.state from the original real SO100 checkpoint's dataset.
DATASET_MEAN_STATE = np.array(
    [14.4717, -55.7695, 54.3857, 63.2262, 85.8417, 9.3545],
    dtype=np.float64,
)
GRIPPER_JOINT_RANGE = (-0.175, 1.745)


def _require_success(result: dict[str, Any], operation: str) -> dict[str, Any]:
    """Raise a conventional exception for a failed strands-robots result envelope."""

    if result.get("status") == "success":
        return result
    text = " | ".join(
        str(item.get("text"))
        for item in result.get("content", [])
        if isinstance(item, dict) and item.get("text")
    )
    raise RuntimeError(f"{operation} failed: {text or result}")


def _dataset_mean_action() -> dict[str, float]:
    """Convert the real-dataset mean state into MuJoCo joint units."""

    values = np.empty(6, dtype=np.float64)
    values[:5] = np.deg2rad(DATASET_MEAN_STATE[:5])
    lo, hi = GRIPPER_JOINT_RANGE
    values[5] = lo + (DATASET_MEAN_STATE[5] / 100.0) * (hi - lo)
    return dict(zip(JOINT_KEYS, values.tolist(), strict=True))


def _add_target_box(sim: Any) -> None:
    """Build a collision-correct open tray from five static convex boxes."""

    blue = [0.12, 0.28, 0.75, 1.0]
    x, y = BOX_CENTER
    pieces = {
        "target_base": ([x, y, 0.005], [0.12, 0.12, 0.01]),
        "target_left": ([x - 0.055, y, 0.025], [0.01, 0.12, 0.05]),
        "target_right": ([x + 0.055, y, 0.025], [0.01, 0.12, 0.05]),
        "target_back": ([x, y + 0.055, 0.025], [0.10, 0.01, 0.05]),
        "target_front": ([x, y - 0.055, 0.025], [0.10, 0.01, 0.05]),
    }
    for name, (position, size) in pieces.items():
        _require_success(
            sim.add_object(
                name=name,
                shape="box",
                position=list(position),
                size=list(size),
                color=blue,
                is_static=True,
            ),
            f"add {name}",
        )


def build_scene() -> Any:
    """Create the SO100, task objects, cameras, and checkpoint-aligned start pose."""

    sim = Robot("so100", mesh=False)
    _require_success(
        sim.add_object(
            name="cube",
            shape="box",
            position=[0.0, -0.35, 0.015],
            size=[0.03, 0.03, 0.03],
            color=[0.9, 0.12, 0.08, 1.0],
            mass=0.03,
        ),
        "add cube",
    )
    _add_target_box(sim)
    _require_success(
        sim.add_camera(
            name="top",
            position=[0.45, -0.60, 0.42],
            target=[0.06, -0.31, 0.06],
            fov=55,
            width=640,
            height=480,
        ),
        "add top camera",
    )
    _require_success(
        sim.add_camera(
            name="wrist",
            position=[0.02, -0.56, 0.18],
            target=[0.04, -0.31, 0.04],
            fov=70,
            width=640,
            height=480,
        ),
        "add wrist camera",
    )
    _require_success(
        sim.send_action(_dataset_mean_action(), robot_name="so100", n_substeps=600),
        "move SO100 to dataset-mean pose",
    )
    return sim


def prepare_episode(sim: Any, episode_index: int) -> dict[str, Any]:
    """Reset physics, vary the cube pose, and restore the common start state."""

    cube_xy = DEMO_CUBE_POSITIONS[episode_index % len(DEMO_CUBE_POSITIONS)]
    _require_success(sim.reset(), "reset scene")
    _require_success(
        sim.move_object("cube", position=[float(cube_xy[0]), float(cube_xy[1]), 0.015]),
        "move cube",
    )
    _require_success(
        sim.send_action(_dataset_mean_action(), robot_name="so100", n_substeps=600),
        "restore SO100 start pose",
    )
    return {"cube_xy": cube_xy}


def diagnostics(sim: Any) -> dict[str, Any]:
    """Return the cube pose and the geometric ``placed_in_box`` metric."""

    cube_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    if cube_id < 0:
        raise RuntimeError("MuJoCo body 'cube' disappeared from the scene.")
    position = np.asarray(sim.mj_data.xpos[cube_id], dtype=float).copy()
    in_box_xy = bool(np.all(np.abs(position[:2] - BOX_CENTER) < np.array([0.045, 0.045])))
    return {
        "cube_position_m": np.round(position, 4).tolist(),
        "placed_in_box": in_box_xy and position[2] < 0.10,
    }


def make_teacher(sim: Any) -> Any:
    """Import lazily and construct the pick-and-place teacher."""

    from so100_teacher import SO100PickPlaceTeacher

    return SO100PickPlaceTeacher(sim)


SCENARIO = SimulationScenario(
    name="so100_pick_place",
    description="Pick a red cube and place it in a blue target box.",
    robot_name="so100",
    instruction=INSTRUCTION,
    joint_keys=JOINT_KEYS,
    camera_names=("top", "wrist"),
    video_camera="top",
    fps=30,
    inference_steps=400,
    dataset_repo_id="local/so100_sim_pickplace",
    dataset_relative_path="datasets/so100_sim_pickplace",
    success_key="placed_in_box",
    build_scene_fn=build_scene,
    prepare_episode_fn=prepare_episode,
    diagnostics_fn=diagnostics,
    teacher_factory=make_teacher,
)

__all__ = [
    "BOX_CENTER",
    "DATASET_MEAN_STATE",
    "DEMO_CUBE_POSITIONS",
    "GRIPPER_JOINT_RANGE",
    "INSTRUCTION",
    "JOINT_KEYS",
    "SCENARIO",
]
