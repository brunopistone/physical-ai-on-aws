#!/usr/bin/env python3
"""Version 2: run a small Vision-Language-Action model on a local SO100 sim.

The important distinction is:

    SmolVLA                         = the learned neural model
    LerobotLocalPolicy(SmolVLA)     = the Policy implementation used by run_policy()

This example is deliberately simulation-only.  The checkpoint was fine-tuned on
real SO100 footage, so local inference works but task success in this visually
different MuJoCo scene is not guaranteed.  Fine-tuning on matching simulation
demonstrations is the next step if reliable success is the goal.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

# These must be set before importing torch/MuJoCo/strands_robots.
os.environ.setdefault("MUJOCO_GL", "cgl")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# strands-robots deliberately gates its local Hugging Face provider.  This demo
# opts in only while using the pinned package and checkpoint revisions below.
os.environ.setdefault("STRANDS_TRUST_REMOTE_CODE", "1")

import mujoco
import numpy as np
import torch

from strands_robots import Robot
from strands_robots.policies import create_policy

MODEL_ID = "lucarrr/smolvla_so100_pickplace_finetuned_v2"
MODEL_REVISION = "4fde42badae91b5c88bfe1a399a3f4b994fc0698"
TRAINING_DATASET = "lerobot/svla_so100_pickplace"
INSTRUCTION = "Pick up the cube and place it in the box."

JOINT_KEYS = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]

# The checkpoint consumes camera1/camera2 internally.  Its saved preprocessor
# says that those correspond to the real dataset's wrist/top streams.
SO100_SMOLVLA_EMBODIMENT = {
    "name": "so100_smolvla_pickplace_sim",
    "obs_rename": {
        "wrist": "observation.images.camera1",
        "top": "observation.images.camera2",
    },
    "state_keys": JOINT_KEYS,
    "action_keys": JOINT_KEYS,
    "dim_policy": "strict",
    # Real LeRobot SO-arm data uses mid-centred degrees for arm motors and
    # RANGE_0_100 for the gripper.  MuJoCo uses radians for all six joints.
    "state_units": "degrees",
    "action_units": "degrees",
    "gripper_index": 5,
    "gripper_joint_range": [-0.175, 1.745],
}

# Mean observation.state from the checkpoint's training dataset.  Starting here
# avoids asking the model to interpret SO100's all-zero MuJoCo pose, which is far
# outside the real demonstrations for wrist pitch and wrist roll.
DATASET_MEAN_STATE = np.array(
    [14.4717, -55.7695, 54.3857, 63.2262, 85.8417, 9.3545],
    dtype=np.float64,
)

BOX_CENTER = np.array([0.16, -0.30], dtype=np.float64)
VIDEO_PATH = Path(__file__).resolve().parents[1] / "smolvla_pick.mp4"


def _require_success(result: dict[str, Any], operation: str) -> dict[str, Any]:
    """Turn an agent-tool error envelope into a normal Python exception."""
    if result.get("status") == "success":
        return result
    text = " | ".join(
        str(item.get("text"))
        for item in result.get("content", [])
        if isinstance(item, dict) and item.get("text")
    )
    raise RuntimeError(f"{operation} failed: {text or result}")


def _json_payload(result: dict[str, Any]) -> dict[str, Any]:
    return next(
        (
            item["json"]
            for item in result.get("content", [])
            if isinstance(item, dict) and isinstance(item.get("json"), dict)
        ),
        {},
    )


def choose_device(requested: str = "auto") -> str:
    """Prefer Apple Metal; retain CPU as a slower, portable fallback."""
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but torch.backends.mps.is_available() is False. "
            "Run with --device cpu or install an Apple-Silicon PyTorch build."
        )
    return requested


def _dataset_mean_action() -> dict[str, float]:
    """Convert the real-dataset mean state into MuJoCo joint units."""
    values = np.empty(6, dtype=np.float64)
    values[:5] = np.deg2rad(DATASET_MEAN_STATE[:5])
    lo, hi = SO100_SMOLVLA_EMBODIMENT["gripper_joint_range"]
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
    """Create the safe MuJoCo-only scene and its two model-facing cameras."""
    sim = Robot("so100", mesh=False)  # mode="sim" is the safe default

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

    # A literal camera on so100/Fixed_Jaw is almost completely occluded by the
    # large MuJoCo jaw mesh.  Keep the checkpoint's expected "wrist" source key,
    # but use a close fixed view that actually exposes hand, cube and target.
    # This is another explicit real->sim approximation, not a claim that the
    # camera geometry matches the real training rig.
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

    # Let the position servos reach an in-distribution starting pose and let the
    # cube settle before the first image is sent to the VLA.
    _require_success(
        sim.send_action(_dataset_mean_action(), robot_name="so100", n_substeps=600),
        "move SO100 to dataset-mean pose",
    )
    return sim


def load_policy(device: str = "auto") -> tuple[Any, str]:
    """Download once, then load the pinned SmolVLA checkpoint on MPS or CPU."""
    resolved_device = choose_device(device)
    policy = create_policy(
        "lerobot_local",
        pretrained_name_or_path=MODEL_ID,
        revision=MODEL_REVISION,
        policy_type="smolvla",
        device=resolved_device,
        embodiment=SO100_SMOLVLA_EMBODIMENT,
        strict_keys=True,
    )
    return policy, resolved_device


def run_rollout(
    sim: Any,
    policy: Any,
    *,
    steps: int = 400,
    video_path: Path | None = VIDEO_PATH,
) -> dict[str, Any]:
    """Run one 30 Hz rollout; SmolVLA supplies 50-action chunks."""
    video = None
    if video_path is not None:
        video = {"path": str(video_path), "camera": "top", "fps": 30}
    return sim.run_policy(
        robot_name="so100",
        policy_object=policy,
        instruction=INSTRUCTION,
        n_steps=steps,
        control_frequency=30,
        fast_mode=True,
        video=video,
    )


def cube_diagnostics(sim: Any) -> dict[str, Any]:
    cube_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    if cube_id < 0:
        raise RuntimeError("MuJoCo body 'cube' disappeared from the scene.")
    position = np.asarray(sim.mj_data.xpos[cube_id], dtype=float).copy()
    in_box_xy = bool(np.all(np.abs(position[:2] - BOX_CENTER) < np.array([0.045, 0.045])))
    return {
        "cube_position_m": np.round(position, 4).tolist(),
        "placed_in_box": in_box_xy and position[2] < 0.10,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    print(f"Building SO100 simulation for task: {INSTRUCTION}")
    sim = build_scene()

    print(f"Loading {MODEL_ID}@{MODEL_REVISION[:8]}…")
    policy, device = load_policy(args.device)
    print(f"Policy wrapper: {type(policy).__name__}; learned model: SmolVLA; device: {device}")

    result = run_rollout(
        sim,
        policy,
        steps=args.steps,
        video_path=None if args.no_video else VIDEO_PATH,
    )
    telemetry = _json_payload(result)
    print(f"run_policy status: {result.get('status')}")
    print(f"diagnostics: {cube_diagnostics(sim)}")
    if telemetry:
        interesting = {
            key: telemetry[key]
            for key in (
                "policy_load_time_s",
                "policy_load_cache_hit",
                "positional_fallback_used",
                "generic_state_keys_used",
            )
            if key in telemetry
        }
        print(f"policy telemetry: {interesting}")
    if not args.no_video:
        print(f"video: {VIDEO_PATH}")
    print(
        "Note: inference success is not task success. This checkpoint learned from "
        "real-camera SO100 demonstrations; MuJoCo is a domain shift."
    )


if __name__ == "__main__":
    main()
