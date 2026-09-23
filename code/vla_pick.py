#!/usr/bin/env python3
"""Run the zero-shot SmolVLA baseline in a registered simulation scenario.

SmolVLA is the learned neural model. ``LerobotLocalPolicy`` adapts that model
to the common ``Policy`` interface expected by ``run_policy()``. Scenario
construction and task metrics live under ``code/scenarios`` so additional
simulation tasks can reuse this inference path without copying the rollout
logic.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

# These must be set before importing torch/MuJoCo/strands_robots.
os.environ.setdefault("MUJOCO_GL", "cgl")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("STRANDS_TRUST_REMOTE_CODE", "1")

import torch

from strands_robots.policies import create_policy

from scenarios import get_scenario, list_scenarios
from scenarios.pick_place import BOX_CENTER, GRIPPER_JOINT_RANGE

DEFAULT_SCENARIO_NAME = "so100_pick_place"
DEFAULT_SCENARIO = get_scenario(DEFAULT_SCENARIO_NAME)

MODEL_ID = "lucarrr/smolvla_so100_pickplace_finetuned_v2"
MODEL_REVISION = "4fde42badae91b5c88bfe1a399a3f4b994fc0698"
TRAINING_DATASET = "lerobot/svla_so100_pickplace"

# Backward-compatible exports used by the existing notebooks.
INSTRUCTION = DEFAULT_SCENARIO.instruction
JOINT_KEYS = list(DEFAULT_SCENARIO.joint_keys)

SO100_SMOLVLA_EMBODIMENT = {
    "name": "so100_smolvla_pickplace_sim",
    "obs_rename": {
        "wrist": "observation.images.camera1",
        "top": "observation.images.camera2",
    },
    "state_keys": JOINT_KEYS,
    "action_keys": JOINT_KEYS,
    "dim_policy": "strict",
    # Real SO-arm data uses degrees and a 0-100 gripper convention. MuJoCo
    # supplies radians, so the original checkpoint needs this conversion.
    "state_units": "degrees",
    "action_units": "degrees",
    "gripper_index": 5,
    "gripper_joint_range": list(GRIPPER_JOINT_RANGE),
}

FINETUNED_SMOLVLA_EMBODIMENT = {
    "name": "so100_smolvla_sim_finetuned",
    "obs_rename": {
        "wrist": "observation.images.wrist",
        "top": "observation.images.top",
    },
    "state_keys": JOINT_KEYS,
    "action_keys": JOINT_KEYS,
    "dim_policy": "strict",
    "state_units": "radians",
    "action_units": "radians",
}

VIDEO_PATH = Path(__file__).resolve().parents[1] / "smolvla_pick.mp4"


def _json_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the first structured JSON block from a tool-style result envelope."""

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


def build_scene(scenario_name: str = DEFAULT_SCENARIO_NAME) -> Any:
    """Build a fresh scene from the explicit scenario registry."""

    return get_scenario(scenario_name).build_scene()


def load_policy(device: str = "auto") -> tuple[Any, str]:
    """Download once, then load the pinned zero-shot checkpoint."""

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


def load_finetuned_policy(
    model_dir: str | Path,
    device: str = "auto",
) -> tuple[Any, str]:
    """Load a simulation-fine-tuned SmolVLA checkpoint from a local directory."""

    checkpoint = Path(model_dir).expanduser().resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"No SmolVLA config.json found under {checkpoint}")

    resolved_device = choose_device(device)
    policy = create_policy(
        "lerobot_local",
        pretrained_name_or_path=str(checkpoint),
        policy_type="smolvla",
        device=resolved_device,
        embodiment=FINETUNED_SMOLVLA_EMBODIMENT,
        strict_keys=True,
    )
    return policy, resolved_device


def run_rollout(
    sim: Any,
    policy: Any,
    *,
    steps: int | None = None,
    video_path: Path | None = VIDEO_PATH,
    scenario_name: str = DEFAULT_SCENARIO_NAME,
) -> dict[str, Any]:
    """Run one learned-policy rollout using the selected scenario contract."""

    scenario = get_scenario(scenario_name)
    rollout_steps = scenario.inference_steps if steps is None else steps
    video = None
    if video_path is not None:
        video = {
            "path": str(video_path),
            "camera": scenario.video_camera,
            "fps": scenario.fps,
        }
    return sim.run_policy(
        robot_name=scenario.robot_name,
        policy_object=policy,
        instruction=scenario.instruction,
        n_steps=rollout_steps,
        control_frequency=scenario.fps,
        fast_mode=True,
        video=video,
    )


def cube_diagnostics(
    sim: Any,
    scenario_name: str = DEFAULT_SCENARIO_NAME,
) -> dict[str, Any]:
    """Return diagnostics from the selected scenario."""

    return get_scenario(scenario_name).diagnostics(sim)


def main() -> None:
    """Run the standalone zero-shot policy in one registered scenario."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=list_scenarios(),
        default=DEFAULT_SCENARIO_NAME,
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")

    scenario = get_scenario(args.scenario)
    print(f"Building scenario {scenario.name!r}: {scenario.instruction}")
    sim = scenario.build_scene()

    print(f"Loading {MODEL_ID}@{MODEL_REVISION[:8]}…")
    policy, device = load_policy(args.device)
    print(
        f"Policy wrapper: {type(policy).__name__}; "
        f"learned model: SmolVLA; device: {device}"
    )

    result = run_rollout(
        sim,
        policy,
        steps=args.steps,
        video_path=None if args.no_video else VIDEO_PATH,
        scenario_name=scenario.name,
    )
    telemetry = _json_payload(result)
    print(f"run_policy status: {result.get('status')}")
    print(f"diagnostics: {scenario.diagnostics(sim)}")
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
        "Note: inference success is not task success. Inspect the scenario's "
        "physical success metric and rollout video."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BOX_CENTER",
    "DEFAULT_SCENARIO_NAME",
    "FINETUNED_SMOLVLA_EMBODIMENT",
    "INSTRUCTION",
    "JOINT_KEYS",
    "MODEL_ID",
    "MODEL_REVISION",
    "SO100_SMOLVLA_EMBODIMENT",
    "TRAINING_DATASET",
    "VIDEO_PATH",
    "build_scene",
    "choose_device",
    "cube_diagnostics",
    "load_finetuned_policy",
    "load_policy",
    "run_rollout",
]
