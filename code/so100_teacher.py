"""Scripted SO100 teacher for the pick-and-place workshop dataset.

The learned policy in ``02_zero_shot_vla_emulation.ipynb`` and this teacher use the same
robot, scene, cameras, language instruction, state keys, and action keys.  The
teacher is deliberately privileged: it reads the cube pose from MuJoCo and
builds a smooth joint-space trajectory around it.  Its purpose is to generate
successful demonstrations, not to model what a real robot can observe.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from strands_robots.policies.base import Policy

from scenarios.pick_place import JOINT_KEYS

# Joint-space templates in degrees.  Position IK only makes the small
# per-episode correction needed for the randomized cube position.
_PREGRASP_DEG = np.array([0.0, -55.0, 40.0, 90.0, 0.0])
_GRASP_DEG = np.array([0.0, -30.0, 20.0, 95.0, 0.0])
_LIFT_DEG = np.array([0.0, -60.0, 45.0, 90.0, 0.0])
_CARRY_DEG = np.array([32.0, -55.0, 40.0, 90.0, 0.0])
_LOWER_DEG = np.array([32.0, -39.0, 18.0, 95.0, 0.0])
_RETREAT_DEG = np.array([32.0, -60.0, 45.0, 90.0, 0.0])

JAW_OPEN = 0.60
JAW_CLOSE = -0.15


class SO100PickPlaceTeacher(Policy):
    """Ground-truth teacher: open → grasp → lift → carry → release.

    The grasp waypoints are corrected with position-only numerical IK around
    stable, hand-verified templates.  Actions between waypoints use a cosine
    interpolation, which avoids discontinuities in both the dataset and the
    position servos.
    """

    def __init__(self, sim: Any, horizon: int = 20):
        """Bind scene geometry and precompute a complete demonstration trajectory."""

        super().__init__()
        self.sim = sim
        self.horizon = horizon
        self._cursor = 0
        self.phase_boundaries: list[tuple[str, int]] = []

        model = sim.mj_model
        self._arm_lo = np.asarray(model.jnt_range[:5, 0], dtype=float)
        self._arm_hi = np.asarray(model.jnt_range[:5, 1], dtype=float)
        self._scratch = mujoco.MjData(model)
        self._fixed_pad = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "so100/fixed_jaw_pad_1"
        )
        self._moving_pad = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "so100/moving_jaw_pad_1"
        )
        self._cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        missing = [
            name
            for name, index in (
                ("so100/fixed_jaw_pad_1", self._fixed_pad),
                ("so100/moving_jaw_pad_1", self._moving_pad),
                ("cube", self._cube),
            )
            if index < 0
        ]
        if missing:
            raise ValueError(f"Teacher cannot find required MuJoCo objects: {missing}")

        self.actions = self._build_trajectory()

    @property
    def provider_name(self) -> str:
        """Return the provider label exposed through the Policy interface."""

        return "scripted-so100-teacher"

    @property
    def requires_images(self) -> bool:
        """Declare that the privileged teacher does not consume camera images."""

        return False

    @property
    def execution_horizon(self) -> int:
        """Return the maximum number of actions emitted per policy request."""

        return self.horizon

    @property
    def n_steps(self) -> int:
        """Return the number of control steps in the generated demonstration."""

        return len(self.actions)

    def set_robot_state_keys(self, keys: list[str]) -> None:
        """Accept the Policy hook; this teacher uses fixed SO100 joint keys."""

        # The teacher ignores observations; keys are fixed by the SO100.
        return None

    def _pinch_center(self, arm_q: np.ndarray, jaw_q: float) -> np.ndarray:
        """Compute the world-space midpoint between the two jaw-pad geoms."""

        data = self._scratch
        data.qpos[:] = self.sim.mj_data.qpos
        data.qpos[:5] = arm_q
        data.qpos[5] = jaw_q
        mujoco.mj_forward(self.sim.mj_model, data)
        return 0.5 * (
            data.geom_xpos[self._fixed_pad] + data.geom_xpos[self._moving_pad]
        )

    def _solve_position(
        self,
        template_deg: np.ndarray,
        jaw_q: float,
        target: np.ndarray,
    ) -> np.ndarray:
        """Make a small DLS correction while staying near a stable grasp pose."""
        q = np.deg2rad(template_deg).astype(float)
        eps = 2e-4
        for _ in range(100):
            point = self._pinch_center(q, jaw_q)
            error = target - point
            if np.linalg.norm(error) < 5e-4:
                break
            jacobian = np.empty((3, 4), dtype=float)
            for joint in range(4):
                shifted = q.copy()
                shifted[joint] += eps
                jacobian[:, joint] = (
                    self._pinch_center(shifted, jaw_q) - point
                ) / eps
            damping = 2e-5
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(3),
                error,
            )
            q[:4] = np.clip(
                q[:4] + 0.7 * delta,
                self._arm_lo[:4],
                self._arm_hi[:4],
            )

        residual = float(np.linalg.norm(target - self._pinch_center(q, jaw_q)))
        if residual > 0.004:
            raise RuntimeError(
                f"Teacher IK residual {residual:.4f} m is unsafe for grasp target "
                f"{np.round(target, 4).tolist()}"
            )
        return q

    @staticmethod
    def _with_jaw(arm_q: np.ndarray, jaw_q: float) -> np.ndarray:
        """Append one jaw command to a five-joint arm configuration."""

        return np.r_[np.asarray(arm_q, dtype=float), float(jaw_q)]

    def _build_trajectory(self) -> list[dict[str, float]]:
        """Plan all smooth joint-space phases for one pick-and-place episode."""

        cube = np.asarray(self.sim.mj_data.xpos[self._cube], dtype=float).copy()

        # These offsets reproduce the verified physical grasp: the open pads
        # approach slightly behind and above the cube centre, then closing the
        # jaw centres the cube before the lift.
        pregrasp = self._solve_position(
            _PREGRASP_DEG,
            JAW_OPEN,
            cube + np.array([0.0, -0.005, 0.067]),
        )
        grasp = self._solve_position(
            _GRASP_DEG,
            JAW_OPEN,
            cube + np.array([0.0, -0.005, 0.009]),
        )
        lift = self._solve_position(
            _LIFT_DEG,
            0.20,
            cube + np.array([0.0, 0.017, 0.065]),
        )

        carry = np.deg2rad(_CARRY_DEG)
        lower = np.deg2rad(_LOWER_DEG)
        retreat = np.deg2rad(_RETREAT_DEG)
        current = np.asarray(self.sim.mj_data.ctrl[:6], dtype=float).copy()
        plan: list[dict[str, float]] = []

        def segment(name: str, target: np.ndarray, steps: int) -> None:
            """Append one cosine-interpolated phase to the action plan."""

            nonlocal current
            start = current.copy()
            for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
                smooth = 0.5 - 0.5 * np.cos(np.pi * fraction)
                command = start + smooth * (target - start)
                plan.append(
                    dict(zip(JOINT_KEYS, command.tolist(), strict=True))
                )
            current = target.copy()
            self.phase_boundaries.append((name, len(plan)))

        segment("open", self._with_jaw(current[:5], JAW_OPEN), 15)
        segment("pregrasp", self._with_jaw(pregrasp, JAW_OPEN), 35)
        segment("descend", self._with_jaw(grasp, JAW_OPEN), 30)
        segment("close", self._with_jaw(grasp, JAW_CLOSE), 25)
        segment("lift", self._with_jaw(lift, JAW_CLOSE), 45)
        segment("carry", self._with_jaw(carry, JAW_CLOSE), 55)
        segment("lower", self._with_jaw(lower, JAW_CLOSE), 35)
        segment("release", self._with_jaw(lower, JAW_OPEN), 25)
        segment("retreat", self._with_jaw(retreat, JAW_OPEN), 35)
        segment("settle", self._with_jaw(retreat, JAW_OPEN), 20)
        return plan

    async def get_actions(
        self,
        observation: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """Return the next precomputed action chunk and advance the trajectory cursor."""

        del observation, instruction, kwargs
        start = self._cursor
        stop = min(start + self.horizon, len(self.actions))
        chunk = self.actions[start:stop]
        self._cursor = stop
        if chunk:
            return chunk
        return [self.actions[-1].copy()]


__all__ = ["SO100PickPlaceTeacher"]
