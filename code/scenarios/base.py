"""Shared contract implemented by every workshop simulation scenario."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

BuildScene = Callable[[], Any]
PrepareEpisode = Callable[[Any, int], dict[str, Any]]
Diagnostics = Callable[[Any], dict[str, Any]]
TeacherFactory = Callable[[Any], Any]


@dataclass(frozen=True)
class SimulationScenario:
    """Describe one complete data-collection and evaluation environment."""

    name: str
    description: str
    robot_name: str
    instruction: str
    joint_keys: tuple[str, ...]
    camera_names: tuple[str, ...]
    video_camera: str
    fps: int
    inference_steps: int
    dataset_repo_id: str
    dataset_relative_path: str
    success_key: str
    build_scene_fn: BuildScene
    prepare_episode_fn: PrepareEpisode
    diagnostics_fn: Diagnostics
    teacher_factory: TeacherFactory | None = None

    def __post_init__(self) -> None:
        """Validate static fields when a scenario module is imported."""

        if not self.name.isidentifier():
            raise ValueError(f"Scenario name must be a non-empty identifier, got {self.name!r}")
        if self.fps <= 0 or self.inference_steps <= 0:
            raise ValueError("Scenario fps and inference_steps must be positive.")
        if not self.joint_keys:
            raise ValueError("Scenario must declare at least one joint key.")
        if not self.camera_names:
            raise ValueError("Scenario must declare at least one camera.")
        if self.video_camera not in self.camera_names:
            raise ValueError(
                f"video_camera {self.video_camera!r} is not in {self.camera_names}"
            )

    @property
    def image_feature_keys(self) -> set[str]:
        """Return the LeRobot feature names produced by the declared cameras."""

        return {f"observation.images.{name}" for name in self.camera_names}

    def dataset_path(self, project_root: Path) -> Path:
        """Resolve this scenario's default local dataset directory."""

        return (project_root / self.dataset_relative_path).resolve()

    def build_scene(self) -> Any:
        """Construct a fresh simulation for this scenario."""

        return self.build_scene_fn()

    def prepare_episode(self, sim: Any, episode_index: int) -> dict[str, Any]:
        """Reset and randomize one episode, returning printable metadata."""

        return self.prepare_episode_fn(sim, episode_index)

    def make_teacher(self, sim: Any) -> Any:
        """Construct the scripted teacher configured for the live scene."""

        if self.teacher_factory is None:
            raise RuntimeError(
                f"Scenario {self.name!r} has no teacher_factory and cannot collect demonstrations."
            )
        return self.teacher_factory(sim)

    def diagnostics(self, sim: Any) -> dict[str, Any]:
        """Compute task-specific diagnostics for the live scene."""

        return self.diagnostics_fn(sim)

    def is_success(self, sim: Any) -> bool:
        """Return the task-success boolean from the scenario diagnostics."""

        diagnostics = self.diagnostics(sim)
        if self.success_key not in diagnostics:
            raise KeyError(
                f"Scenario {self.name!r} diagnostics do not contain "
                f"success_key={self.success_key!r}: {diagnostics}"
            )
        return bool(diagnostics[self.success_key])
