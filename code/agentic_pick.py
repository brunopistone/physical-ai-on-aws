"""Bounded Strands Agent tools for supervising the SO100 pick-place policy.

The language model operates at task level: it may inspect deterministic task
state and request another policy segment. The SmolVLA policy remains the only
component that emits joint actions, and ``strands-robots`` remains the owner of
the high-frequency observation-inference-action loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strands import tool

from scenarios import get_scenario
from scenarios.pick_place import BOX_CENTER

SUCCESS_STOP_WHEN = {
    "predicate": "inside_region",
    "body": "cube",
    "min": [float(BOX_CENTER[0] - 0.045), float(BOX_CENTER[1] - 0.045), 0.0],
    "max": [float(BOX_CENTER[0] + 0.045), float(BOX_CENTER[1] + 0.045), 0.10],
}


@dataclass
class AgenticLoopState:
    """Track the bounded policy budget and machine-readable attempt history."""

    segment_steps: int
    max_total_steps: int
    total_steps_used: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    terminal_error: str | None = None

    @property
    def remaining_steps(self) -> int:
        """Return the number of control steps still available to the agent."""

        return max(0, self.max_total_steps - self.total_steps_used)


def _json_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the first JSON block from a Strands tool-result envelope."""

    return next(
        (
            item["json"]
            for item in result.get("content", [])
            if isinstance(item, dict) and isinstance(item.get("json"), dict)
        ),
        {},
    )


def _text_payload(result: dict[str, Any]) -> str:
    """Join human-readable text blocks from a Strands tool-result envelope."""

    return " | ".join(
        str(item["text"])
        for item in result.get("content", [])
        if isinstance(item, dict) and item.get("text")
    )


def _tool_result(payload: dict[str, Any], *, status: str = "success") -> dict[str, Any]:
    """Wrap a JSON-serializable payload in the standard Strands tool envelope."""

    return {
        "status": status,
        "content": [
            {"text": json.dumps(payload, sort_keys=True)},
            {"json": payload},
        ],
    }


def create_agentic_pick_tools(
    sim: Any,
    policy: Any,
    *,
    scenario_name: str = "so100_pick_place",
    output_dir: str | Path = "outputs/agentic-pick",
    segment_steps: int = 200,
    max_total_steps: int = 400,
    record_video: bool = True,
) -> tuple[list[Any], AgenticLoopState]:
    """Create bounded tools that capture one simulation and one loaded policy.

    Args:
        sim: Live ``strands-robots`` simulation built for ``scenario_name``.
        policy: Preloaded policy object reused by every rollout segment.
        scenario_name: Registered workshop scenario to supervise.
        output_dir: Directory in which each policy segment writes an MP4.
        segment_steps: Maximum control steps executed by one tool call.
        max_total_steps: Hard control-step budget across all tool calls.
        record_video: Whether each segment records the scenario video camera.

    Returns:
        The decorated tools for ``Agent(tools=...)`` and their shared state.

    Raises:
        ValueError: If either step budget is not positive or the segment is
            larger than the total budget.
    """

    if segment_steps < 1 or max_total_steps < 1:
        raise ValueError("segment_steps and max_total_steps must be positive.")
    if segment_steps > max_total_steps:
        raise ValueError("segment_steps cannot exceed max_total_steps.")

    scenario = get_scenario(scenario_name)
    video_dir = Path(output_dir).resolve()
    state = AgenticLoopState(
        segment_steps=segment_steps,
        max_total_steps=max_total_steps,
    )

    @tool
    def inspect_pick_task() -> dict[str, Any]:
        """Read the authoritative task metric without moving the robot."""

        diagnostics = scenario.diagnostics(sim)
        return _tool_result(
            {
                "task_success": bool(diagnostics[scenario.success_key]),
                "diagnostics": diagnostics,
                "segments_completed": len(state.attempts),
                "steps_used": state.total_steps_used,
                "remaining_steps": state.remaining_steps,
            }
        )

    @tool
    def run_smolvla_segment() -> dict[str, Any]:
        """Run the loaded SmolVLA for one bounded segment, then report task state."""

        if state.terminal_error is not None:
            return _tool_result(
                {
                    "error": "A previous rollout ended with a software error.",
                    "detail": state.terminal_error,
                    "task_success": scenario.is_success(sim),
                    "remaining_steps": state.remaining_steps,
                },
                status="error",
            )
        if scenario.is_success(sim):
            return _tool_result(
                {
                    "run_status": "skipped",
                    "task_success": True,
                    "steps_used": 0,
                    "remaining_steps": state.remaining_steps,
                    "diagnostics": scenario.diagnostics(sim),
                }
            )
        if state.remaining_steps == 0:
            return _tool_result(
                {
                    "error": "The policy-step budget is exhausted.",
                    "task_success": scenario.is_success(sim),
                    "steps_used": state.total_steps_used,
                    "remaining_steps": 0,
                },
                status="error",
            )

        steps = min(state.segment_steps, state.remaining_steps)
        segment_number = len(state.attempts) + 1
        video_path = video_dir / f"segment_{segment_number:02d}.mp4"
        video = None
        if record_video:
            video_dir.mkdir(parents=True, exist_ok=True)
            video = {
                "path": str(video_path),
                "camera": scenario.video_camera,
                "fps": scenario.fps,
            }

        result = sim.run_policy(
            robot_name=scenario.robot_name,
            policy_object=policy,
            instruction=scenario.instruction,
            n_steps=steps,
            control_frequency=scenario.fps,
            fast_mode=True,
            video=video,
            stop_when=SUCCESS_STOP_WHEN,
        )
        report = _json_payload(result)
        measured_steps = int(report.get("steps_used", report.get("n_steps", 0)) or 0)
        state.total_steps_used += min(measured_steps, state.remaining_steps)

        diagnostics = scenario.diagnostics(sim)
        task_success = bool(diagnostics[scenario.success_key])
        attempt = {
            "segment": segment_number,
            "run_status": result.get("status"),
            "task_success": task_success,
            "steps_requested": steps,
            "steps_used": measured_steps,
            "remaining_steps": state.remaining_steps,
            "stopped_reason": report.get("stopped_reason"),
            "action_errors": report.get("action_errors"),
            "partial_action_failure_rate": report.get("partial_action_failure_rate"),
            "diagnostics": diagnostics,
            "video_path": str(video_path) if record_video else None,
            "error_text": _text_payload(result) if result.get("status") != "success" else None,
        }
        state.attempts.append(attempt)
        if result.get("status") != "success":
            state.terminal_error = attempt["error_text"] or "Unknown run_policy error."
        return _tool_result(
            attempt,
            status="success" if result.get("status") == "success" else "error",
        )

    return [inspect_pick_task, run_smolvla_segment], state


def build_agent_system_prompt(*, segment_steps: int, max_total_steps: int) -> str:
    """Return the high-level supervisor contract read by the Strands Agent."""

    return f"""You are a high-level robot task supervisor.

The SmolVLA policy, not you, controls the robot joints. You may only:
1. call inspect_pick_task to read the deterministic physical task metric;
2. call run_smolvla_segment to let the policy act for up to {segment_steps} control steps.

Follow this closed loop:
- Inspect before acting.
- If task_success is already true, stop.
- Otherwise run one policy segment and inspect the returned task_success.
- If it is false and remaining_steps is positive, inspect once more and run another segment.
- Stop immediately on task_success=true, a software/tool error, or exhausted budget.

The total policy budget is {max_total_steps} steps. Never infer success from
run_status='success': that only means the software loop ran. Report success
only when the tool returns task_success=true. If the budget ends with
task_success=false, clearly report a physical task failure and recommend more
or better demonstrations rather than claiming that agent reasoning repaired
the low-level policy."""


__all__ = [
    "AgenticLoopState",
    "SUCCESS_STOP_WHEN",
    "build_agent_system_prompt",
    "create_agentic_pick_tools",
]
