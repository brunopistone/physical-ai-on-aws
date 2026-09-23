"""Bounded Strands Agent tools for supervising the SO100 pick-place policy.

The language model operates at task level: it may inspect deterministic task
state and request another policy segment. The SmolVLA policy remains the only
component that emits joint actions, and ``strands-robots`` remains the owner of
the high-frequency observation-inference-action loop.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from strands import tool

from scenarios import get_scenario
from scenarios.pick_place import BOX_CENTER

SUPPORTED_SKILLS = {
    "pick_and_place": {
        "targets": ["cube"],
        "destinations": ["box"],
        "instruction_template": "Pick up the {target} and place it in the {destination}.",
    }
}


def create_agent_model(provider: str) -> Any:
    """Create the task-level reasoning model for Ollama or Amazon Bedrock.

    Args:
        provider: Model provider name: ``"OLLAMA"`` or ``"BEDROCK"``.

    Returns:
        A Strands model instance configured from environment variables.

    Raises:
        RuntimeError: If Ollama is unreachable or its configured model is absent.
        ValueError: If ``provider`` is not one of the supported names.
    """

    normalized = provider.strip().upper()
    if normalized == "OLLAMA":
        from strands.models.ollama import OllamaModel

        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        if "://" not in host:
            host = "http://" + host
        model_id = os.environ.get("STRANDS_LOCAL_MODEL", "qwen3:4b")
        try:
            with urlopen(f"{host}/api/tags", timeout=3) as response:
                tags = json.load(response)
        except (URLError, TimeoutError) as error:
            raise RuntimeError(
                f"Cannot reach Ollama at {host}. Start it with: ollama serve"
            ) from error

        available = {
            model_name
            for item in tags.get("models", [])
            if (model_name := item.get("name") or item.get("model"))
        }
        if model_id not in available:
            raise RuntimeError(
                f"Local model {model_id!r} is not installed. "
                f"Run: ollama pull {model_id}. Available: {sorted(available)}"
            )
        return OllamaModel(
            host=host,
            model_id=model_id,
            temperature=0.0,
            keep_alive="15m",
        )

    if normalized == "BEDROCK":
        import boto3
        from strands.models import BedrockModel

        region = (
            os.environ.get("AWS_REGION")
            or boto3.Session().region_name
            or "us-east-1"
        )
        model_id = os.environ.get(
            "STRANDS_BEDROCK_MODEL_ID",
            "global.anthropic.claude-sonnet-4-6",
        )
        return BedrockModel(model_id=model_id, region_name=region)

    raise ValueError(
        f"Unsupported agent model provider {provider!r}. "
        "Choose 'OLLAMA' or 'BEDROCK'."
    )


@dataclass
class AgenticLoopState:
    """Track the bounded policy budget and machine-readable attempt history."""

    segment_steps: int
    max_total_steps: int
    total_steps_used: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    escalations: list[dict[str, Any]] = field(default_factory=list)
    terminal_error: str | None = None
    escalated: bool = False

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
    segment_steps: int = 400,
    max_total_steps: int = 800,
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
    def inspect_robot_workspace() -> dict[str, Any]:
        """Discover capabilities and authoritative task state without moving."""

        diagnostics = scenario.diagnostics(sim)
        return _tool_result(
            {
                "task_success": bool(diagnostics[scenario.success_key]),
                "diagnostics": diagnostics,
                "available_entities": {
                    "targets": ["cube"],
                    "destinations": ["box"],
                },
                "supported_skills": SUPPORTED_SKILLS,
                "segments_completed": len(state.attempts),
                "escalated": state.escalated,
                "steps_used": state.total_steps_used,
                "remaining_steps": state.remaining_steps,
            }
        )

    @tool
    def execute_robot_skill(
        skill: str,
        target: str,
        destination: str,
    ) -> dict[str, Any]:
        """Execute one supported skill through the local SmolVLA policy.

        Args:
            skill: Skill name discovered through ``inspect_robot_workspace``.
            target: Object to manipulate, using its workspace name.
            destination: Named destination for the object.
        """

        contract = SUPPORTED_SKILLS.get(skill)
        if contract is None:
            return _tool_result(
                {
                    "error": f"Unsupported skill: {skill!r}",
                    "supported_skills": sorted(SUPPORTED_SKILLS),
                },
                status="error",
            )
        if target not in contract["targets"] or destination not in contract["destinations"]:
            return _tool_result(
                {
                    "error": "The requested target or destination is unsupported.",
                    "supported_targets": contract["targets"],
                    "supported_destinations": contract["destinations"],
                },
                status="error",
            )
        if state.escalated:
            return _tool_result(
                {
                    "error": "Physical execution is disabled after escalation.",
                    "task_success": False,
                },
                status="error",
            )
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

        policy_instruction = contract["instruction_template"].format(
            target=target,
            destination=destination,
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
            instruction=policy_instruction,
            n_steps=steps,
            control_frequency=scenario.fps,
            fast_mode=True,
            video=video,
        )
        report = _json_payload(result)
        measured_steps = int(report.get("steps_used", report.get("n_steps", 0)) or 0)
        state.total_steps_used += min(measured_steps, state.remaining_steps)

        diagnostics = scenario.diagnostics(sim)
        task_success = bool(diagnostics[scenario.success_key])
        attempt = {
            "segment": segment_number,
            "skill": skill,
            "target": target,
            "destination": destination,
            "policy_instruction": policy_instruction,
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

    @tool
    def request_human_help(reason: str) -> dict[str, Any]:
        """Stop physical execution and record why assistance is required.

        Args:
            reason: Concise explanation of the unsupported goal or repeated failure.
        """

        escalation = {
            "reason": reason,
            "diagnostics": scenario.diagnostics(sim),
            "steps_used": state.total_steps_used,
        }
        state.escalations.append(escalation)
        state.escalated = True
        return _tool_result({"escalated": True, **escalation})

    return [inspect_robot_workspace, execute_robot_skill, request_human_help], state


def build_agent_system_prompt(*, segment_steps: int, max_total_steps: int) -> str:
    """Return the high-level supervisor contract read by the Strands Agent."""

    return f"""You are the local high-level supervisor running on a robot companion computer.

Interpret the user's free-form goal, but never command robot joints yourself.
The local SmolVLA policy owns joint actions. You may only use these tools:
1. inspect_robot_workspace: discover entities, supported skills, physical state, and remaining budget;
2. execute_robot_skill: choose a supported skill, target, and destination;
3. request_human_help: stop physical execution when the goal is unsupported or recovery is exhausted.

Always inspect before acting. Map the goal only onto capabilities returned by
inspection. Execute one segment, inspect its deterministic result, and retry
at most once while budget remains. Request human help for an unsupported goal,
a tool failure, or an unsuccessful retry.

The hard policy budget is {max_total_steps} steps; each segment is at most
{segment_steps} steps. run_status='success' means only that software executed.
Report physical success only when task_success=true."""


__all__ = [
    "AgenticLoopState",
    "SUPPORTED_SKILLS",
    "build_agent_system_prompt",
    "create_agent_model",
    "create_agentic_pick_tools",
]
