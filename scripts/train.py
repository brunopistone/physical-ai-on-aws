#!/usr/bin/env python3
"""Infrastructure-neutral SmolVLA fine-tuning entrypoint.

The training logic is intentionally infrastructure-neutral.  Local processes,
SageMaker Training Jobs, EKS Jobs, and HyperPod workers all invoke this same
file.  Infrastructure is responsible only for making three paths available:

* a local LeRobot dataset tree;
* a writable work/checkpoint directory;
* a writable model-export directory.

LeRobot owns the actual optimization loop, Accelerate/DDP/FSDP integration,
checkpointing, and resume semantics.  This wrapper adds:

* one YAML interface shared by every launcher;
* SageMaker path auto-discovery;
* strict dataset/schema validation;
* pinned checkpoint resolution;
* adaptation of the real-data camera schema to the simulation dataset;
* a stable exported-model directory and provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

LOGGER = logging.getLogger("train_smolvla")

EXPECTED_STATE_NAMES = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]
EXPECTED_CAMERA_KEYS = {
    "observation.images.top",
    "observation.images.wrist",
}


@dataclass(frozen=True)
class ResolvedPaths:
    """Concrete dataset, checkpoint-work, and exported-model directories."""

    dataset_dir: Path
    work_dir: Path
    model_dir: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file and require a top-level mapping."""

    if not path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {path}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must contain a YAML mapping: {path}")
    return payload


def _nested(config: dict[str, Any], section: str) -> dict[str, Any]:
    """Return one mapping-valued configuration section."""

    value = config.get(section, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{section}' must be a mapping, got {type(value).__name__}")
    return value


def _first_nonempty(*values: str | os.PathLike[str] | None) -> str | None:
    """Return the first non-empty path-like value, preserving priority order."""

    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return None


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    """Expand and resolve a path relative to the supplied project directory."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_paths(
    config: dict[str, Any],
    *,
    config_path: Path,
    dataset_override: str | None,
    work_override: str | None,
    model_override: str | None,
) -> ResolvedPaths:
    """Resolve portable config paths against CLI, platform environment, and YAML."""

    paths_cfg = _nested(config, "paths")
    project_root = config_path.resolve().parents[1]

    dataset_value = _first_nonempty(
        dataset_override,
        os.environ.get("TRAIN_DATASET_DIR"),
        os.environ.get("SM_CHANNEL_TRAIN"),
        paths_cfg.get("dataset_dir"),
    )
    if dataset_value is None:
        raise ValueError(
            "No dataset path was provided. Set --dataset-dir, TRAIN_DATASET_DIR, "
            "SM_CHANNEL_TRAIN, or paths.dataset_dir in the YAML config."
        )

    # SageMaker creates /opt/ml/checkpoints and /opt/ml/model before the script
    # starts.  Train under a child path because LeRobot correctly refuses to
    # overwrite an already-existing output directory.
    sm_checkpoint_dir = os.environ.get("SM_CHECKPOINT_DIR")
    sm_model_dir = os.environ.get("SM_MODEL_DIR")
    sagemaker_checkpoint_default = (
        f"{sm_checkpoint_dir}/smolvla-run"
        if sm_checkpoint_dir
        else "/opt/ml/checkpoints/smolvla-run"
        if sm_model_dir
        else None
    )
    work_value = _first_nonempty(
        work_override,
        os.environ.get("TRAIN_WORK_DIR"),
        sagemaker_checkpoint_default,
        paths_cfg.get("work_dir"),
        "outputs/smolvla-so100/work",
    )
    model_value = _first_nonempty(
        model_override,
        os.environ.get("TRAIN_MODEL_DIR"),
        f"{sm_model_dir}/smolvla" if sm_model_dir else None,
        paths_cfg.get("model_dir"),
        "outputs/smolvla-so100/model",
    )
    assert work_value is not None and model_value is not None

    return ResolvedPaths(
        dataset_dir=_resolve_path(dataset_value, base_dir=project_root),
        work_dir=_resolve_path(work_value, base_dir=project_root),
        model_dir=_resolve_path(model_value, base_dir=project_root),
    )


def _dataset_info(dataset_dir: Path) -> dict[str, Any]:
    """Read the finalized LeRobot ``meta/info.json`` object."""

    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"{dataset_dir} is not a finalized LeRobot dataset: missing {info_path.relative_to(dataset_dir)}"
        )
    info = json.loads(info_path.read_text())
    if not isinstance(info, dict):
        raise ValueError(f"Invalid JSON object in {info_path}")
    return info


def validate_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Validate the workshop's SO100 state, action, camera, and artifact contract."""

    info = _dataset_info(dataset_dir)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("Dataset meta/info.json has no feature mapping.")

    problems: list[str] = []
    if int(info.get("total_episodes", 0)) < 1:
        problems.append("the dataset has no episodes")
    if int(info.get("total_frames", 0)) < 1:
        problems.append("the dataset has no frames")
    if int(info.get("fps", 0)) != 30:
        problems.append(f"fps is {info.get('fps')}, expected 30")
    if info.get("robot_type") != "so100":
        problems.append(f"robot_type is {info.get('robot_type')!r}, expected 'so100'")

    state = features.get("observation.state", {})
    action = features.get("action", {})
    if state.get("shape") != [6]:
        problems.append(f"observation.state shape is {state.get('shape')}, expected [6]")
    if action.get("shape") != [6]:
        problems.append(f"action shape is {action.get('shape')}, expected [6]")
    if state.get("names") != EXPECTED_STATE_NAMES:
        problems.append(
            f"observation.state names are {state.get('names')}, expected {EXPECTED_STATE_NAMES}"
        )
    if action.get("names") != EXPECTED_STATE_NAMES:
        problems.append(f"action names are {action.get('names')}, expected {EXPECTED_STATE_NAMES}")

    camera_keys = {key for key in features if key.startswith("observation.images.")}
    if camera_keys != EXPECTED_CAMERA_KEYS:
        problems.append(
            f"camera keys are {sorted(camera_keys)}, expected {sorted(EXPECTED_CAMERA_KEYS)}"
        )

    required_artifacts = [
        dataset_dir / "meta" / "stats.json",
        dataset_dir / "meta" / "tasks.parquet",
    ]
    required_artifacts.extend(
        dataset_dir / "videos" / key / "chunk-000" / "file-000.mp4"
        for key in EXPECTED_CAMERA_KEYS
    )
    missing = [str(path.relative_to(dataset_dir)) for path in required_artifacts if not path.is_file()]
    if missing:
        problems.append(f"required artifacts are missing: {missing}")

    if problems:
        raise ValueError("Dataset contract validation failed:\n  - " + "\n  - ".join(problems))

    LOGGER.info(
        "Dataset contract OK: %s episodes, %s frames, 30 FPS, SO100 6D, top+wrist",
        info["total_episodes"],
        info["total_frames"],
    )
    return info


def _dataset_fingerprint(dataset_dir: Path) -> str:
    """Hash the metadata files that identify the exact training dataset."""

    digest = hashlib.sha256()
    for relative in ("meta/info.json", "meta/stats.json", "meta/tasks.parquet"):
        path = dataset_dir / relative
        digest.update(relative.encode())
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def select_device(requested: str) -> str:
    """Resolve ``auto`` to CUDA, Apple MPS, or CPU in preference order."""

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _snapshot_model(model_id: str, revision: str | None) -> Path:
    """Resolve a local checkpoint or download a pinned Hugging Face snapshot."""

    candidate = Path(model_id).expanduser()
    if candidate.is_dir():
        return candidate.resolve()

    from huggingface_hub import snapshot_download

    LOGGER.info("Resolving pinned model %s@%s", model_id, revision or "main")
    snapshot = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.jinja",
        ],
    )
    return Path(snapshot).resolve()


def _stage_model_for_dataset(
    source: Path,
    dataset_info: dict[str, Any],
    camera_mapping: dict[str, str],
) -> Path:
    """Create a lightweight local checkpoint view with dataset-native features.

    The real-data checkpoint has camera1/camera2/camera3.  The simulation has
    wrist/top.  We preserve the semantic camera order (camera1=wrist,
    camera2=top), remove the unused third camera, and replace only config.json.
    All weights and processor files remain symlinked to the pinned snapshot.
    """
    config_path = source / "config.json"
    weights_path = source / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"Pretrained checkpoint must contain config.json and model.safetensors: {source}"
        )

    config = json.loads(config_path.read_text())
    if config.get("type") != "smolvla":
        raise ValueError(f"Expected a SmolVLA checkpoint, found type={config.get('type')!r}")

    old_inputs = config.get("input_features", {})
    old_cameras = {
        key for key, feature in old_inputs.items() if feature.get("type") == "VISUAL"
    }
    mapped_targets = set(camera_mapping.values())
    if not mapped_targets.issubset(old_cameras):
        raise ValueError(
            f"camera_mapping targets {sorted(mapped_targets)} are not all present in "
            f"the checkpoint cameras {sorted(old_cameras)}"
        )
    if set(camera_mapping) != EXPECTED_CAMERA_KEYS:
        raise ValueError(
            f"camera_mapping sources must be {sorted(EXPECTED_CAMERA_KEYS)}, "
            f"got {sorted(camera_mapping)}"
        )

    features = dataset_info["features"]
    input_features: dict[str, Any] = {
        "observation.state": {
            "type": "STATE",
            "shape": features["observation.state"]["shape"],
        }
    }
    # Dict insertion order controls the image order seen by SmolVLA.  Sort by
    # the original checkpoint key so camera1 remains first.
    for dataset_key, _checkpoint_key in sorted(camera_mapping.items(), key=lambda item: item[1]):
        input_features[dataset_key] = {
            "type": "VISUAL",
            "shape": features[dataset_key]["shape"],
        }

    config["input_features"] = input_features
    config["output_features"] = {
        "action": {
            "type": "ACTION",
            "shape": features["action"]["shape"],
        }
    }

    stage = Path(tempfile.mkdtemp(prefix="smolvla-dataset-schema-"))
    for child in source.iterdir():
        if child.name == "config.json":
            continue
        target = stage / child.name
        try:
            target.symlink_to(child.resolve(), target_is_directory=child.is_dir())
        except OSError:
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
    (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    LOGGER.info(
        "Staged checkpoint schema: %s -> %s; removed cameras=%s",
        source,
        stage,
        sorted(old_cameras - mapped_targets),
    )
    return stage


def _bool_arg(value: bool) -> str:
    """Render a Python boolean in the form expected by LeRobot CLI overrides."""

    return "true" if value else "false"


def _training_cli_args(
    config: dict[str, Any],
    paths: ResolvedPaths,
    *,
    policy_path: str | None,
    device: str,
    steps_override: int | None,
    batch_override: int | None,
    resume_from: str | None,
) -> list[str]:
    """Translate the portable YAML configuration into native LeRobot CLI arguments."""

    model_cfg = _nested(config, "model")
    train_cfg = _nested(config, "training")
    data_cfg = _nested(config, "dataset")
    tracking_cfg = _nested(config, "tracking")

    steps = int(steps_override or train_cfg.get("steps", 1000))
    batch_size = int(batch_override or train_cfg.get("batch_size", 4))
    num_workers_value = train_cfg.get("num_workers", "auto")
    if num_workers_value == "auto":
        num_workers = min(8, max(1, (os.cpu_count() or 4) // 2)) if device == "cuda" else 0
    else:
        num_workers = int(num_workers_value)
    use_amp_value = train_cfg.get("use_amp", "auto")
    use_amp = device == "cuda" if use_amp_value == "auto" else bool(use_amp_value)

    args = [
        f"--dataset.repo_id={data_cfg.get('repo_id', 'local/so100_sim_pickplace')}",
        f"--dataset.root={paths.dataset_dir}",
        f"--dataset.video_backend={data_cfg.get('video_backend', 'pyav')}",
        f"--dataset.return_uint8={_bool_arg(bool(data_cfg.get('return_uint8', True)))}",
        f"--dataset.eval_split={float(data_cfg.get('eval_split', 0.125))}",
        f"--output_dir={paths.work_dir}",
        f"--job_name={train_cfg.get('job_name', 'smolvla-so100-sim')}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--num_workers={num_workers}",
        f"--prefetch_factor={int(train_cfg.get('prefetch_factor', 2))}",
        f"--persistent_workers={_bool_arg(bool(train_cfg.get('persistent_workers', num_workers > 0)))}",
        f"--seed={int(train_cfg.get('seed', 42))}",
        f"--log_freq={int(train_cfg.get('log_freq', 10))}",
        f"--save_checkpoint={_bool_arg(bool(train_cfg.get('save_checkpoint', True)))}",
        f"--save_freq={int(train_cfg.get('save_freq', 250))}",
        f"--eval_steps={int(train_cfg.get('eval_steps', 100))}",
        f"--max_eval_samples={int(train_cfg.get('max_eval_samples', 256))}",
        "--env_eval_freq=0",
        "--use_policy_training_preset=true",
        f"--wandb.enable={_bool_arg(bool(tracking_cfg.get('wandb', False)))}",
    ]

    if resume_from:
        args.extend([f"--resume=true", f"--config_path={resume_from}"])
    else:
        if policy_path is None:
            raise ValueError("policy_path is required for a fresh fine-tuning run")
        args.extend(
            [
                f"--policy.path={policy_path}",
                f"--policy.device={device}",
                f"--policy.use_amp={_bool_arg(use_amp)}",
                "--policy.push_to_hub=false",
                f"--policy.optimizer_lr={float(train_cfg.get('learning_rate', 1e-4))}",
                f"--policy.scheduler_warmup_steps={int(train_cfg.get('warmup_steps', 100))}",
                f"--policy.scheduler_decay_steps={int(train_cfg.get('decay_steps', steps))}",
                f"--policy.scheduler_decay_lr={float(train_cfg.get('decay_lr', 2.5e-6))}",
                f"--policy.freeze_vision_encoder={_bool_arg(bool(model_cfg.get('freeze_vision_encoder', True)))}",
                f"--policy.train_expert_only={_bool_arg(bool(model_cfg.get('train_expert_only', True)))}",
                f"--policy.train_state_proj={_bool_arg(bool(model_cfg.get('train_state_proj', True)))}",
            ]
        )
    return args


def _find_final_model(work_dir: Path) -> Path:
    """Locate and validate the final LeRobot ``pretrained_model`` checkpoint."""

    last = work_dir / "checkpoints" / "last"
    if not last.exists():
        candidates = sorted((work_dir / "checkpoints").glob("[0-9]*"))
        if not candidates:
            raise FileNotFoundError(f"No LeRobot checkpoint was written under {work_dir}")
        last = candidates[-1]
    model = last.resolve() / "pretrained_model"
    if not (model / "config.json").is_file() or not (model / "model.safetensors").is_file():
        raise FileNotFoundError(f"Final checkpoint is incomplete: {model}")
    return model


def _export_model(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
    manifest: dict[str, Any],
) -> None:
    """Copy the final checkpoint to a stable inference directory with provenance."""

    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Model export directory already exists: {destination}. "
                "Choose another --model-dir or pass --overwrite-export."
            )
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=False)
    (destination / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info("Exported fine-tuned model to %s", destination)


def _redact_cli(args: list[str]) -> list[str]:
    """Remove token- or secret-bearing arguments before logging provenance."""

    return [arg for arg in args if "token" not in arg.lower() and "secret" not in arg.lower()]


def build_parser() -> argparse.ArgumentParser:
    """Build the portable wrapper CLI and its optional YAML overrides."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Portable YAML training config.")
    parser.add_argument("--dataset-dir", help="Override dataset root.")
    parser.add_argument("--work-dir", help="Override LeRobot work/checkpoint directory.")
    parser.add_argument("--model-dir", help="Override final exported model directory.")
    parser.add_argument("--model-id", help="Override model.id from YAML.")
    parser.add_argument("--model-revision", help="Override model.revision from YAML.")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--steps", type=int, help="Override training steps.")
    parser.add_argument("--batch-size", type=int, help="Override per-process batch size.")
    parser.add_argument("--resume-from", help="LeRobot checkpoint config/model path to resume.")
    parser.add_argument("--overwrite-export", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/schema and print the resolved LeRobot command without downloading or training.",
    )
    return parser


def train(lerobot_args: list[str]) -> None:
    """Run LeRobot's native trainer in-process with distributed env variables intact."""

    # LeRobot's Accelerate integration reads WORLD_SIZE/RANK/LOCAL_RANK from
    # torchrun or the platform launcher.  Calling main() in-process preserves
    # those variables and avoids a nested launcher.
    from lerobot.scripts import lerobot_train

    original_argv = sys.argv
    try:
        sys.argv = ["lerobot-train", *lerobot_args]
        lerobot_train.main()
    finally:
        sys.argv = original_argv


def main() -> None:
    """Validate inputs, launch training, and export the rank-zero model artifact."""

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cli = build_parser().parse_args()
    config_path = Path(cli.config).expanduser().resolve()
    config = _load_yaml(config_path)
    model_cfg = _nested(config, "model")
    model_id = cli.model_id or model_cfg.get("id")
    revision = cli.model_revision or model_cfg.get("revision")
    if not model_id and not cli.resume_from:
        raise ValueError("Set model.id in YAML or pass --model-id.")

    paths = resolve_paths(
        config,
        config_path=config_path,
        dataset_override=cli.dataset_dir,
        work_override=cli.work_dir,
        model_override=cli.model_dir,
    )
    dataset_info = validate_dataset(paths.dataset_dir)
    device = select_device(cli.device)

    camera_mapping = model_cfg.get(
        "camera_mapping",
        {
            "observation.images.wrist": "observation.images.camera1",
            "observation.images.top": "observation.images.camera2",
        },
    )
    if not isinstance(camera_mapping, dict):
        raise TypeError("model.camera_mapping must be a mapping.")

    policy_path: str | None = None
    if not cli.resume_from and not cli.dry_run:
        source = _snapshot_model(str(model_id), str(revision) if revision else None)
        policy_path = str(_stage_model_for_dataset(source, dataset_info, camera_mapping))
    elif not cli.resume_from:
        # Dry-run does not stage/download the pinned snapshot. Keep policy.path
        # syntactically valid and report the revision separately below.
        policy_path = str(model_id)

    lerobot_args = _training_cli_args(
        config,
        paths,
        policy_path=policy_path,
        device=device,
        steps_override=cli.steps,
        batch_override=cli.batch_size,
        resume_from=cli.resume_from,
    )

    LOGGER.info("Resolved dataset: %s", paths.dataset_dir)
    LOGGER.info("Resolved work dir: %s", paths.work_dir)
    LOGGER.info("Resolved model dir: %s", paths.model_dir)
    LOGGER.info("Resolved device: %s", device)
    LOGGER.info("LeRobot argv:\n  %s", "\n  ".join(_redact_cli(lerobot_args)))

    if cli.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry-run-ok",
                    "dataset_dir": str(paths.dataset_dir),
                    "work_dir": str(paths.work_dir),
                    "model_dir": str(paths.model_dir),
                    "device": device,
                    "model_revision": revision,
                    "dataset_fingerprint": _dataset_fingerprint(paths.dataset_dir),
                    "lerobot_args": _redact_cli(lerobot_args),
                },
                indent=2,
            )
        )
        return

    if not cli.resume_from and paths.work_dir.exists():
        raise FileExistsError(
            f"Training work directory already exists: {paths.work_dir}. "
            "Choose another --work-dir or resume from its last checkpoint."
        )

    paths.work_dir.parent.mkdir(parents=True, exist_ok=True)

    train(lerobot_args)

    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return

    final_model = _find_final_model(paths.work_dir)
    manifest = {
        "base_model": model_id,
        "base_revision": revision,
        "dataset_repo_id": _nested(config, "dataset").get(
            "repo_id", "local/so100_sim_pickplace"
        ),
        "dataset_fingerprint": _dataset_fingerprint(paths.dataset_dir),
        "dataset_episodes": dataset_info["total_episodes"],
        "dataset_frames": dataset_info["total_frames"],
        "camera_keys": sorted(EXPECTED_CAMERA_KEYS),
        "state_action_units": "radians",
        "source_checkpoint": str(final_model.relative_to(paths.work_dir)),
        "lerobot_args": _redact_cli(lerobot_args),
    }
    _export_model(
        final_model,
        paths.model_dir,
        overwrite=cli.overwrite_export,
        manifest=manifest,
    )


if __name__ == "__main__":
    main()
