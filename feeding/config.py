"""Central configuration dataclasses and YAML loader for the feeding runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VisionConfig:
    camera_index: int = 0
    open_threshold: float = 0.12
    open_hold_seconds: float = 0.5
    show_camera: bool = True


@dataclass
class ControlConfig:
    loop_hz: float = 20.0
    max_vision_stale_seconds: float = 0.25
    rearm_cooldown_seconds: float = 2.0
    require_close_before_rearm: bool = True
    approach_update_period_s: float = 0.25


@dataclass
class SafetyConfig:
    max_linear_speed_mps: float = 0.08
    neutral_pose_xyz: tuple[float, float, float] = (0.22, 0.0, 0.18)
    workspace_min_xyz: tuple[float, float, float] = (0.12, -0.18, 0.05)
    workspace_max_xyz: tuple[float, float, float] = (0.40, 0.18, 0.30)


@dataclass
class PlanningConfig:
    mouth_target_x_m: float = 0.20
    y_scale_m: float = 0.16
    z_scale_m: float = 0.12


@dataclass
class LeRobotConfig:
    robot_type: str = "so101_follower"
    port: str = "/dev/tty.usbmodem5AB90674581"
    robot_id: str = "synopsys2026"
    motors_csv: str = "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex"
    max_relative_target: float = 8.0
    move_duration_s: float = 1.6
    disable_torque_on_disconnect: bool = False
    release_torque_on_shutdown: bool = True
    persist_neutral_on_shutdown: bool = False
    neutral_config_path: str = "configs/feeding_default.yaml"
    use_joint_midpoint_neutral: bool = True
    use_horizontal_ik: bool = True
    link_l1_m: float = 0.165
    link_l2_m: float = 0.190
    spoon_offset_forward_m: float = 0.110
    spoon_offset_up_m: float = 0.040
    compact_on_shutdown: bool = True
    compact_move_duration_s: float = 1.2
    compact_joints_deg: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.0,
            "shoulder_lift": -105.0,
            "elbow_flex": 125.0,
            "wrist_flex": -20.0,
        }
    )
    neutral_joints_deg: dict[str, float] = field(default_factory=dict)
    ik_motor_sign: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_lift": 1.0,
            "elbow_flex": 1.0,
            "wrist_flex": 1.0,
        }
    )
    ik_motor_offset_deg: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
        }
    )
    joint_limits_deg: dict[str, list[float]] = field(
        default_factory=lambda: {
            "shoulder_pan": [-150.0, 150.0],
            "shoulder_lift": [-140.0, 140.0],
            "elbow_flex": [-140.0, 140.0],
            "wrist_flex": [-120.0, 120.0],
        }
    )
    approach_gains_deg_per_m: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 20.0,
            "shoulder_lift": -140.0,
            "elbow_flex": -180.0,
            "wrist_flex": 0.0,
        }
    )
    forward_reach_deg_per_m: float = 120.0
    max_forward_delta_deg: float = 22.0
    shoulder_forward_ratio: float = -0.45
    elbow_forward_ratio: float = 0.75


@dataclass
class AppConfig:
    backend: str = "fake"
    vision: VisionConfig = field(default_factory=VisionConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    lerobot: LeRobotConfig = field(default_factory=LeRobotConfig)


def _merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> AppConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults = {
        "backend": "fake",
        "vision": VisionConfig().__dict__,
        "control": ControlConfig().__dict__,
        "safety": SafetyConfig().__dict__,
        "planning": PlanningConfig().__dict__,
        "lerobot": LeRobotConfig().__dict__,
    }
    merged = _merge_dict(defaults, raw)

    return AppConfig(
        backend=merged["backend"],
        vision=VisionConfig(**merged["vision"]),
        control=ControlConfig(**merged["control"]),
        safety=SafetyConfig(**merged["safety"]),
        planning=PlanningConfig(**merged["planning"]),
        lerobot=LeRobotConfig(**merged["lerobot"]),
    )
