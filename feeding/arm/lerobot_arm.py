from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
import math

from feeding.arm.base import ArmBackend
from feeding.config import LeRobotConfig, SafetyConfig
from feeding.kinematics import Planar3LinkHorizontalIK
from feeding.types import Pose3D
import yaml


@dataclass
class _MotionPlan:
    target: dict[str, float]
    done_at: float


class LeRobotArmBackend(ArmBackend):
    """LeRobot backend for SO-101 custom 4-motor follower builds."""

    def __init__(self, safety_cfg: SafetyConfig, robot_cfg: LeRobotConfig) -> None:
        self.safety_cfg = safety_cfg
        self.robot_cfg = robot_cfg
        self.robot = None
        self._motion: _MotionPlan | None = None
        self._neutral_joints: dict[str, float] = {}
        self._joint_order = [m.strip() for m in robot_cfg.motors_csv.split(",") if m.strip()]
        self._next_command_time = 0.0
        self._joint_tolerance_deg = 1.5
        self._command_period_s = 0.12
        self._ik = Planar3LinkHorizontalIK(
            l1_m=self.robot_cfg.link_l1_m,
            l2_m=self.robot_cfg.link_l2_m,
            l3_forward_m=self.robot_cfg.spoon_offset_forward_m,
            l3_up_m=self.robot_cfg.spoon_offset_up_m,
        )
        self._neutral_tip_x = 0.0
        self._neutral_tip_z = 0.0
        self._ik_phi_ref = 0.0
        self._last_ik_q = (0.0, 0.0, 0.0)

    def _to_kin_deg(self, joint: str, motor_deg: float) -> float:
        sign = float(self.robot_cfg.ik_motor_sign.get(joint, 1.0))
        if abs(sign) < 1e-9:
            sign = 1.0
        offset = float(self.robot_cfg.ik_motor_offset_deg.get(joint, 0.0))
        return sign * (motor_deg - offset)

    def _to_motor_deg(self, joint: str, kin_deg: float) -> float:
        sign = float(self.robot_cfg.ik_motor_sign.get(joint, 1.0))
        if abs(sign) < 1e-9:
            sign = 1.0
        offset = float(self.robot_cfg.ik_motor_offset_deg.get(joint, 0.0))
        return (kin_deg / sign) + offset

    def start(self) -> None:
        if self.robot_cfg.robot_type != "so101_follower":
            raise ValueError(f"Unsupported robot_type: {self.robot_cfg.robot_type}")
        os.environ["LEROBOT_SO_FOLLOWER_MOTORS"] = self.robot_cfg.motors_csv
        from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        cfg = SOFollowerRobotConfig(
            port=self.robot_cfg.port,
            id=self.robot_cfg.robot_id,
            max_relative_target=self.robot_cfg.max_relative_target,
            disable_torque_on_disconnect=self.robot_cfg.disable_torque_on_disconnect,
        )
        self.robot = make_robot_from_config(cfg)
        self.robot.connect()
        self._ensure_torque_enabled()

        if self.robot_cfg.neutral_joints_deg:
            self._neutral_joints = dict(self.robot_cfg.neutral_joints_deg)
            logging.info("Using neutral joints from config.")
        elif self.robot_cfg.use_joint_midpoint_neutral:
            self._neutral_joints = self._midpoint_joints_from_limits()
            logging.info("Using midpoint neutral from configured joint limits.")
        else:
            self._neutral_joints = self._read_current_joints()
            logging.info("Neutral joints inferred from current arm pose.")

        self._update_neutral_ik_reference()

    def _ensure_torque_enabled(self) -> None:
        if self.robot is None:
            return
        for motor in self._joint_order:
            try:
                self.robot.bus.enable_torque(motor, num_retry=2)
            except Exception as e:
                logging.warning("Failed to enable torque on %s: %s", motor, e)

    def shutdown(self) -> None:
        if self.robot is not None:
            try:
                current = self._read_current_joints()
                if current and self.robot_cfg.persist_neutral_on_shutdown:
                    self._persist_neutral_to_config(current)

                if self.robot_cfg.compact_on_shutdown:
                    compact_target = {
                        name: float(self.robot_cfg.compact_joints_deg.get(name, current.get(name, 0.0)))
                        for name in self._joint_order
                    }
                    now = time.monotonic()
                    self._send_joint_target(compact_target, now)
                    end_time = now + max(self.robot_cfg.compact_move_duration_s, 0.2)
                    while time.monotonic() < end_time:
                        self.tick(time.monotonic())
                        time.sleep(0.03)

                if self.robot_cfg.release_torque_on_shutdown:
                    for motor in self._joint_order:
                        try:
                            self.robot.bus.disable_torque(motor, num_retry=1)
                        except Exception as e:
                            logging.warning("Failed to release torque on %s: %s", motor, e)
                self.robot.disconnect()
            except Exception as e:
                logging.warning("Robot disconnect raised error; continuing shutdown: %s", e)
            self.robot = None

    def _persist_neutral_to_config(self, joints: dict[str, float]) -> None:
        cfg_path = Path(self.robot_cfg.neutral_config_path)
        if not cfg_path.is_absolute():
            cfg_path = Path.cwd() / cfg_path
        if not cfg_path.exists():
            logging.warning("Neutral config path not found: %s", cfg_path)
            return
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            if not isinstance(raw, dict):
                raw = {}
            if "lerobot" not in raw or not isinstance(raw["lerobot"], dict):
                raw["lerobot"] = {}
            raw["lerobot"]["neutral_joints_deg"] = {k: round(v, 3) for k, v in joints.items()}
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(raw, f, sort_keys=False)
            logging.info("Saved current joint pose as neutral to %s", cfg_path)
        except Exception as e:
            logging.warning("Failed to persist neutral pose: %s", e)

    def _read_current_joints(self) -> dict[str, float]:
        if self.robot is None:
            return {}
        obs = self.robot.get_observation()
        out = {}
        for name in self._joint_order:
            key = f"{name}.pos"
            if key in obs:
                out[name] = float(obs[key])
        return out

    def _clamp_joint(self, name: str, value_deg: float) -> float:
        limits = self.robot_cfg.joint_limits_deg.get(name)
        if not limits or len(limits) != 2:
            return value_deg
        lo, hi = float(limits[0]), float(limits[1])
        return min(max(value_deg, lo), hi)

    def _issue_action(self, target: dict[str, float]) -> dict[str, float]:
        if self.robot is None:
            raise RuntimeError("Robot not connected.")
        action = {f"{name}.pos": self._clamp_joint(name, val) for name, val in target.items()}
        self.robot.send_action(action)
        return {k: action[f"{k}.pos"] for k in target}

    def _send_joint_target(self, target: dict[str, float], now: float) -> None:
        clamped_target = self._issue_action(target)
        self._motion = _MotionPlan(target=clamped_target, done_at=now + self.robot_cfg.move_duration_s)
        self._next_command_time = now + self._command_period_s

    def _midpoint_joints_from_limits(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in self._joint_order:
            limits = self.robot_cfg.joint_limits_deg.get(name)
            if limits and len(limits) == 2:
                out[name] = 0.5 * (float(limits[0]) + float(limits[1]))
            else:
                out[name] = 0.0
        return out

    def _target_from_pose(self, target: Pose3D) -> dict[str, float]:
        if not self._neutral_joints:
            raise RuntimeError("Neutral joints not initialized.")

        neutral_x = float(self.safety_cfg.neutral_pose_xyz[0])
        neutral_y = float(self.safety_cfg.neutral_pose_xyz[1])
        neutral_z = float(self.safety_cfg.neutral_pose_xyz[2])
        dx = target.x - neutral_x
        dy = target.y - neutral_y
        dz = target.z - neutral_z

        out: dict[str, float] = dict(self._neutral_joints)

        # Shoulder pan remains a direct lateral mapping.
        if "shoulder_pan" in out:
            pan_gain = float(self.robot_cfg.approach_gains_deg_per_m.get("shoulder_pan", 0.0))
            out["shoulder_pan"] = out["shoulder_pan"] + pan_gain * dy

        if (
            self.robot_cfg.use_horizontal_ik
            and all(k in out for k in ("shoulder_lift", "elbow_flex", "wrist_flex"))
        ):
            current = self._read_current_joints()
            if current and all(k in current for k in ("shoulder_lift", "elbow_flex", "wrist_flex")):
                seed_q2 = math.radians(self._to_kin_deg("elbow_flex", current["elbow_flex"]))
            else:
                seed_q2 = self._last_ik_q[1]
            sol = None
            # Try full request first, then progressively back off toward neutral if unreachable.
            for scale in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25):
                desired_x = self._neutral_tip_x + scale * dx
                desired_z = self._neutral_tip_z + scale * dz
                sol = self._ik.solve(
                    x_tip=desired_x,
                    z_tip=desired_z,
                    phi=self._ik_phi_ref,
                    seed_q2=seed_q2,
                )
                if sol is not None:
                    break
            if sol is not None:
                out["shoulder_lift"] = self._to_motor_deg("shoulder_lift", math.degrees(sol.q1))
                out["elbow_flex"] = self._to_motor_deg("elbow_flex", math.degrees(sol.q2))
                out["wrist_flex"] = self._to_motor_deg("wrist_flex", math.degrees(sol.q3))
                self._last_ik_q = (sol.q1, sol.q2, sol.q3)
                return out
            logging.warning("IK solve failed for target; holding neutral on lift/elbow/wrist.")
            out["shoulder_lift"] = self._neutral_joints["shoulder_lift"]
            out["elbow_flex"] = self._neutral_joints["elbow_flex"]
            out["wrist_flex"] = self._neutral_joints["wrist_flex"]
            return out

        # Fallback if IK disabled or infeasible.
        has_triplet = all(k in out for k in ("shoulder_lift", "elbow_flex", "wrist_flex"))
        if has_triplet:
            shoulder_gain = float(self.robot_cfg.approach_gains_deg_per_m.get("shoulder_lift", 0.0))
            elbow_gain = float(self.robot_cfg.approach_gains_deg_per_m.get("elbow_flex", 0.0))

            forward_mag = float(self.robot_cfg.forward_reach_deg_per_m) * dx
            max_forward = float(self.robot_cfg.max_forward_delta_deg)
            forward_mag = min(max(forward_mag, -max_forward), max_forward)

            shoulder_delta = shoulder_gain * dz + float(self.robot_cfg.shoulder_forward_ratio) * forward_mag
            elbow_delta = elbow_gain * dz + float(self.robot_cfg.elbow_forward_ratio) * forward_mag
            wrist_delta = -(shoulder_delta + elbow_delta)

            out["shoulder_lift"] = self._neutral_joints["shoulder_lift"] + shoulder_delta
            out["elbow_flex"] = self._neutral_joints["elbow_flex"] + elbow_delta
            out["wrist_flex"] = self._neutral_joints["wrist_flex"] + wrist_delta
        else:
            for name in ("shoulder_lift", "elbow_flex", "wrist_flex"):
                if name in out:
                    gain_per_m = float(self.robot_cfg.approach_gains_deg_per_m.get(name, 0.0))
                    out[name] = self._neutral_joints[name] + gain_per_m * dz
        return out

    def _update_neutral_ik_reference(self) -> None:
        if not all(k in self._neutral_joints for k in ("shoulder_lift", "elbow_flex", "wrist_flex")):
            return
        q1 = math.radians(self._to_kin_deg("shoulder_lift", float(self._neutral_joints["shoulder_lift"])))
        q2 = math.radians(self._to_kin_deg("elbow_flex", float(self._neutral_joints["elbow_flex"])))
        q3 = math.radians(self._to_kin_deg("wrist_flex", float(self._neutral_joints["wrist_flex"])))
        self._ik_phi_ref = q1 + q2 + q3
        self._neutral_tip_x, self._neutral_tip_z = self._ik.forward(q1, q2, q3)
        self._last_ik_q = (q1, q2, q3)

    def get_current_joints(self) -> dict[str, float]:
        return self._read_current_joints()

    def get_neutral_joints(self) -> dict[str, float]:
        return dict(self._neutral_joints)

    def get_neutral_tip_pose(self) -> tuple[float, float]:
        return (self._neutral_tip_x, self._neutral_tip_z)

    def tip_pose_from_joints(self, joints: dict[str, float]) -> tuple[float, float] | None:
        req = ("shoulder_lift", "elbow_flex", "wrist_flex")
        if not all(k in joints for k in req):
            return None
        q1 = math.radians(self._to_kin_deg("shoulder_lift", float(joints["shoulder_lift"])))
        q2 = math.radians(self._to_kin_deg("elbow_flex", float(joints["elbow_flex"])))
        q3 = math.radians(self._to_kin_deg("wrist_flex", float(joints["wrist_flex"])))
        return self._ik.forward(q1, q2, q3)

    def get_current_tip_pose(self) -> tuple[float, float] | None:
        joints = self._read_current_joints()
        if not joints:
            return None
        return self.tip_pose_from_joints(joints)

    def current_kinematic_angles_rad(self) -> tuple[float, float, float] | None:
        joints = self._read_current_joints()
        req = ("shoulder_lift", "elbow_flex", "wrist_flex")
        if not joints or not all(k in joints for k in req):
            return None
        q1 = math.radians(self._to_kin_deg("shoulder_lift", float(joints["shoulder_lift"])))
        q2 = math.radians(self._to_kin_deg("elbow_flex", float(joints["elbow_flex"])))
        q3 = math.radians(self._to_kin_deg("wrist_flex", float(joints["wrist_flex"])))
        return (q1, q2, q3)

    def neutral_kinematic_angles_rad(self) -> tuple[float, float, float]:
        q1 = math.radians(self._to_kin_deg("shoulder_lift", float(self._neutral_joints["shoulder_lift"])))
        q2 = math.radians(self._to_kin_deg("elbow_flex", float(self._neutral_joints["elbow_flex"])))
        q3 = math.radians(self._to_kin_deg("wrist_flex", float(self._neutral_joints["wrist_flex"])))
        return (q1, q2, q3)

    def _solve_horizontal_ik_backoff(
        self,
        requested_dx: float,
        requested_dz: float,
    ) -> tuple[dict[str, float], float, float, float]:
        if not all(k in self._neutral_joints for k in ("shoulder_lift", "elbow_flex", "wrist_flex")):
            raise RuntimeError("Neutral joints missing shoulder/elbow/wrist for IK.")

        current = self._read_current_joints()
        if current and all(k in current for k in ("shoulder_lift", "elbow_flex", "wrist_flex")):
            seed_q2 = math.radians(self._to_kin_deg("elbow_flex", current["elbow_flex"]))
        else:
            seed_q2 = self._last_ik_q[1]

        best_sol = None
        best_scale = 0.0
        best_achieved_dx = 0.0
        best_achieved_dz = 0.0
        for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
            desired_x = self._neutral_tip_x + scale * requested_dx
            desired_z = self._neutral_tip_z + scale * requested_dz
            candidates = self._ik.solve_candidates(
                x_tip=desired_x,
                z_tip=desired_z,
                phi=self._ik_phi_ref,
            )
            if not candidates:
                continue

            # Objective: keep height near requested dz first, then maximize forward progress.
            # Continuity is tertiary to avoid abrupt branch flips.
            target_dx = scale * requested_dx
            target_dz = scale * requested_dz
            chosen = None
            chosen_cost = float("inf")
            chosen_dx = 0.0
            chosen_dz = 0.0
            for cand in candidates:
                ach_x, ach_z = self._ik.forward(cand.q1, cand.q2, cand.q3)
                ach_dx = ach_x - self._neutral_tip_x
                ach_dz = ach_z - self._neutral_tip_z
                height_err = abs(ach_dz - target_dz)
                forward_err = abs(ach_dx - target_dx)
                continuity = abs(cand.q2 - seed_q2)
                cost = 8.0 * height_err + 2.0 * forward_err + 0.2 * continuity
                if cost < chosen_cost:
                    chosen_cost = cost
                    chosen = cand
                    chosen_dx = ach_dx
                    chosen_dz = ach_dz

            if chosen is not None:
                best_sol = chosen
                best_scale = scale
                best_achieved_dx = chosen_dx
                best_achieved_dz = chosen_dz
                break

        if best_sol is None:
            return (
                {
                    "shoulder_lift": self._neutral_joints["shoulder_lift"],
                    "elbow_flex": self._neutral_joints["elbow_flex"],
                    "wrist_flex": self._neutral_joints["wrist_flex"],
                },
                0.0,
                0.0,
                0.0,
            )

        self._last_ik_q = (best_sol.q1, best_sol.q2, best_sol.q3)
        return (
            {
                "shoulder_lift": self._to_motor_deg("shoulder_lift", math.degrees(best_sol.q1)),
                "elbow_flex": self._to_motor_deg("elbow_flex", math.degrees(best_sol.q2)),
                "wrist_flex": self._to_motor_deg("wrist_flex", math.degrees(best_sol.q3)),
            },
            best_achieved_dx,
            best_achieved_dz,
            best_scale,
        )

    def start_cartesian_offset_from_neutral(
        self,
        requested_dx: float,
        requested_dz: float,
        now: float,
    ) -> tuple[float, float, bool]:
        out = dict(self._neutral_joints)

        if "shoulder_pan" in out:
            out["shoulder_pan"] = self._neutral_joints["shoulder_pan"]

        joint_targets, achieved_dx, achieved_dz, scale = self._solve_horizontal_ik_backoff(
            requested_dx=requested_dx,
            requested_dz=requested_dz,
        )
        out.update(joint_targets)
        self._send_joint_target(out, now)
        saturated = scale < 0.999
        return achieved_dx, achieved_dz, saturated

    def goto_neutral(self, now: float) -> None:
        if not self._neutral_joints:
            self._neutral_joints = self._read_current_joints()
        self._send_joint_target(self._neutral_joints, now)

    def start_approach(self, target: Pose3D, now: float) -> None:
        joint_target = self._target_from_pose(target)
        self._send_joint_target(joint_target, now)

    def retreat_to_neutral(self, now: float) -> None:
        self._send_joint_target(self._neutral_joints, now)

    def stop(self, now: float) -> None:
        current = self._read_current_joints()
        if current:
            self._send_joint_target(current, now)
            self._motion = _MotionPlan(target=current, done_at=now)

    def tick(self, now: float) -> None:
        if self.robot is None or self._motion is None:
            return

        current = self._read_current_joints()
        if not current:
            return

        complete = True
        for name, target in self._motion.target.items():
            if name not in current:
                continue
            if abs(current[name] - target) > self._joint_tolerance_deg:
                complete = False
                break

        if complete:
            self._motion = None
            return

        # Fallback: don't block state machine forever if exact convergence is not reached.
        if now >= self._motion.done_at:
            self._motion = None
            return

        if now >= self._next_command_time:
            # Keep driving toward same target without extending motion deadline.
            self._issue_action(self._motion.target)
            self._next_command_time = now + self._command_period_s

    def is_motion_complete(self) -> bool:
        if self._motion is None:
            return True
        return time.monotonic() >= self._motion.done_at
