from __future__ import annotations

import math

from feeding.arm.base import ArmBackend
from feeding.config import SafetyConfig
from feeding.types import Pose3D


class FakeArmBackend(ArmBackend):
    """Simple kinematic simulator so control logic can be tested without hardware."""

    def __init__(self, safety_cfg: SafetyConfig) -> None:
        nx, ny, nz = safety_cfg.neutral_pose_xyz
        self.neutral_pose = Pose3D(nx, ny, nz)
        self.max_speed = safety_cfg.max_linear_speed_mps

        self.current_pose = Pose3D(nx, ny, nz)
        self._start_pose = self.current_pose
        self._target_pose = self.current_pose
        self._motion_start = 0.0
        self._motion_end = 0.0
        self._in_motion = False

    def start(self) -> None:
        return

    def shutdown(self) -> None:
        return

    def _distance(self, a: Pose3D, b: Pose3D) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    def _schedule_motion(self, target: Pose3D, now: float) -> None:
        self.tick(now)
        self._start_pose = self.current_pose
        self._target_pose = target
        distance = self._distance(self._start_pose, target)
        duration = max(distance / max(self.max_speed, 1e-3), 0.15)
        self._motion_start = now
        self._motion_end = now + duration
        self._in_motion = True

    def goto_neutral(self, now: float) -> None:
        self._schedule_motion(self.neutral_pose, now)

    def start_approach(self, target: Pose3D, now: float) -> None:
        self._schedule_motion(target, now)

    def retreat_to_neutral(self, now: float) -> None:
        self._schedule_motion(self.neutral_pose, now)

    def stop(self, now: float) -> None:
        self.tick(now)
        self._in_motion = False

    def tick(self, now: float) -> None:
        if not self._in_motion:
            return

        if now >= self._motion_end:
            self.current_pose = self._target_pose
            self._in_motion = False
            return

        alpha = (now - self._motion_start) / max(self._motion_end - self._motion_start, 1e-6)
        self.current_pose = Pose3D(
            x=self._start_pose.x + alpha * (self._target_pose.x - self._start_pose.x),
            y=self._start_pose.y + alpha * (self._target_pose.y - self._start_pose.y),
            z=self._start_pose.z + alpha * (self._target_pose.z - self._start_pose.z),
        )

    def is_motion_complete(self) -> bool:
        return not self._in_motion
