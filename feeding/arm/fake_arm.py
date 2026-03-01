"""Fake arm backend that simulates motion for logic testing without hardware."""

from __future__ import annotations

import math

from feeding.arm.base import ArmBackend
from feeding.config import SafetyConfig
from feeding.trajectory import CartesianTrajectory
from feeding.types import Pose3D


class FakeArmBackend(ArmBackend):
    """Simple kinematic simulator so control logic can be tested without hardware."""

    def __init__(self, safety_cfg: SafetyConfig) -> None:
        nx, ny, nz = safety_cfg.neutral_pose_xyz
        self.neutral_pose = Pose3D(nx, ny, nz)
        self.max_speed = safety_cfg.max_linear_speed_mps
        self._min_duration = safety_cfg.min_trajectory_duration_s

        self.current_pose = Pose3D(nx, ny, nz)
        self._traj: CartesianTrajectory | None = None

    def start(self) -> None:
        return

    def shutdown(self) -> None:
        return

    def _pose_to_tuple(self, p: Pose3D) -> tuple[float, float, float]:
        return (p.x, p.y, p.z)

    def _schedule_motion(self, target: Pose3D, now: float) -> None:
        self.tick(now)
        start = self._pose_to_tuple(self.current_pose)
        end = self._pose_to_tuple(target)
        if self._traj is not None and not self._traj.is_complete(now):
            self._traj = CartesianTrajectory.from_retarget(
                self._traj, now, end, self.max_speed, self._min_duration,
            )
        else:
            self._traj = CartesianTrajectory(
                start, end, now, self.max_speed, min_duration=self._min_duration,
            )

    def goto_neutral(self, now: float) -> None:
        self._schedule_motion(self.neutral_pose, now)

    def start_approach(self, target: Pose3D, now: float) -> None:
        self._schedule_motion(target, now)

    def update_approach_target(self, target: Pose3D, now: float) -> None:
        if self._traj is not None and not self._traj.is_complete(now):
            end = self._pose_to_tuple(target)
            self._traj = CartesianTrajectory.from_retarget(
                self._traj, now, end, self.max_speed, self._min_duration,
            )
        else:
            self.start_approach(target, now)

    def retreat_to_neutral(self, now: float) -> None:
        self._schedule_motion(self.neutral_pose, now)

    def stop(self, now: float) -> None:
        self.tick(now)
        self._traj = None

    def tick(self, now: float) -> None:
        if self._traj is None:
            return
        pos, _vel, done = self._traj.sample(now)
        self.current_pose = Pose3D(pos[0], pos[1], pos[2])
        if done:
            self._traj = None

    def is_motion_complete(self) -> bool:
        return self._traj is None
