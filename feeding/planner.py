"""Maps mouth detections in image space to a coarse Cartesian target pose."""

from __future__ import annotations

from feeding.config import PlanningConfig, SafetyConfig
from feeding.types import MouthObservation, Pose3D


class MouthTargetPlanner:
    """Maps a mouth pixel center to a simple 3D target pose for v1."""

    def __init__(self, planning_cfg: PlanningConfig, safety_cfg: SafetyConfig) -> None:
        self.p = planning_cfg
        nx, ny, nz = safety_cfg.neutral_pose_xyz
        self.neutral_pose = Pose3D(nx, ny, nz)

    def plan_target(self, obs: MouthObservation) -> Pose3D:
        if obs.center_xy is None or obs.frame_size is None:
            return self.neutral_pose

        cx, cy = obs.center_xy
        width, height = obs.frame_size

        # Normalize to [-1, 1] in image space.
        nx = (cx / max(width, 1) - 0.5) * 2.0
        ny = (cy / max(height, 1) - 0.5) * 2.0

        target_y = -nx * self.p.y_scale_m
        target_z = self.neutral_pose.z - ny * self.p.z_scale_m

        return Pose3D(
            x=self.p.mouth_target_x_m,
            y=target_y,
            z=target_z,
        )
