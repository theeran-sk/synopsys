from __future__ import annotations

from feeding.config import SafetyConfig
from feeding.types import Pose3D


class SafetyFilter:
    def __init__(self, cfg: SafetyConfig) -> None:
        self.cfg = cfg

    def clamp_pose(self, pose: Pose3D) -> Pose3D:
        min_x, min_y, min_z = self.cfg.workspace_min_xyz
        max_x, max_y, max_z = self.cfg.workspace_max_xyz
        return Pose3D(
            x=min(max(pose.x, min_x), max_x),
            y=min(max(pose.y, min_y), max_y),
            z=min(max(pose.z, min_z), max_z),
        )
