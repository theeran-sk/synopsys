"""Maps mouth detections in image space to a coarse Cartesian target pose."""

from __future__ import annotations

import logging

from feeding.config import PlanningConfig, SafetyConfig
from feeding.types import MouthObservation, Pose3D

log = logging.getLogger(__name__)


class MouthTargetPlanner:
    """Maps a mouth pixel center to a simple 3D target pose for v1."""

    _EMA_ALPHA = 0.05  # smoothing factor for eye_px

    def __init__(self, planning_cfg: PlanningConfig, safety_cfg: SafetyConfig) -> None:
        self.p = planning_cfg
        nx, ny, nz = safety_cfg.neutral_pose_xyz
        self.neutral_pose = Pose3D(nx, ny, nz)
        self._ema_eye_px: float | None = None

    def reset(self) -> None:
        """Clear smoothing state for a new approach cycle."""
        self._ema_eye_px = None

    def plan_target(self, obs: MouthObservation) -> Pose3D:
        if obs.center_xy is None or obs.frame_size is None:
            return self.neutral_pose

        cx, cy = obs.center_xy
        width, height = obs.frame_size

        # Normalize to [-1, 1] in image space.
        nx = (cx / max(width, 1) - 0.5) * 2.0
        ny = (cy / max(height, 1) - 0.5) * 2.0

        target_y = -nx * self.p.y_scale_m
        target_z = self.neutral_pose.z + self.p.z_target_offset_m - ny * self.p.z_scale_m

        if obs.eye_px is not None and obs.eye_px > self.p.min_eye_px:
            if self._ema_eye_px is None:
                self._ema_eye_px = obs.eye_px
            else:
                self._ema_eye_px += self._EMA_ALPHA * (obs.eye_px - self._ema_eye_px)
            x_target = self.p.forward_depth_scale / self._ema_eye_px
            log.info("eye_px=%.1f  ema=%.1f  x_target=%.3fm (depth-adaptive)", obs.eye_px, self._ema_eye_px, x_target)
        else:
            x_target = self.p.mouth_target_x_m
            log.info("eye_px=%s  x_target=%.3fm (fallback)", obs.eye_px, x_target)

        # Compensate for arm droop at extended reach
        dx_forward = x_target - self.neutral_pose.x
        if dx_forward > 0:
            target_z += dx_forward * self.p.z_droop_compensation

        return Pose3D(
            x=x_target,
            y=target_y,
            z=target_z,
        )
