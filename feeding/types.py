"""Shared data structures for mouth observations and Cartesian target poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Pose3D:
    x: float
    y: float
    z: float


@dataclass
class MouthObservation:
    timestamp: float
    detected: bool
    is_open: bool
    openness_ratio: float
    eye_px: Optional[float]
    mouth_px: Optional[float]
    center_xy: Optional[Tuple[int, int]]
    frame_size: Optional[Tuple[int, int]]
