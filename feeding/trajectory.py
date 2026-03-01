"""Minimum-jerk (quintic polynomial) trajectory interpolation in Cartesian space."""

from __future__ import annotations

import math


class QuinticSegment:
    """Single-axis minimum-jerk polynomial: position and velocity over [0, T].

    Boundary conditions: x(0) = x0, x(T) = xf, v(0) = v0, v(T) = 0, a(0) = a(T) = 0.
    Normalised form: x(s) = c0 + c1·s + c3·s³ + c4·s⁴ + c5·s⁵  where s = t/T.
    """

    __slots__ = ("c0", "c1", "c3", "c4", "c5", "T")

    def __init__(self, x0: float, xf: float, v0: float, T: float) -> None:
        self.T = max(T, 1e-9)
        dx = xf - x0
        v0T = v0 * self.T
        self.c0 = x0
        self.c1 = v0T
        self.c3 = 10.0 * dx - 6.0 * v0T
        self.c4 = -15.0 * dx + 8.0 * v0T
        self.c5 = 6.0 * dx - 3.0 * v0T

    def sample(self, t: float) -> tuple[float, float]:
        """Return (position, velocity) at time *t* (clamped to [0, T])."""
        t = min(max(t, 0.0), self.T)
        s = t / self.T
        s2 = s * s
        s3 = s2 * s
        s4 = s3 * s
        s5 = s4 * s
        pos = self.c0 + self.c1 * s + self.c3 * s3 + self.c4 * s4 + self.c5 * s5
        # velocity = dx/dt = (1/T) * dx/ds
        ds = self.c1 + 3.0 * self.c3 * s2 + 4.0 * self.c4 * s3 + 5.0 * self.c5 * s4
        vel = ds / self.T
        return pos, vel


class CartesianTrajectory:
    """Three synchronised QuinticSegments (x, y, z) sharing the same duration.

    Peak speed is bounded by *v_max* through duration sizing:
        T = max(min_duration, 15 / (8 · v_max) · distance)
    The factor 15/8 comes from the peak-speed of a zero-initial-velocity quintic.
    """

    def __init__(
        self,
        start_xyz: tuple[float, float, float],
        end_xyz: tuple[float, float, float],
        t_start: float,
        v_max: float,
        v0_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        min_duration: float = 0.4,
    ) -> None:
        dx = end_xyz[0] - start_xyz[0]
        dy = end_xyz[1] - start_xyz[1]
        dz = end_xyz[2] - start_xyz[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        v_max = max(v_max, 1e-6)
        T = max(min_duration, (15.0 / (8.0 * v_max)) * distance)

        self._t_start = t_start
        self._T = T
        self._seg_x = QuinticSegment(start_xyz[0], end_xyz[0], v0_xyz[0], T)
        self._seg_y = QuinticSegment(start_xyz[1], end_xyz[1], v0_xyz[1], T)
        self._seg_z = QuinticSegment(start_xyz[2], end_xyz[2], v0_xyz[2], T)

    @property
    def duration(self) -> float:
        return self._T

    @property
    def t_end(self) -> float:
        return self._t_start + self._T

    def sample(self, now: float) -> tuple[tuple[float, float, float], tuple[float, float, float], bool]:
        """Return ((x,y,z), (vx,vy,vz), is_done) at wall-clock time *now*."""
        t = now - self._t_start
        done = t >= self._T
        px, vx = self._seg_x.sample(t)
        py, vy = self._seg_y.sample(t)
        pz, vz = self._seg_z.sample(t)
        return (px, py, pz), (vx, vy, vz), done

    def is_complete(self, now: float) -> bool:
        return (now - self._t_start) >= self._T

    @classmethod
    def from_retarget(
        cls,
        active: CartesianTrajectory,
        now: float,
        new_end: tuple[float, float, float],
        v_max: float,
        min_duration: float = 0.4,
    ) -> CartesianTrajectory:
        """Create a new trajectory that smoothly continues from *active*'s state at *now*."""
        pos, vel, _ = active.sample(now)
        return cls(
            start_xyz=pos,
            end_xyz=new_end,
            t_start=now,
            v_max=v_max,
            v0_xyz=vel,
            min_duration=min_duration,
        )
