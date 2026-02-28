"""Simple planar kinematics helpers used by the LeRobot arm backend."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class IKSolution:
    q1: float
    q2: float
    q3: float


class Planar3LinkHorizontalIK:
    """3-link planar IK with fixed end-effector pitch."""

    def __init__(self, l1_m: float, l2_m: float, l3_forward_m: float, l3_up_m: float) -> None:
        self.l1 = l1_m
        self.l2 = l2_m
        self.l3f = l3_forward_m
        self.l3u = l3_up_m

    @staticmethod
    def _rot2(theta: float, x: float, z: float) -> tuple[float, float]:
        c = math.cos(theta)
        s = math.sin(theta)
        return (c * x - s * z, s * x + c * z)

    def forward(self, q1: float, q2: float, q3: float) -> tuple[float, float]:
        q12 = q1 + q2
        q123 = q12 + q3
        x1 = self.l1 * math.cos(q1)
        z1 = self.l1 * math.sin(q1)
        x2 = x1 + self.l2 * math.cos(q12)
        z2 = z1 + self.l2 * math.sin(q12)
        ox, oz = self._rot2(q123, self.l3f, self.l3u)
        return (x2 + ox, z2 + oz)

    def solve_candidates(self, x_tip: float, z_tip: float, phi: float) -> list[IKSolution]:
        ox, oz = self._rot2(phi, self.l3f, self.l3u)
        xw = x_tip - ox
        zw = z_tip - oz

        d2 = xw * xw + zw * zw
        denom = 2.0 * self.l1 * self.l2
        if denom <= 1e-9:
            return None

        cos_q2 = (d2 - self.l1 * self.l1 - self.l2 * self.l2) / denom
        if cos_q2 < -1.0001 or cos_q2 > 1.0001:
            return []
        cos_q2 = min(1.0, max(-1.0, cos_q2))

        q2_candidates = [math.acos(cos_q2), -math.acos(cos_q2)]
        out: list[IKSolution] = []

        for q2 in q2_candidates:
            k1 = self.l1 + self.l2 * math.cos(q2)
            k2 = self.l2 * math.sin(q2)
            q1 = math.atan2(zw, xw) - math.atan2(k2, k1)
            q3 = phi - q1 - q2
            out.append(IKSolution(q1=q1, q2=q2, q3=q3))

        return out

    def solve(self, x_tip: float, z_tip: float, phi: float, seed_q2: float = 0.0) -> IKSolution | None:
        candidates = self.solve_candidates(x_tip=x_tip, z_tip=z_tip, phi=phi)
        if not candidates:
            return None
        best: IKSolution | None = None
        best_cost = float("inf")
        for cand in candidates:
            cost = abs(cand.q2 - seed_q2)
            if cost < best_cost:
                best_cost = cost
                best = cand

        return best
