from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from feeding.arm.lerobot_arm import LeRobotArmBackend
from feeding.config import load_config
from feeding.kinematics import IKSolution, Planar3LinkHorizontalIK


def _rot(theta: float, x: float, z: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return (c * x - s * z, s * x + c * z)


def _arm_points(q1: float, q2: float, q3: float, l1: float, l2: float, l3f: float, l3u: float):
    q12 = q1 + q2
    q123 = q12 + q3
    p0 = (0.0, 0.0)
    p1 = (l1 * math.cos(q1), l1 * math.sin(q1))
    p2 = (p1[0] + l2 * math.cos(q12), p1[1] + l2 * math.sin(q12))
    ox, oz = _rot(q123, l3f, l3u)
    p3 = (p2[0] + ox, p2[1] + oz)
    return p0, p1, p2, p3


@dataclass
class SessionLogger:
    writer: csv.DictWriter
    fp: object

    @classmethod
    def create(cls, path: Path) -> "SessionLogger":
        path.parent.mkdir(parents=True, exist_ok=True)
        fp = path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "timestamp",
                "mode",
                "target_x",
                "target_z",
                "actual_x",
                "actual_z",
                "error_x",
                "error_z",
                "saturated",
            ],
        )
        writer.writeheader()
        return cls(writer=writer, fp=fp)

    def log(
        self,
        mode: str,
        target_x: float,
        target_z: float,
        actual_x: float,
        actual_z: float,
        saturated: bool,
    ) -> None:
        self.writer.writerow(
            {
                "timestamp": time.time(),
                "mode": mode,
                "target_x": target_x,
                "target_z": target_z,
                "actual_x": actual_x,
                "actual_z": actual_z,
                "error_x": target_x - actual_x,
                "error_z": target_z - actual_z,
                "saturated": int(bool(saturated)),
            }
        )

    def close(self) -> None:
        self.fp.close()


@dataclass
class EmpiricalVisualMap:
    back_coef: np.ndarray
    front_coef: np.ndarray
    spoon_coef: np.ndarray

    @classmethod
    def from_csv(cls, path: Path) -> "EmpiricalVisualMap | None":
        if not path.exists():
            return None
        rows = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    rows.append(
                        (
                            float(r["shoulder_lift"]),
                            float(r["elbow_flex"]),
                            float(r["wrist_flex"]),
                            float(r["backarm_deg"]),
                            float(r["frontarm_deg"]),
                            float(r["spoon_deg"]),
                        )
                    )
                except Exception:
                    continue
        if len(rows) < 4:
            return None

        arr = np.array(rows, dtype=float)
        sh = arr[:, 0]
        el = arr[:, 1]
        wr = arr[:, 2]
        y_back = arr[:, 3]
        y_front = arr[:, 4]
        y_spoon = arr[:, 5]

        xb = np.column_stack([np.ones_like(sh), sh])
        xf = np.column_stack([np.ones_like(sh), sh, el])
        xs = np.column_stack([np.ones_like(sh), sh, el, wr])

        cb, *_ = np.linalg.lstsq(xb, y_back, rcond=None)
        cf, *_ = np.linalg.lstsq(xf, y_front, rcond=None)
        cs, *_ = np.linalg.lstsq(xs, y_spoon, rcond=None)
        return cls(back_coef=cb, front_coef=cf, spoon_coef=cs)

    def predict_abs_deg(self, shoulder: float, elbow: float, wrist: float) -> tuple[float, float, float]:
        back = float(self.back_coef[0] + self.back_coef[1] * shoulder)
        front = float(self.front_coef[0] + self.front_coef[1] * shoulder + self.front_coef[2] * elbow)
        spoon = float(
            self.spoon_coef[0]
            + self.spoon_coef[1] * shoulder
            + self.spoon_coef[2] * elbow
            + self.spoon_coef[3] * wrist
        )
        return back, front, spoon


class SimState:
    def __init__(self, cfg_path: str, mode: str, log_path: Path, cmd_period_s: float):
        self.cfg = load_config(cfg_path)
        self.mode = mode
        self.cmd_period_s = max(cmd_period_s, 0.05)
        self.last_cmd_t = 0.0
        self.dragging = False
        self.last_saturated = False
        self.command_enabled = False
        self._startup_locked = False
        self.x_display_sign = -1.0  # show forward direction to the left on screen
        self.drag_joint = ""
        self.safe_min_z = max(float(self.cfg.safety.workspace_min_xyz[2]), -0.05)
        self.safe_max_z = min(float(self.cfg.safety.workspace_max_xyz[2]), 0.30)
        self.max_target_step_m = 0.012
        self.target_smooth_alpha = 0.25
        self.height_lock_on_drag = True
        self.height_lock_deadband_m = 0.02

        self.l1 = float(self.cfg.lerobot.link_l1_m)
        self.l2 = float(self.cfg.lerobot.link_l2_m)
        self.l3f = float(self.cfg.lerobot.spoon_offset_forward_m)
        self.l3u = float(self.cfg.lerobot.spoon_offset_up_m)
        self.ik = Planar3LinkHorizontalIK(self.l1, self.l2, self.l3f, self.l3u)
        # Disabled by default: current empirical fit can collapse rendering.
        # Re-enable after explicit visual-calibration pass.
        self.visual_map = None

        self.arm: LeRobotArmBackend | None = None
        self.visual_cfg_path = Path("configs/visual_neutral.yaml")
        self.vis_back0_deg = 58.0
        self.vis_front0_deg = 124.0
        self.vis_spoon0_deg = 180.0
        self._load_visual_baseline()

        if mode in ("live_view", "command_live"):
            self.cfg.backend = "lerobot"
            self.cfg.lerobot.use_horizontal_ik = True
            # Keep simulator exits gentle; don't auto-compact/fold hardware.
            self.cfg.lerobot.compact_on_shutdown = False
            self.arm = LeRobotArmBackend(self.cfg.safety, self.cfg.lerobot)
            self.arm.start()
            self._startup_locked = True
            self.command_enabled = False
            self.arm.goto_neutral(time.monotonic())
            t0 = time.monotonic()
            while time.monotonic() - t0 < 4.0:
                self.arm.tick(time.monotonic())
                if self.arm.is_motion_complete():
                    break
                time.sleep(0.03)
            neutral_tip = self.arm.get_neutral_tip_pose()
            kin = self.arm.neutral_kinematic_angles_rad()
            self.q1, self.q2, self.q3 = kin
        else:
            nj = self.cfg.lerobot.neutral_joints_deg
            sign = self.cfg.lerobot.ik_motor_sign
            off = self.cfg.lerobot.ik_motor_offset_deg
            self.q1 = math.radians(float(sign["shoulder_lift"]) * (float(nj["shoulder_lift"]) - float(off["shoulder_lift"])))
            self.q2 = math.radians(float(sign["elbow_flex"]) * (float(nj["elbow_flex"]) - float(off["elbow_flex"])))
            self.q3 = math.radians(float(sign["wrist_flex"]) * (float(nj["wrist_flex"]) - float(off["wrist_flex"])))
            neutral_tip = self.ik.forward(self.q1, self.q2, self.q3)
            if mode == "visual_calib":
                self._load_visual_neutral_if_exists()

        self.phi_ref = self.q1 + self.q2 + self.q3
        self.neutral_x, self.neutral_z = neutral_tip
        self.target_x, self.target_z = neutral_tip
        self.command_x, self.command_z = neutral_tip
        # Safety default: do not command hardware until user explicitly drags.
        self.command_enabled = False
        self._startup_locked = False
        self.logger = SessionLogger.create(log_path)
        self._n_sh = float(self.cfg.lerobot.neutral_joints_deg.get("shoulder_lift", 0.0))
        self._n_el = float(self.cfg.lerobot.neutral_joints_deg.get("elbow_flex", 0.0))
        self._n_wr = float(self.cfg.lerobot.neutral_joints_deg.get("wrist_flex", 0.0))

    def _load_visual_neutral_if_exists(self) -> None:
        if not self.visual_cfg_path.exists():
            return
        try:
            with self.visual_cfg_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            q1 = float(data.get("q1_deg", 0.0))
            q2 = float(data.get("q2_deg", 0.0))
            q3 = float(data.get("q3_deg", 0.0))
            self.q1, self.q2, self.q3 = math.radians(q1), math.radians(q2), math.radians(q3)
        except Exception:
            pass

    def _load_visual_baseline(self) -> None:
        if not self.visual_cfg_path.exists():
            return
        try:
            with self.visual_cfg_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self.vis_back0_deg = float(data.get("back_abs_deg", self.vis_back0_deg))
            self.vis_front0_deg = float(data.get("front_abs_deg", self.vis_front0_deg))
            self.vis_spoon0_deg = float(data.get("spoon_abs_deg", self.vis_spoon0_deg))
        except Exception:
            pass

    def save_visual_neutral(self) -> Path:
        out = {
            "q1_deg": round(math.degrees(self.q1), 4),
            "q2_deg": round(math.degrees(self.q2), 4),
            "q3_deg": round(math.degrees(self.q3), 4),
            "back_abs_deg": round(math.degrees(self.q1), 4),
            "front_abs_deg": round(math.degrees(self.q1 + self.q2), 4),
            "spoon_abs_deg": round(math.degrees(self.q1 + self.q2 + self.q3), 4),
        }
        self.visual_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with self.visual_cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, sort_keys=False)
        return self.visual_cfg_path

    def _update_draw_angles_from_joint_obs(self) -> None:
        if self.arm is None:
            return
        joints = self.arm.get_current_joints()
        if not joints:
            return
        if not all(k in joints for k in ("shoulder_lift", "elbow_flex", "wrist_flex")):
            kin = self.arm.current_kinematic_angles_rad()
            if kin is not None:
                self.q1, self.q2, self.q3 = kin
            return

        # Render-only side-view mapping tuned to match your real neutral silhouette:
        # back arm up-right, front arm mirrored above, spoon flat left.
        sh = float(joints["shoulder_lift"])
        el = float(joints["elbow_flex"])
        wr = float(joints["wrist_flex"])
        dsh = sh - self._n_sh
        delv = el - self._n_el
        dwr = wr - self._n_wr

        back_deg = self.vis_back0_deg + 0.95 * dsh
        front_deg = self.vis_front0_deg - 0.60 * dsh + 0.95 * delv
        spoon_deg = self.vis_spoon0_deg + 0.10 * dwr

        back_deg = max(-170.0, min(170.0, back_deg))
        front_deg = max(-170.0, min(260.0, front_deg))
        spoon_deg = max(-170.0, min(260.0, spoon_deg))

        q1 = math.radians(back_deg)
        q2 = math.radians(front_deg - back_deg)
        q3 = math.radians(spoon_deg - front_deg)
        if not all(math.isfinite(v) for v in (q1, q2, q3)):
            kin = self.arm.current_kinematic_angles_rad()
            if kin is not None:
                self.q1, self.q2, self.q3 = kin
            return
        # Guard against absurd render values; fallback to kinematic angles.
        if any(abs(v) > math.radians(540.0) for v in (q1, q2, q3)):
            kin = self.arm.current_kinematic_angles_rad()
            if kin is not None:
                self.q1, self.q2, self.q3 = kin
            return
        self.q1, self.q2, self.q3 = q1, q2, q3

    def world_to_display(self, x: float, z: float) -> tuple[float, float]:
        return (self.x_display_sign * x, z)

    def display_to_world(self, x_disp: float, z_disp: float) -> tuple[float, float]:
        return (self.x_display_sign * x_disp, z_disp)

    def clamp_world_target(self, x: float, z: float) -> tuple[float, float]:
        lo = self.cfg.safety.workspace_min_xyz
        hi = self.cfg.safety.workspace_max_xyz
        x = min(max(x, float(lo[0])), float(hi[0]))
        z = min(max(z, self.safe_min_z), self.safe_max_z)
        return (x, z)

    def _step_toward_command(self, actual_x: float, actual_z: float) -> tuple[float, float]:
        # Smooth and rate-limit command target so mouse clicks can't cause violent jumps.
        tx = (1.0 - self.target_smooth_alpha) * self.command_x + self.target_smooth_alpha * self.target_x
        tz = (1.0 - self.target_smooth_alpha) * self.command_z + self.target_smooth_alpha * self.target_z
        if self.height_lock_on_drag and self.dragging:
            # During normal drag, keep spoon height unless user clearly drags vertically.
            if abs(self.target_z - actual_z) <= self.height_lock_deadband_m:
                tz = actual_z
        dx = tx - actual_x
        dz = tz - actual_z
        dist = math.hypot(dx, dz)
        if dist > self.max_target_step_m:
            s = self.max_target_step_m / max(dist, 1e-9)
            tx = actual_x + s * dx
            tz = actual_z + s * dz
        tx, tz = self.clamp_world_target(tx, tz)
        self.command_x, self.command_z = tx, tz
        return tx, tz

    def _solve_with_backoff(self, req_dx: float, req_dz: float) -> tuple[IKSolution | None, bool]:
        seed_q2 = self.q2
        for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
            desired_x = self.neutral_x + scale * req_dx
            desired_z = self.neutral_z + scale * req_dz
            cands = self.ik.solve_candidates(desired_x, desired_z, self.phi_ref)
            if not cands:
                continue
            best = min(cands, key=lambda c: abs(c.q2 - seed_q2))
            return best, scale < 0.999
        return None, True

    def step(self) -> tuple[tuple[float, float], tuple[float, float], bool]:
        now = time.monotonic()
        saturated = False

        if self.mode == "sim":
            req_dx = self.target_x - self.neutral_x
            req_dz = self.target_z - self.neutral_z
            sol, saturated = self._solve_with_backoff(req_dx, req_dz)
            if sol is not None:
                self.q1, self.q2, self.q3 = sol.q1, sol.q2, sol.q3
            _, _, _, tip = _arm_points(self.q1, self.q2, self.q3, self.l1, self.l2, self.l3f, self.l3u)
            actual = tip

        elif self.mode == "live_view":
            assert self.arm is not None
            self.arm.tick(now)
            tip = self.arm.get_current_tip_pose()
            if tip is None:
                tip = (self.neutral_x, self.neutral_z)
            if self.command_enabled and self.dragging and now - self.last_cmd_t >= self.cmd_period_s:
                cmd_x, cmd_z = self._step_toward_command(tip[0], tip[1])
                req_dx = cmd_x - self.neutral_x
                req_dz = cmd_z - self.neutral_z
                _, _, saturated = self.arm.start_cartesian_offset_from_neutral(req_dx, req_dz, now)
                self.last_cmd_t = now
            self._update_draw_angles_from_joint_obs()
            tip = self.arm.get_current_tip_pose()
            if tip is None:
                _, _, _, tip = _arm_points(self.q1, self.q2, self.q3, self.l1, self.l2, self.l3f, self.l3u)
            actual = tip

        elif self.mode == "command_live":
            assert self.arm is not None
            self.arm.tick(now)
            tip_now = self.arm.get_current_tip_pose()
            if tip_now is None:
                tip_now = (self.neutral_x, self.neutral_z)
            if self.command_enabled and self.dragging and now - self.last_cmd_t >= self.cmd_period_s:
                cmd_x, cmd_z = self._step_toward_command(tip_now[0], tip_now[1])
                req_dx = cmd_x - self.neutral_x
                req_dz = cmd_z - self.neutral_z
                _, _, saturated = self.arm.start_cartesian_offset_from_neutral(req_dx, req_dz, now)
                self.last_cmd_t = now
            self._update_draw_angles_from_joint_obs()
            tip = self.arm.get_current_tip_pose()
            if tip is None:
                _, _, _, tip = _arm_points(self.q1, self.q2, self.q3, self.l1, self.l2, self.l3f, self.l3u)
            actual = tip
        else:
            # visual_calib: no hardware motion, render-only interactive pose shaping
            _, _, _, tip = _arm_points(self.q1, self.q2, self.q3, self.l1, self.l2, self.l3f, self.l3u)
            actual = tip
            self.target_x, self.target_z = tip

        self.last_saturated = saturated
        self.logger.log(
            mode=self.mode,
            target_x=self.target_x,
            target_z=self.target_z,
            actual_x=actual[0],
            actual_z=actual[1],
            saturated=saturated,
        )
        return (self.target_x, self.target_z), actual, saturated

    def close(self) -> None:
        self.logger.close()
        if self.arm is not None:
            self.arm.shutdown()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="2D simulator and live digital twin for feeding arm")
    p.add_argument("--mode", choices=["sim", "live_view", "command_live", "visual_calib"], default="sim")
    p.add_argument("--config", default="configs/feeding_default.yaml")
    p.add_argument("--cmd-period-s", type=float, default=0.2)
    p.add_argument("--log", default="", help="CSV output path (default: logs/sim_<timestamp>.csv)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log) if args.log else Path("logs") / f"sim_{args.mode}_{ts}.csv"
    state = SimState(cfg_path=args.config, mode=args.mode, log_path=log_path, cmd_period_s=args.cmd_period_s)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.1)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("forward-left (m)")
    ax.set_ylabel("z (m)")
    # Build view from configured workspace, then center around neutral for easier dragging.
    wmin = state.cfg.safety.workspace_min_xyz
    wmax = state.cfg.safety.workspace_max_xyz
    xlo_disp = state.world_to_display(float(wmax[0]), 0.0)[0]
    xhi_disp = state.world_to_display(float(wmin[0]), 0.0)[0]
    zlo = float(wmin[2])
    zhi = float(wmax[2])
    mx = 0.16
    mz = 0.16
    # Always include shoulder base at (0,0) so links never disappear off-screen.
    base_x_disp = state.world_to_display(0.0, 0.0)[0]
    view_x_min = min(xlo_disp - mx, xhi_disp - mx, base_x_disp - 0.24)
    view_x_max = max(xlo_disp + mx, xhi_disp + mx, base_x_disp + 0.24)
    view_z_min = min(zlo - mz, 0.0 - 0.24)
    view_z_max = max(zhi + mz, 0.0 + 0.24)
    ax.set_xlim(view_x_min, view_x_max)
    ax.set_ylim(view_z_min, view_z_max)
    ax.grid(True, alpha=0.25)

    arm_line, = ax.plot([], [], "-o", lw=3, color="tab:blue")
    target_dot, = ax.plot([], [], "rx", ms=10, mew=2)
    actual_dot, = ax.plot([], [], "go", ms=7)
    status_text = ax.text(0.02, 0.98, "", va="top", ha="left", transform=ax.transAxes)

    def _set_target_from_event(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        tx, tz = state.display_to_world(float(event.xdata), float(event.ydata))
        tx, tz = state.clamp_world_target(tx, tz)
        state.target_x = tx
        state.target_z = tz

    def _world_points():
        return _arm_points(state.q1, state.q2, state.q3, state.l1, state.l2, state.l3f, state.l3u)

    def on_press(event):
        if event.button == 1:
            state.dragging = True
            if state.mode == "visual_calib":
                if event.xdata is None or event.ydata is None:
                    return
                wx, wz = state.display_to_world(float(event.xdata), float(event.ydata))
                p0, p1, p2, p3 = _world_points()
                cands = [("elbow", p1), ("wrist", p2), ("tip", p3)]
                name, pt = min(cands, key=lambda x: (x[1][0] - wx) ** 2 + (x[1][1] - wz) ** 2)
                d = math.hypot(pt[0] - wx, pt[1] - wz)
                state.drag_joint = name if d <= 0.09 else ""
            else:
                state.command_enabled = True
                _set_target_from_event(event)

    def on_release(event):
        state.dragging = False
        if state.mode == "visual_calib":
            state.drag_joint = ""
        else:
            # Stop active commanding as soon as mouse is released.
            state.command_enabled = False

    def on_move(event):
        if not state.dragging:
            return
        if state.mode != "visual_calib":
            _set_target_from_event(event)
            return
        if event.xdata is None or event.ydata is None or not state.drag_joint:
            return
        wx, wz = state.display_to_world(float(event.xdata), float(event.ydata))
        p0, p1, p2, _ = _world_points()
        if state.drag_joint == "elbow":
            state.q1 = math.atan2(wz - p0[1], wx - p0[0])
        elif state.drag_joint == "wrist":
            abs12 = math.atan2(wz - p1[1], wx - p1[0])
            state.q2 = abs12 - state.q1
        elif state.drag_joint == "tip":
            abs123 = math.atan2(wz - p2[1], wx - p2[0])
            state.q3 = abs123 - (state.q1 + state.q2)

    def on_key(event):
        if event.key == "q":
            plt.close(fig)
        elif event.key == "n":
            state.target_x, state.target_z = state.neutral_x, state.neutral_z
            if state.mode == "visual_calib":
                nj = state.cfg.lerobot.neutral_joints_deg
                sign = state.cfg.lerobot.ik_motor_sign
                off = state.cfg.lerobot.ik_motor_offset_deg
                state.q1 = math.radians(float(sign["shoulder_lift"]) * (float(nj["shoulder_lift"]) - float(off["shoulder_lift"])))
                state.q2 = math.radians(float(sign["elbow_flex"]) * (float(nj["elbow_flex"]) - float(off["elbow_flex"])))
                state.q3 = math.radians(float(sign["wrist_flex"]) * (float(nj["wrist_flex"]) - float(off["wrist_flex"])))
        elif event.key == "c":
            base_x_disp = state.world_to_display(0.0, 0.0)[0]
            cx = 0.5 * (base_x_disp + state.world_to_display(state.neutral_x, state.neutral_z)[0])
            cz = 0.5 * (0.0 + state.neutral_z)
            xspan = max(ax.get_xlim()[1] - ax.get_xlim()[0], 0.6)
            zspan = max(ax.get_ylim()[1] - ax.get_ylim()[0], 0.6)
            ax.set_xlim(cx - 0.5 * xspan, cx + 0.5 * xspan)
            ax.set_ylim(cz - 0.5 * zspan, cz + 0.5 * zspan)
        elif event.key == "g" and state.mode in ("live_view", "command_live"):
            state.command_enabled = not state.command_enabled
        elif event.key == "s" and state.mode == "visual_calib":
            out = state.save_visual_neutral()
            print(f"Saved visual neutral to {out}")

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)

    if args.mode == "command_live":
        print("Note: command_live is kept as alias; live_view now supports drag-to-command.")
    if args.mode == "visual_calib":
        print("Visual calibration mode: drag elbow/wrist/tip markers to shape neutral.")
        print("Keys: s=save visual neutral, n=reset pose, q=quit")
    print(f"Mode: {state.mode}")
    print("Controls: hold left-drag to command, release to stop, n=neutral target, c=recenter view, q=quit, g=toggle command arming")
    print(f"Logging to: {log_path}")

    try:
        while plt.fignum_exists(fig.number):
            target, actual, saturated = state.step()
            p0, p1, p2, p3 = _arm_points(state.q1, state.q2, state.q3, state.l1, state.l2, state.l3f, state.l3u)
            p0d = state.world_to_display(p0[0], p0[1])
            p1d = state.world_to_display(p1[0], p1[1])
            p2d = state.world_to_display(p2[0], p2[1])
            p3d = state.world_to_display(p3[0], p3[1])
            td = state.world_to_display(target[0], target[1])
            # Keep the green "actual" marker visually attached to the rendered spoon tip.
            ad = p3d
            xs = [p0d[0], p1d[0], p2d[0], p3d[0]]
            zs = [p0d[1], p1d[1], p2d[1], p3d[1]]
            if all(math.isfinite(v) for v in xs + zs):
                arm_line.set_data(xs, zs)
            else:
                arm_line.set_data([], [])
            target_dot.set_data([td[0]], [td[1]])
            actual_dot.set_data([ad[0]], [ad[1]])
            status_text.set_text(
                f"mode={state.mode}  sat={int(saturated)}\n"
                f"target=({target[0]:.3f}, {target[1]:.3f})\n"
                f"cmd=({state.command_x:.3f}, {state.command_z:.3f})\n"
                f"actual=({actual[0]:.3f}, {actual[1]:.3f})\n"
                f"err=({target[0]-actual[0]:.3f}, {target[1]-actual[1]:.3f})\n"
                f"drag_joint={state.drag_joint if state.drag_joint else '-'}"
            )
            fig.canvas.draw_idle()
            plt.pause(0.03)
    finally:
        state.close()
        print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
