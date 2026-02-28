from __future__ import annotations

import argparse
import logging
import time

from feeding.arm.lerobot_arm import LeRobotArmBackend
from feeding.config import load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="IK forward-motion test from neutral pose")
    p.add_argument("--config", type=str, default="configs/feeding_default.yaml")
    p.add_argument("--dx-cm", type=float, default=60.0, help="Requested forward distance from neutral (cm)")
    p.add_argument("--dz-cm", type=float, default=0.0, help="Requested vertical delta from neutral (cm)")
    p.add_argument("--step-cm", type=float, default=2.0, help="Step size for trajectory (cm)")
    p.add_argument("--settle-s", type=float, default=2.0, help="Max wait per step (s)")
    p.add_argument("--pause-s", type=float, default=0.15, help="Pause between steps (s)")
    p.add_argument(
        "--stop-on-saturation",
        action="store_true",
        help="Stop when request exceeds reachable workspace",
    )
    return p


def _wait_motion(arm: LeRobotArmBackend, timeout_s: float) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        now = time.monotonic()
        arm.tick(now)
        if arm.is_motion_complete():
            return
        time.sleep(0.03)


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    cfg.backend = "lerobot"
    cfg.lerobot.use_horizontal_ik = True

    arm = LeRobotArmBackend(cfg.safety, cfg.lerobot)

    dx_m = max(0.0, args.dx_cm) / 100.0
    dz_m = args.dz_cm / 100.0
    step_m = max(args.step_cm, 0.5) / 100.0

    print("Starting IK forward test.")
    print(f"Requested final delta: dx={dx_m:.3f} m, dz={dz_m:.3f} m")
    print("Note: if unreachable, solver will saturate at max reachable point.")

    arm.start()
    try:
        now = time.monotonic()
        arm.goto_neutral(now)
        _wait_motion(arm, timeout_s=max(args.settle_s, 0.4))

        neutral_tip_x, neutral_tip_z = arm.get_neutral_tip_pose()
        print(f"Neutral tip pose (planar): x={neutral_tip_x:.3f} m, z={neutral_tip_z:.3f} m")
        print(f"Neutral joints: {arm.get_neutral_joints()}")

        requested = 0.0
        step_idx = 0
        while requested < dx_m - 1e-9:
            requested = min(dx_m, requested + step_m)
            step_idx += 1
            now = time.monotonic()
            achieved_dx, achieved_dz, saturated = arm.start_cartesian_offset_from_neutral(
                requested_dx=requested,
                requested_dz=dz_m,
                now=now,
            )
            _wait_motion(arm, timeout_s=max(args.settle_s, 0.4))
            joints = arm.get_current_joints()
            sat_text = "YES" if saturated else "no"
            print(
                f"[{step_idx:02d}] req_dx={requested*100:.1f}cm -> "
                f"ach_dx={achieved_dx*100:.1f}cm, ach_dz={achieved_dz*100:.1f}cm, "
                f"saturated={sat_text}, joints={joints}"
            )
            if saturated and args.stop_on_saturation:
                print("Stopping early due to workspace saturation.")
                break
            time.sleep(max(args.pause_s, 0.0))
    finally:
        arm.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
