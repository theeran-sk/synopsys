from __future__ import annotations

import argparse
import time

from feeding.arm.lerobot_arm import LeRobotArmBackend
from feeding.config import load_config


def wait_motion(arm: LeRobotArmBackend, timeout_s: float = 2.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        arm.tick(time.monotonic())
        if arm.is_motion_complete():
            return
        time.sleep(0.03)


def main() -> None:
    p = argparse.ArgumentParser(description="Move one joint at a time from neutral to verify mapping")
    p.add_argument("--config", type=str, default="configs/feeding_default.yaml")
    p.add_argument("--delta-deg", type=float, default=10.0)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.backend = "lerobot"
    arm = LeRobotArmBackend(cfg.safety, cfg.lerobot)
    arm.start()
    try:
        now = time.monotonic()
        arm.goto_neutral(now)
        wait_motion(arm, timeout_s=2.5)
        neutral = arm.get_neutral_joints()
        print("At neutral:", neutral)
        print("For each step: watch which link moves, then press ENTER.")

        for joint in ("shoulder_lift", "elbow_flex", "wrist_flex"):
            input(f"\nPress ENTER to test {joint} +{args.delta_deg} deg...")
            t = dict(neutral)
            t[joint] = t[joint] + args.delta_deg
            arm._send_joint_target(t, time.monotonic())
            wait_motion(arm, timeout_s=2.5)
            print("Now at:", arm.get_current_joints())

            input(f"Press ENTER to test {joint} -{args.delta_deg} deg from neutral...")
            t = dict(neutral)
            t[joint] = t[joint] - args.delta_deg
            arm._send_joint_target(t, time.monotonic())
            wait_motion(arm, timeout_s=2.5)
            print("Now at:", arm.get_current_joints())

            input("Press ENTER to return neutral...")
            arm.goto_neutral(time.monotonic())
            wait_motion(arm, timeout_s=2.5)
            print("Back to neutral.")
    finally:
        arm.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
