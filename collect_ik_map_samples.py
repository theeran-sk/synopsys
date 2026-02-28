from __future__ import annotations

import argparse
import csv
import time

from feeding.arm.lerobot_arm import LeRobotArmBackend
from feeding.config import load_config


def _set_torque(arm: LeRobotArmBackend, enabled: bool) -> None:
    if arm.robot is None:
        return
    for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"):
        try:
            if enabled:
                arm.robot.bus.enable_torque(m, num_retry=1)
            else:
                arm.robot.bus.disable_torque(m, num_retry=1)
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="Collect IK motor-map samples")
    p.add_argument("--config", type=str, default="configs/feeding_default.yaml")
    p.add_argument("--out", type=str, default="configs/ik_map_samples.csv")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.backend = "lerobot"
    arm = LeRobotArmBackend(cfg.safety, cfg.lerobot)
    arm.start()
    arm.goto_neutral(time.monotonic())
    time.sleep(1.0)
    torque_on = True

    print("Collecting samples.")
    print("Commands:")
    print("  r = toggle torque (release/stiffen)")
    print("  s = sample current pose")
    print("  q = finish")
    print("For each sample: enter measured angles in degrees.")
    print("Angles convention: trig style from +X, CCW positive.")
    print("Columns: backarm_deg, frontarm_deg, spoon_deg, shoulder_lift, elbow_flex, wrist_flex")

    rows: list[dict[str, float]] = []
    try:
        while True:
            cmd = input("[r/s/q] > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "r":
                torque_on = not torque_on
                _set_torque(arm, enabled=torque_on)
                print("Torque ON" if torque_on else "Torque OFF")
                continue
            if cmd != "s":
                print("Use r, s, or q.")
                continue

            back = input("backarm_deg: ").strip()
            front = input("frontarm_deg: ").strip()
            spoon = input("spoon_deg: ").strip()
            obs = arm.get_current_joints()
            row = {
                "backarm_deg": float(back),
                "frontarm_deg": float(front),
                "spoon_deg": float(spoon),
                "shoulder_lift": float(obs["shoulder_lift"]),
                "elbow_flex": float(obs["elbow_flex"]),
                "wrist_flex": float(obs["wrist_flex"]),
            }
            rows.append(row)
            print(f"Captured #{len(rows)}: {row}")
    finally:
        arm.shutdown()

    if not rows:
        print("No samples captured.")
        return

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backarm_deg",
                "frontarm_deg",
                "spoon_deg",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} samples -> {args.out}")


if __name__ == "__main__":
    main()
