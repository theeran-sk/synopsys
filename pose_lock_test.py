from __future__ import annotations

import argparse
import os
import sys
import termios
import time
import tty


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lock SO-101 into test poses with horizontal spoon compensation.")
    p.add_argument("--port", type=str, default="/dev/tty.usbmodem5AB90674581")
    p.add_argument("--robot-id", type=str, default="synopsys2026")
    p.add_argument("--move-time-s", type=float, default=2.5, help="Seconds per pose move.")
    p.add_argument("--rate-hz", type=float, default=20.0, help="Interpolation update rate.")
    p.add_argument("--pan", type=float, default=0.0)
    p.add_argument("--auto", action="store_true", help="Run through poses automatically.")
    return p


def read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        return ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> None:
    args = build_parser().parse_args()
    os.environ["LEROBOT_SO_FOLLOWER_MOTORS"] = "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex"

    from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cfg = SOFollowerRobotConfig(
        port=args.port,
        id=args.robot_id,
        max_relative_target=10.0,
        disable_torque_on_disconnect=True,
    )
    robot = make_robot_from_config(cfg)
    robot.connect()

    neutral = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -67.56043956043956,
        "elbow_flex": 68.13186813186813,
        "wrist_flex": -77.89010989010988,
    }

    def make_pose(shoulder: float, elbow: float, pan: float) -> dict[str, float]:
        # Horizontal spoon rule fitted from your measured poses.
        wrist = -(shoulder + elbow) - 78.5
        wrist = max(min(wrist, 120.0), -120.0)
        return {
            "shoulder_pan": pan,
            "shoulder_lift": shoulder,
            "elbow_flex": elbow,
            "wrist_flex": wrist,
        }

    def read_pose() -> dict[str, float]:
        obs = robot.get_observation()
        return {
            "shoulder_pan": float(obs["shoulder_pan.pos"]),
            "shoulder_lift": float(obs["shoulder_lift.pos"]),
            "elbow_flex": float(obs["elbow_flex.pos"]),
            "wrist_flex": float(obs["wrist_flex.pos"]),
        }

    def send_pose_slow(target: dict[str, float], move_time_s: float, rate_hz: float) -> None:
        start = read_pose()
        steps = max(2, int(move_time_s * max(rate_hz, 1.0)))
        dt = move_time_s / steps
        print("Target:", target)
        for i in range(1, steps + 1):
            a = i / steps
            cmd = {}
            for k in target:
                cmd[f"{k}.pos"] = (1.0 - a) * start[k] + a * target[k]
            robot.send_action(cmd)
            time.sleep(dt)

    tests = [
        (-40.0, 20.0),
        (-20.0, 50.0),
        (10.0, 0.0),
        (30.0, -20.0),
        (45.0, 15.0),
    ]

    try:
        # Always normalize start state so behavior is deterministic.
        print("Moving to neutral first...")
        send_pose_slow(neutral, move_time_s=2.5, rate_hz=args.rate_hz)

        if args.auto:
            for shoulder, elbow in tests:
                target = make_pose(shoulder, elbow, pan=args.pan)
                send_pose_slow(target, move_time_s=args.move_time_s, rate_hz=args.rate_hz)
            print("Auto run complete.")
        else:
            i = 0
            while True:
                shoulder, elbow = tests[i]
                print(f"\nPose {i + 1}/{len(tests)}")
                target = make_pose(shoulder, elbow, pan=args.pan)
                send_pose_slow(target, move_time_s=args.move_time_s, rate_hz=args.rate_hz)
                print("Press n=next, b=back, q=quit")
                cmd = read_key()
                print(f"  key: {cmd}")
                if cmd == "q":
                    break
                if cmd == "b":
                    i = max(0, i - 1)
                    continue
                if cmd == "n":
                    i = min(len(tests) - 1, i + 1)
                    continue
    finally:
        robot.disconnect()
        print("Done")


if __name__ == "__main__":
    main()
