"""Validate 2D Cartesian IK mapping (forward/up) on the LeRobot arm."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

from feeding.arm.lerobot_arm import LeRobotArmBackend
from feeding.config import load_config


@dataclass
class OffsetStep:
    dx_m: float
    dz_m: float


def _parse_offsets(raw: str) -> list[OffsetStep]:
    steps: list[OffsetStep] = []
    chunks = [c.strip() for c in raw.split(";") if c.strip()]
    if not chunks:
        raise ValueError("No offsets provided.")

    for chunk in chunks:
        pair = [p.strip() for p in chunk.split(",") if p.strip()]
        if len(pair) != 2:
            raise ValueError(
                f"Invalid offset '{chunk}'. Expected 'dx,dz' with meters, e.g. '0.03,0.00'."
            )
        steps.append(OffsetStep(dx_m=float(pair[0]), dz_m=float(pair[1])))
    return steps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate planar Cartesian offsets (dx,dz) against achieved tip motion.",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/feeding_default.yaml",
        help="Path to runtime YAML config.",
    )
    p.add_argument(
        "--offsets",
        type=str,
        default="0.00,0.00;0.03,0.00;0.00,0.03;0.03,0.03;0.00,0.00",
        help="Semicolon-separated offsets in meters: 'dx,dz;dx,dz;...'",
    )
    p.add_argument(
        "--settle-s",
        type=float,
        default=1.8,
        help="Wait time after each command before measurement.",
    )
    p.add_argument(
        "--pause-s",
        type=float,
        default=0.5,
        help="Extra pause between steps.",
    )
    return p


def _wait_with_ticks(arm: LeRobotArmBackend, duration_s: float) -> None:
    end = time.monotonic() + max(0.0, duration_s)
    while time.monotonic() < end:
        arm.tick(time.monotonic())
        time.sleep(0.03)


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    arm = LeRobotArmBackend(cfg.safety, cfg.lerobot)

    steps = _parse_offsets(args.offsets)
    logging.info("Loaded %d Cartesian steps.", len(steps))

    arm.start()
    try:
        now = time.monotonic()
        arm.goto_neutral(now)
        _wait_with_ticks(arm, max(cfg.lerobot.move_duration_s, 1.0))

        neutral_tip = arm.get_neutral_tip_pose()
        logging.info(
            "Neutral tip pose x,z = (%.4f, %.4f) m",
            neutral_tip[0],
            neutral_tip[1],
        )

        print("")
        print("Step | Requested (dx,dz)m | Achieved (dx,dz)m | Error (dx,dz)m | Saturated")
        print("-----+---------------------+-------------------+-----------------+----------")

        for i, step in enumerate(steps, start=1):
            now = time.monotonic()
            achieved_dx_cmd, achieved_dz_cmd, saturated = arm.start_cartesian_offset_from_neutral(
                requested_dx=step.dx_m,
                requested_dz=step.dz_m,
                now=now,
            )

            _wait_with_ticks(arm, args.settle_s)
            tip = arm.get_current_tip_pose()
            if tip is None:
                raise RuntimeError("Could not read current tip pose after command.")

            achieved_dx_meas = tip[0] - neutral_tip[0]
            achieved_dz_meas = tip[1] - neutral_tip[1]
            err_dx = achieved_dx_meas - step.dx_m
            err_dz = achieved_dz_meas - step.dz_m

            print(
                f"{i:>4} | ({step.dx_m:+.3f},{step.dz_m:+.3f})      | "
                f"({achieved_dx_meas:+.3f},{achieved_dz_meas:+.3f})    | "
                f"({err_dx:+.3f},{err_dz:+.3f})   | {str(saturated):>8}"
            )
            logging.info(
                "Step %d command-solver achieved (dx,dz)=(%.4f, %.4f) m",
                i,
                achieved_dx_cmd,
                achieved_dz_cmd,
            )
            _wait_with_ticks(arm, args.pause_s)

        now = time.monotonic()
        arm.retreat_to_neutral(now)
        _wait_with_ticks(arm, max(cfg.lerobot.move_duration_s, 1.0))

    finally:
        arm.shutdown()


if __name__ == "__main__":
    main()
