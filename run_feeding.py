"""Run the feeding assistant using either fake or LeRobot hardware backends."""

from __future__ import annotations

import argparse
import logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feeding Assistant v1")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/feeding_default.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["fake", "lerobot"],
        default=None,
        help="Override backend from config",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable camera visualization window",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from feeding.arm.fake_arm import FakeArmBackend
    from feeding.arm.lerobot_arm import LeRobotArmBackend
    from feeding.config import load_config
    from feeding.controller import FeedingController
    from feeding.perception.mediapipe_mouth import MediaPipeMouthPerception

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config)
    if args.backend is not None:
        cfg.backend = args.backend
    if args.headless:
        cfg.vision.show_camera = False

    perception = MediaPipeMouthPerception(cfg.vision)

    if cfg.backend == "fake":
        arm = FakeArmBackend(cfg.safety)
    elif cfg.backend == "lerobot":
        arm = LeRobotArmBackend(cfg.safety, cfg.lerobot)
    else:
        raise ValueError(f"Unsupported backend: {cfg.backend}")

    controller = FeedingController(cfg, perception, arm)
    controller.run()


if __name__ == "__main__":
    main()
