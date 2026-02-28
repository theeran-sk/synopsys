"""Main feeding state machine: wait for open mouth, approach, and retreat on close/loss."""

from __future__ import annotations

import logging
import time
from enum import Enum

from feeding.arm.base import ArmBackend
from feeding.config import AppConfig
from feeding.gating import OpenDurationGate
from feeding.perception.base import MouthPerception
from feeding.planner import MouthTargetPlanner
from feeding.safety import SafetyFilter


class FeedState(str, Enum):
    WAIT_OPEN = "WAIT_OPEN"
    APPROACH = "APPROACH"
    RETREAT = "RETREAT"


class FeedingController:
    def __init__(
        self,
        cfg: AppConfig,
        perception: MouthPerception,
        arm: ArmBackend,
    ) -> None:
        self.cfg = cfg
        self.perception = perception
        self.arm = arm

        self.gate = OpenDurationGate(cfg.vision.open_hold_seconds)
        self.planner = MouthTargetPlanner(cfg.planning, cfg.safety)
        self.safety = SafetyFilter(cfg.safety)

        self.state = FeedState.WAIT_OPEN
        self._cooldown_until = 0.0
        self._last_obs_time = 0.0
        self._saw_closed_since_last_cycle = True
        self._last_approach_cmd_time = 0.0

    def _begin_retreat(self, now: float, reason: str) -> None:
        logging.info("Retreating to neutral: %s", reason)
        self.arm.stop(now)
        self.arm.retreat_to_neutral(now)
        self.gate.reset()
        self._cooldown_until = now + self.cfg.control.rearm_cooldown_seconds
        self.state = FeedState.RETREAT

    def run(self) -> None:
        period = 1.0 / max(self.cfg.control.loop_hz, 1.0)

        self.perception.start()
        self.arm.start()
        now = time.monotonic()
        self.arm.goto_neutral(now)
        logging.info("Controller started. Press q in camera window to quit.")

        try:
            while True:
                loop_start = time.monotonic()
                self.arm.tick(loop_start)

                obs = self.perception.read(loop_start)
                self._last_obs_time = obs.timestamp

                if self.state == FeedState.WAIT_OPEN:
                    if obs.detected and not obs.is_open:
                        self._saw_closed_since_last_cycle = True

                    if loop_start < self._cooldown_until:
                        self.gate.reset()
                    elif (
                        obs.detected
                        and (
                            self._saw_closed_since_last_cycle
                            or not self.cfg.control.require_close_before_rearm
                        )
                        and self.gate.update(loop_start, obs.is_open)
                    ):
                        self.state = FeedState.APPROACH
                        self.gate.reset()
                        self._saw_closed_since_last_cycle = False
                        self._last_approach_cmd_time = 0.0
                        logging.info("Mouth open gate passed, approach started.")
                    elif not obs.detected:
                        self.gate.reset()

                elif self.state == FeedState.APPROACH:
                    if not (obs.detected and obs.is_open):
                        self._begin_retreat(loop_start, "mouth closed or lost during approach")
                    else:
                        target = self.planner.plan_target(obs)
                        target = self.safety.clamp_pose(target)
                        if (
                            loop_start - self._last_approach_cmd_time
                            >= self.cfg.control.approach_update_period_s
                        ):
                            self.arm.start_approach(target, loop_start)
                            self._last_approach_cmd_time = loop_start

                elif self.state == FeedState.RETREAT:
                    if self.arm.is_motion_complete():
                        self.state = FeedState.WAIT_OPEN
                        logging.info("Back at neutral, waiting for mouth-open gate.")

                if self.state == FeedState.APPROACH:
                    vision_age = loop_start - self._last_obs_time
                    if vision_age > self.cfg.control.max_vision_stale_seconds:
                        self._begin_retreat(loop_start, "vision stale")

                if self.cfg.vision.show_camera:
                    keep_running = self.perception.render(self.state.value)
                    if not keep_running:
                        logging.info("Quit requested from UI.")
                        break

                elapsed = time.monotonic() - loop_start
                sleep_s = max(0.0, period - elapsed)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received, stopping controller.")

        finally:
            self.perception.close()
            self.arm.shutdown()
            logging.info("Controller stopped.")
