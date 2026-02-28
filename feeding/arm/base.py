"""Abstract interface for arm backends used by the feeding controller."""

from __future__ import annotations

from abc import ABC, abstractmethod

from feeding.types import Pose3D


class ArmBackend(ABC):
    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def goto_neutral(self, now: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_approach(self, target: Pose3D, now: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def retreat_to_neutral(self, now: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self, now: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def tick(self, now: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_motion_complete(self) -> bool:
        raise NotImplementedError
