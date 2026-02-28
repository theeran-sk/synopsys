"""Abstract interface for perception backends that output mouth observations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from feeding.types import MouthObservation


class MouthPerception(ABC):
    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self, now: float) -> MouthObservation:
        raise NotImplementedError

    @abstractmethod
    def render(self, state_label: str) -> bool:
        """Render UI. Return False when user asks to quit."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
