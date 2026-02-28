"""Timing gate helpers used to require continuous mouth-open duration."""

from __future__ import annotations


class OpenDurationGate:
    """Triggers when mouth stays open continuously for required duration."""

    def __init__(self, required_open_seconds: float) -> None:
        self.required_open_seconds = required_open_seconds
        self._open_since: float | None = None

    def reset(self) -> None:
        self._open_since = None

    def update(self, now: float, is_open: bool) -> bool:
        if not is_open:
            self._open_since = None
            return False

        if self._open_since is None:
            self._open_since = now
            return False

        return (now - self._open_since) >= self.required_open_seconds
