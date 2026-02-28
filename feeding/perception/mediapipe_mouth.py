"""MediaPipe FaceMesh-based mouth detector and camera rendering utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np

from feeding.config import VisionConfig
from feeding.perception.base import MouthPerception
from feeding.types import MouthObservation


OUTER_LIP_LANDMARKS: Sequence[int] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    415, 310, 311, 312, 13, 82, 81, 42, 183, 78,
)

LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
UPPER_INNER_LIP = 13
LOWER_INNER_LIP = 14
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


def _get_face_mesh_module():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return mp.solutions.face_mesh
    from mediapipe.python.solutions import face_mesh
    return face_mesh


@dataclass
class _InternalDetection:
    center_xy: tuple[int, int]
    openness_ratio: float
    is_open: bool
    eye_px: float
    mouth_px: float
    mouth_points: np.ndarray


class MediaPipeMouthPerception(MouthPerception):
    def __init__(self, cfg: VisionConfig) -> None:
        self.cfg = cfg
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_detection: Optional[_InternalDetection] = None

        face_mesh_module = _get_face_mesh_module()
        self.face_mesh = face_mesh_module.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _to_pixel_coords(x: float, y: float, width: int, height: int) -> tuple[int, int]:
        px = int(np.clip(x * width, 0, width - 1))
        py = int(np.clip(y * height, 0, height - 1))
        return px, py

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.cfg.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check camera permissions or set camera_index in config."
            )

    def _detect(self, frame_bgr: np.ndarray) -> Optional[_InternalDetection]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]

        mouth_points = []
        for idx in OUTER_LIP_LANDMARKS:
            lm = landmarks[idx]
            mouth_points.append(self._to_pixel_coords(lm.x, lm.y, w, h))
        mouth_points_np = np.array(mouth_points, dtype=np.int32)

        center = mouth_points_np.mean(axis=0).astype(int)
        center_xy = (int(center[0]), int(center[1]))

        left = np.array(self._to_pixel_coords(landmarks[LEFT_MOUTH_CORNER].x, landmarks[LEFT_MOUTH_CORNER].y, w, h))
        right = np.array(self._to_pixel_coords(landmarks[RIGHT_MOUTH_CORNER].x, landmarks[RIGHT_MOUTH_CORNER].y, w, h))
        top = np.array(self._to_pixel_coords(landmarks[UPPER_INNER_LIP].x, landmarks[UPPER_INNER_LIP].y, w, h))
        bottom = np.array(self._to_pixel_coords(landmarks[LOWER_INNER_LIP].x, landmarks[LOWER_INNER_LIP].y, w, h))

        mouth_width = np.linalg.norm(right - left)
        mouth_height = np.linalg.norm(bottom - top)
        openness_ratio = float(mouth_height / (mouth_width + 1e-6))
        left_eye = np.array(self._to_pixel_coords(landmarks[LEFT_EYE_OUTER].x, landmarks[LEFT_EYE_OUTER].y, w, h))
        right_eye = np.array(self._to_pixel_coords(landmarks[RIGHT_EYE_OUTER].x, landmarks[RIGHT_EYE_OUTER].y, w, h))
        eye_px = float(np.linalg.norm(right_eye - left_eye))

        return _InternalDetection(
            center_xy=center_xy,
            openness_ratio=openness_ratio,
            is_open=openness_ratio > self.cfg.open_threshold,
            eye_px=eye_px,
            mouth_px=float(mouth_width),
            mouth_points=mouth_points_np,
        )

    def read(self, now: float) -> MouthObservation:
        if self.cap is None:
            raise RuntimeError("Perception is not started.")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.last_frame = None
            self.last_detection = None
            return MouthObservation(
                timestamp=now,
                detected=False,
                is_open=False,
                openness_ratio=0.0,
                eye_px=None,
                mouth_px=None,
                center_xy=None,
                frame_size=None,
            )

        self.last_frame = frame
        detection = self._detect(frame)
        self.last_detection = detection

        if detection is None:
            h, w = frame.shape[:2]
            return MouthObservation(
                timestamp=now,
                detected=False,
                is_open=False,
                openness_ratio=0.0,
                eye_px=None,
                mouth_px=None,
                center_xy=None,
                frame_size=(w, h),
            )

        h, w = frame.shape[:2]
        return MouthObservation(
            timestamp=now,
            detected=True,
            is_open=detection.is_open,
            openness_ratio=detection.openness_ratio,
            eye_px=detection.eye_px,
            mouth_px=detection.mouth_px,
            center_xy=detection.center_xy,
            frame_size=(w, h),
        )

    def render(self, state_label: str) -> bool:
        if self.last_frame is None:
            return True

        frame = self.last_frame.copy()
        if self.last_detection is not None:
            hull = cv2.convexHull(self.last_detection.mouth_points)
            color = (0, 200, 0) if self.last_detection.is_open else (0, 140, 255)
            cv2.polylines(frame, [hull], isClosed=True, color=color, thickness=2)
            cv2.circle(frame, self.last_detection.center_xy, radius=5, color=(0, 0, 255), thickness=-1)

            cv2.putText(
                frame,
                f"Mouth: {'OPEN' if self.last_detection.is_open else 'CLOSED'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Open ratio: {self.last_detection.openness_ratio:.3f}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"State: {state_label}",
            (10, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Feeding Assistant v1", frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.face_mesh.close()
        cv2.destroyAllWindows()
