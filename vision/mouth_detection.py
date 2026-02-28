"""Real-time mouth detection and open/closed classification.

This script:
- Opens webcam feed
- Detects face landmarks with MediaPipe Face Mesh
- Locates mouth landmarks
- Computes mouth center in pixel coordinates
- Estimates mouth openness (OPEN/CLOSED)
- Draws mouth region and center point
- Prints center + state continuously

Press 'q' to quit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


# Outer lip contour indices for visualization and center estimation.
OUTER_LIP_LANDMARKS: Sequence[int] = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    415,
    310,
    311,
    312,
    13,
    82,
    81,
    42,
    183,
    78,
)

# Specific landmarks used for open/closed estimation.
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
UPPER_INNER_LIP = 13
LOWER_INNER_LIP = 14


def get_face_mesh_module():
    """Return MediaPipe face_mesh module across package variants."""
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return mp.solutions.face_mesh
    # Fallback for installs where `mediapipe.solutions` isn't exposed at top-level.
    from mediapipe.python.solutions import face_mesh

    return face_mesh


@dataclass
class MouthDetection:
    """Mouth detection output in pixel space."""

    mouth_points: np.ndarray
    center_xy: Tuple[int, int]
    openness_ratio: float
    is_open: bool


class MouthDetector:
    """Handles webcam capture, mouth tracking, and mouth state classification."""

    def __init__(self, camera_index: int = 0, open_threshold: float = 0.12) -> None:
        self.camera_index = camera_index
        self.open_threshold = open_threshold
        self.cap: Optional[cv2.VideoCapture] = None

        self.mp_face_mesh = get_face_mesh_module()
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def open_camera(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap.isOpened()

    @staticmethod
    def _to_pixel_coords(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
        px = int(np.clip(x * width, 0, width - 1))
        py = int(np.clip(y * height, 0, height - 1))
        return px, py

    def detect_mouth(self, frame_bgr: np.ndarray) -> Optional[MouthDetection]:
        """Return mouth location + state from a single frame."""
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

        # Mouth center for downstream robotic targeting.
        center = mouth_points_np.mean(axis=0).astype(int)
        center_xy = (int(center[0]), int(center[1]))

        # Openness ratio (normalized) to reduce distance-to-camera sensitivity.
        left = np.array(self._to_pixel_coords(landmarks[LEFT_MOUTH_CORNER].x, landmarks[LEFT_MOUTH_CORNER].y, w, h))
        right = np.array(self._to_pixel_coords(landmarks[RIGHT_MOUTH_CORNER].x, landmarks[RIGHT_MOUTH_CORNER].y, w, h))
        top = np.array(self._to_pixel_coords(landmarks[UPPER_INNER_LIP].x, landmarks[UPPER_INNER_LIP].y, w, h))
        bottom = np.array(self._to_pixel_coords(landmarks[LOWER_INNER_LIP].x, landmarks[LOWER_INNER_LIP].y, w, h))

        mouth_width = np.linalg.norm(right - left)
        mouth_height = np.linalg.norm(bottom - top)

        openness_ratio = float(mouth_height / (mouth_width + 1e-6))
        is_open = openness_ratio > self.open_threshold

        return MouthDetection(
            mouth_points=mouth_points_np,
            center_xy=center_xy,
            openness_ratio=openness_ratio,
            is_open=is_open,
        )

    @staticmethod
    def draw_overlay(frame_bgr: np.ndarray, detection: MouthDetection) -> None:
        """Draw mouth contour, center point, and state text."""
        hull = cv2.convexHull(detection.mouth_points)
        color = (0, 200, 0) if detection.is_open else (0, 140, 255)

        cv2.polylines(frame_bgr, [hull], isClosed=True, color=color, thickness=2)
        cv2.circle(frame_bgr, detection.center_xy, radius=5, color=(0, 0, 255), thickness=-1)

        state_text = "OPEN" if detection.is_open else "CLOSED"
        cv2.putText(
            frame_bgr,
            f"Mouth: {state_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Center: {detection.center_xy}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Openness ratio: {detection.openness_ratio:.3f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def run(self) -> None:
        if not self.open_camera():
            print("Error: Could not open webcam. Check camera permissions or camera index.")
            return

        print("Starting mouth detection. Press 'q' to quit.")

        try:
            while True:
                assert self.cap is not None
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    print("Warning: Failed to read frame from webcam.")
                    break

                detection = self.detect_mouth(frame)
                if detection is not None:
                    self.draw_overlay(frame, detection)
                    state = "OPEN" if detection.is_open else "CLOSED"
                    print(
                        f"Mouth center: {detection.center_xy} | state: {state} | openness: {detection.openness_ratio:.3f}",
                        end="\r",
                        flush=True,
                    )
                else:
                    print("Mouth center: Not detected | state: UNKNOWN          ", end="\r", flush=True)

                cv2.imshow("Mouth Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.close()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.face_mesh.close()
        cv2.destroyAllWindows()
        print("\nClosed mouth detection cleanly.")


def main() -> int:
    detector = MouthDetector(camera_index=0, open_threshold=0.12)
    detector.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
