from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml

# FaceMesh landmark indices
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291


def _to_px(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    return int(np.clip(x * w, 0, w - 1)), int(np.clip(y * h, 0, h - 1))


def _parse_distances(raw: str) -> list[float]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    out = [float(v) for v in vals]
    if len(out) < 3:
        raise ValueError("Provide at least 3 distances, e.g. 25,35,45")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibrate camera-to-mouth distance estimator from landmarks.")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--distances-cm", type=str, default="25,35,45")
    p.add_argument("--output", type=str, default="configs/mouth_distance_calibration.yaml")
    p.add_argument("--buffer-size", type=int, default=20)
    p.add_argument(
        "--feature",
        type=str,
        choices=["eye", "mouth", "ratio"],
        default="eye",
        help="Feature used for fit: eye_px, mouth_px, or mouth_px/eye_px ratio",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    distances_cm = _parse_distances(args.distances_cm)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    ratio_buf: deque[float] = deque(maxlen=max(5, args.buffer_size))
    eye_buf: deque[float] = deque(maxlen=max(5, args.buffer_size))
    mouth_buf: deque[float] = deque(maxlen=max(5, args.buffer_size))
    captured_feature_vals: list[float] = []
    captured_ratios: list[float] = []
    captured_eye_px: list[float] = []
    captured_mouth_px: list[float] = []
    stage_idx = 0

    print("Distance calibration starting.")
    print("For each requested distance, place your mouth that far from camera lens and press SPACE.")
    print("Press q anytime to quit.")
    print("")

    announced_stage = -1

    try:
        while stage_idx < len(distances_cm):
            if stage_idx != announced_stage:
                target_cm = distances_cm[stage_idx]
                print(f"[Step {stage_idx + 1}/{len(distances_cm)}] Target = {target_cm:.1f} cm. Press SPACE to capture.")
                announced_stage = stage_idx

            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            ratio = None
            eye_px = None
            mouth_px = None
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                le = np.array(_to_px(lms[LEFT_EYE_OUTER].x, lms[LEFT_EYE_OUTER].y, w, h), dtype=float)
                re = np.array(_to_px(lms[RIGHT_EYE_OUTER].x, lms[RIGHT_EYE_OUTER].y, w, h), dtype=float)
                lm = np.array(_to_px(lms[LEFT_MOUTH_CORNER].x, lms[LEFT_MOUTH_CORNER].y, w, h), dtype=float)
                rm = np.array(_to_px(lms[RIGHT_MOUTH_CORNER].x, lms[RIGHT_MOUTH_CORNER].y, w, h), dtype=float)

                eye_px = float(np.linalg.norm(re - le))
                mouth_px = float(np.linalg.norm(rm - lm))
                eye_buf.append(eye_px)
                mouth_buf.append(mouth_px)
                if eye_px > 1.0:
                    ratio = mouth_px / eye_px
                    ratio_buf.append(ratio)

                cv2.circle(frame, tuple(le.astype(int)), 4, (0, 255, 255), -1)
                cv2.circle(frame, tuple(re.astype(int)), 4, (0, 255, 255), -1)
                cv2.circle(frame, tuple(lm.astype(int)), 4, (0, 255, 0), -1)
                cv2.circle(frame, tuple(rm.astype(int)), 4, (0, 255, 0), -1)

                cv2.line(frame, tuple(le.astype(int)), tuple(re.astype(int)), (0, 255, 255), 1)
                cv2.line(frame, tuple(lm.astype(int)), tuple(rm.astype(int)), (0, 255, 0), 1)

            target_cm = distances_cm[stage_idx]
            ratio_text = f"{ratio:.4f}" if ratio is not None else "n/a"
            med_ratio = float(np.median(ratio_buf)) if ratio_buf else float("nan")
            med_text = f"{med_ratio:.4f}" if ratio_buf else "n/a"

            cv2.putText(
                frame,
                f"Target distance: {target_cm:.1f} cm  (SPACE to capture)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Current ratio mouth/eye: {ratio_text} | feature={args.feature}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 255, 200),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Median ratio buffer: {med_text}",
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Mouth Distance Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested.")
                return
            if key == ord(" "):
                if len(ratio_buf) < 5:
                    print(
                        f"Not enough stable face data yet for {target_cm:.1f} cm; "
                        "wait a moment and press SPACE again."
                    )
                    continue
                use_ratio = float(np.median(ratio_buf))
                use_eye = float(np.median(eye_buf)) if eye_buf else float("nan")
                use_mouth = float(np.median(mouth_buf)) if mouth_buf else float("nan")

                if args.feature == "eye":
                    feature_val = use_eye
                elif args.feature == "mouth":
                    feature_val = use_mouth
                else:
                    feature_val = use_ratio

                captured_feature_vals.append(feature_val)
                captured_ratios.append(use_ratio)
                captured_eye_px.append(use_eye)
                captured_mouth_px.append(use_mouth)
                print(
                    f"Captured {target_cm:.1f} cm -> feature({args.feature})={feature_val:.5f}, "
                    f"eye_px={use_eye:.2f}, mouth_px={use_mouth:.2f}, ratio={use_ratio:.5f}"
                )
                ratio_buf.clear()
                eye_buf.clear()
                mouth_buf.clear()
                stage_idx += 1

        y = np.array(distances_cm, dtype=float)
        fvals = np.array(captured_feature_vals, dtype=float)
        x = 1.0 / np.clip(fvals, 1e-6, None)

        # distance_cm ~= a * (1/feature) + b
        a, b = np.polyfit(x, y, 1)
        pred = a * x + b
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))

        payload = {
            "model": "distance_cm = a * (1/feature) + b",
            "feature": args.feature,
            "feature_definition": {
                "eye": "feature = eye_px",
                "mouth": "feature = mouth_px",
                "ratio": "feature = mouth_px/eye_px",
            }[args.feature],
            "camera_index": args.camera_index,
            "a": float(a),
            "b": float(b),
            "rmse_cm": rmse,
            "samples": [
                {
                    "distance_cm": float(dc),
                    "feature_value": float(fv),
                    "eye_px": float(ep),
                    "mouth_px": float(mp),
                    "ratio": float(rv),
                }
                for dc, fv, ep, mp, rv in zip(
                    distances_cm, captured_feature_vals, captured_eye_px, captured_mouth_px, captured_ratios, strict=True
                )
            ],
        }

        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

        print(f"Saved calibration to {out_path}")
        print(f"Fitted ({args.feature}): distance_cm = {a:.4f} * (1/feature) + {b:.4f}")
        print(f"RMSE: {rmse:.2f} cm")
    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
