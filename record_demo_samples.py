from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
UPPER_INNER_LIP = 13
LOWER_INNER_LIP = 14


def _to_px(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    return int(np.clip(x * w, 0, w - 1)), int(np.clip(y * h, 0, h - 1))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Record demo dataset: mouth features -> target joint deltas.")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--port", type=str, default="/dev/tty.usbmodem5AB90674581")
    p.add_argument("--robot-id", type=str, default="synopsys2026")
    p.add_argument("--output", type=str, default="configs/demo_samples.csv")
    p.add_argument(
        "--neutral",
        type=str,
        default="0,-67.56043956043956,68.13186813186813,-77.89010989010988",
        help="Neutral joints as pan,shoulder_lift,elbow_flex,wrist_flex",
    )
    return p


def _parse_neutral(raw: str) -> dict[str, float]:
    vals = [float(x.strip()) for x in raw.split(",")]
    if len(vals) != 4:
        raise ValueError("neutral must have 4 comma-separated floats")
    return {
        "shoulder_pan": vals[0],
        "shoulder_lift": vals[1],
        "elbow_flex": vals[2],
        "wrist_flex": vals[3],
    }


def main() -> None:
    args = build_parser().parse_args()
    neutral = _parse_neutral(args.neutral)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["LEROBOT_SO_FOLLOWER_MOTORS"] = "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex"
    from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    robot_cfg = SOFollowerRobotConfig(
        port=args.port,
        id=args.robot_id,
        max_relative_target=30.0,
        disable_torque_on_disconnect=False,
    )
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    torque_enabled = True
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        robot.disconnect()
        raise RuntimeError("Could not open camera.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    header = [
        "timestamp",
        "mouth_cx_norm",
        "mouth_cy_norm",
        "eye_px",
        "openness_ratio",
        "target_shoulder_lift_delta",
        "target_elbow_flex_delta",
        "target_wrist_flex_delta",
    ]

    write_header = not out_path.exists()
    f = out_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)

    print("Demo recorder started.")
    print("Workflow per sample:")
    print("1) Press r to release torque, move arm by hand to desired near-mouth pose.")
    print("2) Press r again to stiffen/hold pose.")
    print("2) Keep mouth visible.")
    print("3) Press SPACE to record one sample.")
    print("Keys: SPACE=record, d=delete last, r=release/stiffen, q=quit.")

    def delete_last_sample() -> None:
        f.flush()
        with out_path.open("r", encoding="utf-8") as rf:
            lines = rf.readlines()
        if len(lines) <= 1:
            print("No samples to delete.")
            return
        # Keep header + all but last sample.
        with out_path.open("w", encoding="utf-8") as wf:
            wf.writelines(lines[:-1])
        print("Deleted last sample.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            feat = None
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                le = np.array(_to_px(lm[LEFT_EYE_OUTER].x, lm[LEFT_EYE_OUTER].y, w, h), dtype=float)
                re = np.array(_to_px(lm[RIGHT_EYE_OUTER].x, lm[RIGHT_EYE_OUTER].y, w, h), dtype=float)
                lm_l = np.array(_to_px(lm[LEFT_MOUTH_CORNER].x, lm[LEFT_MOUTH_CORNER].y, w, h), dtype=float)
                lm_r = np.array(_to_px(lm[RIGHT_MOUTH_CORNER].x, lm[RIGHT_MOUTH_CORNER].y, w, h), dtype=float)
                lm_u = np.array(_to_px(lm[UPPER_INNER_LIP].x, lm[UPPER_INNER_LIP].y, w, h), dtype=float)
                lm_d = np.array(_to_px(lm[LOWER_INNER_LIP].x, lm[LOWER_INNER_LIP].y, w, h), dtype=float)

                eye_px = float(np.linalg.norm(re - le))
                mouth_width = float(np.linalg.norm(lm_r - lm_l))
                mouth_height = float(np.linalg.norm(lm_d - lm_u))
                openness_ratio = mouth_height / (mouth_width + 1e-6)
                mouth_center = (lm_l + lm_r) / 2.0
                cx_norm = float(mouth_center[0] / max(w, 1))
                cy_norm = float(mouth_center[1] / max(h, 1))

                feat = (cx_norm, cy_norm, eye_px, openness_ratio)

                cv2.circle(frame, tuple(mouth_center.astype(int)), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"eye_px={eye_px:.1f} open={openness_ratio:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"SPACE=record, d=delete last, r=release/stiffen, q=quit | torque={'ON' if torque_enabled else 'OFF'}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 255, 180),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Demo Sample Recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                delete_last_sample()
                continue
            if key == ord("r"):
                try:
                    if torque_enabled:
                        for m in motor_names:
                            robot.bus.disable_torque(m, num_retry=1)
                        torque_enabled = False
                        print("Torque OFF: move arm by hand.")
                    else:
                        for m in motor_names:
                            robot.bus.enable_torque(m, num_retry=1)
                        torque_enabled = True
                        print("Torque ON: arm stiffened.")
                except Exception as e:
                    print(f"Failed to toggle torque: {e}")
                continue
            if key == ord(" "):
                if not torque_enabled:
                    print("Torque is OFF. Press r to stiffen before recording sample.")
                    continue
                if feat is None:
                    print("No face detected, sample not recorded.")
                    continue
                obs = robot.get_observation()
                cur = {
                    "shoulder_pan": float(obs["shoulder_pan.pos"]),
                    "shoulder_lift": float(obs["shoulder_lift.pos"]),
                    "elbow_flex": float(obs["elbow_flex.pos"]),
                    "wrist_flex": float(obs["wrist_flex.pos"]),
                }
                row = [
                    time.time(),
                    feat[0],
                    feat[1],
                    feat[2],
                    feat[3],
                    cur["shoulder_lift"] - neutral["shoulder_lift"],
                    cur["elbow_flex"] - neutral["elbow_flex"],
                    cur["wrist_flex"] - neutral["wrist_flex"],
                ]
                writer.writerow(row)
                f.flush()
                print(
                    "Recorded sample: "
                    f"d_sh={row[5]:.2f}, d_el={row[6]:.2f}, d_wr={row[7]:.2f}"
                )
    finally:
        f.close()
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
