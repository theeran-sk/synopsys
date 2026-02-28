from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


FEATURES = ["mouth_cx_norm", "mouth_cy_norm", "eye_px", "openness_ratio"]
TARGETS = [
    "target_shoulder_lift_delta",
    "target_elbow_flex_delta",
    "target_wrist_flex_delta",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit simple linear model from recorded demo samples.")
    p.add_argument("--input", type=str, default="configs/demo_samples.csv")
    p.add_argument("--output", type=str, default="configs/demo_model.yaml")
    p.add_argument("--ridge-lambda", type=float, default=1e-2)
    return p


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if len(rows) < 10:
        raise RuntimeError(f"Need at least 10 samples, got {len(rows)}.")

    x = np.array([[float(r[k]) for k in FEATURES] for r in rows], dtype=float)
    y = np.array([[float(r[k]) for k in TARGETS] for r in rows], dtype=float)
    return x, y


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    # Add bias term
    ones = np.ones((x.shape[0], 1), dtype=float)
    xb = np.concatenate([x, ones], axis=1)
    n_feat = xb.shape[1]
    reg = lam * np.eye(n_feat, dtype=float)
    reg[-1, -1] = 0.0  # don't regularize bias
    w = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)  # (n_feat, n_targets)
    weights = w[:-1, :]  # (4,4)
    bias = w[-1, :]  # (4,)
    return weights, bias


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    args = build_parser().parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x, y = load_csv(in_path)
    w, b = fit_ridge(x, y, args.ridge_lambda)
    y_hat = x @ w + b
    total_rmse = rmse(y, y_hat)
    per_target_rmse = {
        TARGETS[i]: rmse(y[:, i : i + 1], y_hat[:, i : i + 1])
        for i in range(len(TARGETS))
    }

    payload = {
        "model_type": "linear_ridge",
        "features": FEATURES,
        "targets": TARGETS,
        "weights": w.tolist(),  # shape (4 features, 4 targets)
        "bias": b.tolist(),  # shape (4,)
        "num_samples": int(x.shape[0]),
        "ridge_lambda": float(args.ridge_lambda),
        "rmse_total": total_rmse,
        "rmse_per_target": per_target_rmse,
    }

    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    print(f"Saved model to {out_path}")
    print(f"Samples: {x.shape[0]}")
    print(f"Total RMSE: {total_rmse:.3f}")
    print("Per-target RMSE:")
    for k, v in per_target_rmse.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
