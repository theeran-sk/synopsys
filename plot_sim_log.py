from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate PNG plots from simulator CSV log")
    p.add_argument("--csv", required=True, help="Path to log CSV from run_simulator.py")
    p.add_argument("--out-dir", default="", help="Output directory (default: same folder as CSV)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("CSV is empty.")
    t0 = float(df["timestamp"].iloc[0])
    t = df["timestamp"] - t0

    prefix = csv_path.stem

    plt.figure(figsize=(9, 5))
    plt.plot(t, df["target_x"], label="target_x")
    plt.plot(t, df["actual_x"], label="actual_x")
    plt.plot(t, df["target_z"], label="target_z")
    plt.plot(t, df["actual_z"], label="actual_z")
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.title("Target vs Actual")
    plt.grid(alpha=0.25)
    plt.legend()
    p1 = out_dir / f"{prefix}_target_vs_actual.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(t, df["error_x"], label="error_x")
    plt.plot(t, df["error_z"], label="error_z")
    plt.xlabel("time (s)")
    plt.ylabel("error (m)")
    plt.title("Tracking Error")
    plt.grid(alpha=0.25)
    plt.legend()
    p2 = out_dir / f"{prefix}_error.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(df["target_x"], df["target_z"], "r--", label="target path")
    plt.plot(df["actual_x"], df["actual_z"], "b-", label="actual path")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("XY Path (Planar X-Z)")
    plt.grid(alpha=0.25)
    plt.legend()
    p3 = out_dir / f"{prefix}_path.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=150)
    plt.close()

    rmse_x = float((df["error_x"] ** 2).mean() ** 0.5)
    rmse_z = float((df["error_z"] ** 2).mean() ** 0.5)
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")
    print(f"RMSE x={rmse_x*100:.2f} cm, z={rmse_z*100:.2f} cm")


if __name__ == "__main__":
    main()
