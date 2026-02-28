from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import pandas as pd
import yaml


@dataclass
class FitResult:
    signs: dict[str, float]
    offsets: dict[str, float]
    rmse_deg: float


def _fit_with_signs(df: pd.DataFrame, s_sh: float, s_el: float, s_wr: float) -> FitResult:
    t1 = df["backarm_deg"].to_numpy(float)
    t2 = df["frontarm_deg"].to_numpy(float)
    t3 = df["spoon_deg"].to_numpy(float)
    m1 = df["shoulder_lift"].to_numpy(float)
    m2 = df["elbow_flex"].to_numpy(float)
    m3 = df["wrist_flex"].to_numpy(float)

    o1 = float((m1 - (t1 / s_sh)).mean())
    rel2 = t2 - t1
    o2 = float((m2 - (rel2 / s_el)).mean())
    rel3 = t3 - t2
    o3 = float((m3 - (rel3 / s_wr)).mean())

    t1_hat = s_sh * (m1 - o1)
    t2_hat = t1_hat + s_el * (m2 - o2)
    t3_hat = t2_hat + s_wr * (m3 - o3)

    err = ((t1_hat - t1) ** 2 + (t2_hat - t2) ** 2 + (t3_hat - t3) ** 2) / 3.0
    rmse = float((err.mean()) ** 0.5)
    return FitResult(
        signs={"shoulder_lift": s_sh, "elbow_flex": s_el, "wrist_flex": s_wr},
        offsets={"shoulder_lift": o1, "elbow_flex": o2, "wrist_flex": o3},
        rmse_deg=rmse,
    )


def fit_best(df: pd.DataFrame) -> FitResult:
    best: FitResult | None = None
    for s_sh, s_el, s_wr in itertools.product([1.0, -1.0], repeat=3):
        res = _fit_with_signs(df, s_sh, s_el, s_wr)
        if best is None or res.rmse_deg < best.rmse_deg:
            best = res
    assert best is not None
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Fit IK motor->kinematic sign/offset map")
    p.add_argument("--csv", type=str, required=True, help="CSV with measured angles and motor readings")
    p.add_argument(
        "--out",
        type=str,
        default="configs/ik_motor_map.yaml",
        help="Output YAML file",
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    required = [
        "backarm_deg",
        "frontarm_deg",
        "spoon_deg",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    res = fit_best(df)
    out = {
        "ik_motor_sign": res.signs,
        "ik_motor_offset_deg": {k: round(v, 6) for k, v in res.offsets.items()},
        "fit_rmse_deg": round(res.rmse_deg, 4),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("Best fit:")
    print(out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
