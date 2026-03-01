"""Diagnostic: verify IK model matches physical arm and log joint targets for approach."""

import sys
import math
import logging

sys.path.insert(0, ".")
logging.basicConfig(level=logging.DEBUG, format="%(message)s")

from feeding.config import load_config
from feeding.kinematics import Planar3LinkHorizontalIK
from feeding.types import Pose3D

cfg = load_config("configs/feeding_default.yaml")
rc = cfg.lerobot
sc = cfg.safety

# --- Neutral reference ---
neutral_joints = dict(rc.neutral_joints_deg)
print("=== Neutral joints (from config) ===")
for k, v in neutral_joints.items():
    print(f"  {k}: {v:.3f}°")


def to_kin_deg(joint, motor_deg):
    sign = float(rc.ik_motor_sign.get(joint, 1.0))
    if abs(sign) < 1e-9:
        sign = 1.0
    offset = float(rc.ik_motor_offset_deg.get(joint, 0.0))
    return sign * (motor_deg - offset)


def to_motor_deg(joint, kin_deg):
    sign = float(rc.ik_motor_sign.get(joint, 1.0))
    if abs(sign) < 1e-9:
        sign = 1.0
    offset = float(rc.ik_motor_offset_deg.get(joint, 0.0))
    return (kin_deg / sign) + offset


ik = Planar3LinkHorizontalIK(
    l1_m=rc.link_l1_m,
    l2_m=rc.link_l2_m,
    l3_forward_m=rc.spoon_offset_forward_m,
    l3_up_m=rc.spoon_offset_up_m,
)

# Neutral kinematic angles
q1_n = math.radians(to_kin_deg("shoulder_lift", neutral_joints["shoulder_lift"]))
q2_n = math.radians(to_kin_deg("elbow_flex", neutral_joints["elbow_flex"]))
q3_n = math.radians(to_kin_deg("wrist_flex", neutral_joints["wrist_flex"]))
phi_ref = q1_n + q2_n + q3_n
tip_x_n, tip_z_n = ik.forward(q1_n, q2_n, q3_n)

print(f"\n=== Neutral kinematic angles ===")
print(f"  q1 (shoulder_lift kin): {math.degrees(q1_n):.2f}°")
print(f"  q2 (elbow_flex kin):    {math.degrees(q2_n):.2f}°")
print(f"  q3 (wrist_flex kin):    {math.degrees(q3_n):.2f}°")
print(f"  phi (sum):              {math.degrees(phi_ref):.2f}°")
print(f"  Neutral tip (FK):       x={tip_x_n:.4f}m  z={tip_z_n:.4f}m")
print(f"  Neutral Cartesian:      ({sc.neutral_pose_xyz[0]}, {sc.neutral_pose_xyz[1]}, {sc.neutral_pose_xyz[2]})")

# --- Approach target ---
# Typical target when mouth is centered in frame
target = Pose3D(x=cfg.planning.mouth_target_x_m, y=0.0, z=sc.neutral_pose_xyz[2])
dx = target.x - sc.neutral_pose_xyz[0]
dz = target.z - sc.neutral_pose_xyz[2]
print(f"\n=== Approach target ===")
print(f"  Target Cartesian: ({target.x}, {target.y}, {target.z})")
print(f"  dx={dx:.4f}m  dz={dz:.4f}m")

desired_x = tip_x_n + dx
desired_z = tip_z_n + dz
print(f"  IK target tip:    x={desired_x:.4f}m  z={desired_z:.4f}m")

sol = ik.solve(x_tip=desired_x, z_tip=desired_z, phi=phi_ref, seed_q2=q2_n)
if sol is None:
    print("  *** IK FAILED at full scale! ***")
    for scale in (0.85, 0.7, 0.55, 0.4, 0.25):
        sol = ik.solve(
            x_tip=tip_x_n + scale * dx,
            z_tip=tip_z_n + scale * dz,
            phi=phi_ref,
            seed_q2=q2_n,
        )
        if sol:
            print(f"  IK succeeded at scale={scale}")
            break
    if sol is None:
        print("  *** IK failed at all scales! ***")

if sol:
    print(f"\n=== IK solution ===")
    print(f"  q1 (kin): {math.degrees(sol.q1):.2f}°  → motor shoulder_lift: {to_motor_deg('shoulder_lift', math.degrees(sol.q1)):.2f}°")
    print(f"  q2 (kin): {math.degrees(sol.q2):.2f}°  → motor elbow_flex:    {to_motor_deg('elbow_flex', math.degrees(sol.q2)):.2f}°")
    print(f"  q3 (kin): {math.degrees(sol.q3):.2f}°  → motor wrist_flex:    {to_motor_deg('wrist_flex', math.degrees(sol.q3)):.2f}°")

    # Deltas from neutral
    d_sl = to_motor_deg("shoulder_lift", math.degrees(sol.q1)) - neutral_joints["shoulder_lift"]
    d_ef = to_motor_deg("elbow_flex", math.degrees(sol.q2)) - neutral_joints["elbow_flex"]
    d_wf = to_motor_deg("wrist_flex", math.degrees(sol.q3)) - neutral_joints["wrist_flex"]
    print(f"\n=== Motor deltas from neutral (for {dx*100:.1f}cm forward, {dz*100:.1f}cm vertical) ===")
    print(f"  shoulder_lift: {d_sl:+.2f}°")
    print(f"  elbow_flex:    {d_ef:+.2f}°")
    print(f"  wrist_flex:    {d_wf:+.2f}°")

    # Verify FK of solution
    ach_x, ach_z = ik.forward(sol.q1, sol.q2, sol.q3)
    print(f"\n=== FK verification ===")
    print(f"  Achieved tip: x={ach_x:.4f}m  z={ach_z:.4f}m")
    print(f"  Desired tip:  x={desired_x:.4f}m  z={desired_z:.4f}m")
    print(f"  Error:        dx={abs(ach_x-desired_x)*1000:.2f}mm  dz={abs(ach_z-desired_z)*1000:.2f}mm")

# --- Intermediate waypoints (first few trajectory steps) ---
print(f"\n=== First 5 trajectory waypoints (approach from neutral) ===")
from feeding.trajectory import CartesianTrajectory
start = (sc.neutral_pose_xyz[0], sc.neutral_pose_xyz[1], sc.neutral_pose_xyz[2])
end = (target.x, target.y, target.z)
traj = CartesianTrajectory(start, end, t_start=0.0, v_max=sc.max_linear_speed_mps,
                           min_duration=sc.min_trajectory_duration_s)
print(f"  Trajectory duration: {traj.duration:.2f}s")
for i in range(5):
    t = (i + 1) * 0.05
    pos, vel, done = traj.sample(t)
    wp_dx = pos[0] - sc.neutral_pose_xyz[0]
    wp_dz = pos[2] - sc.neutral_pose_xyz[2]
    wp_desired_x = tip_x_n + wp_dx
    wp_desired_z = tip_z_n + wp_dz
    wp_sol = ik.solve(x_tip=wp_desired_x, z_tip=wp_desired_z, phi=phi_ref, seed_q2=q2_n)
    if wp_sol:
        d_sl = to_motor_deg("shoulder_lift", math.degrees(wp_sol.q1)) - neutral_joints["shoulder_lift"]
        d_ef = to_motor_deg("elbow_flex", math.degrees(wp_sol.q2)) - neutral_joints["elbow_flex"]
        d_wf = to_motor_deg("wrist_flex", math.degrees(wp_sol.q3)) - neutral_joints["wrist_flex"]
        print(f"  t={t:.2f}s  cart=({pos[0]:.4f},{pos[2]:.4f})  Δshoulder={d_sl:+.3f}°  Δelbow={d_ef:+.3f}°  Δwrist={d_wf:+.3f}°")
    else:
        print(f"  t={t:.2f}s  cart=({pos[0]:.4f},{pos[2]:.4f})  IK FAILED")
