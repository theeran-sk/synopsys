"""Move each motor individually by +10° from neutral, one at a time.

Run this with the arm at neutral. It will:
1. Print current (neutral) joint positions
2. For each motor: move it +10° from neutral, wait, then return to neutral
3. YOU observe and note which direction the spoon tip moved (up/down/forward/back).

Usage:  python characterize_arm.py
"""

import sys, os, time, logging

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(message)s")

from feeding.config import load_config

cfg = load_config("configs/feeding_default.yaml")
rc = cfg.lerobot

os.environ["LEROBOT_SO_FOLLOWER_MOTORS"] = rc.motors_csv
from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

robot_cfg = SOFollowerRobotConfig(
    port=rc.port,
    id=rc.robot_id,
    max_relative_target=rc.max_relative_target,
    disable_torque_on_disconnect=rc.disable_torque_on_disconnect,
)
robot = make_robot_from_config(robot_cfg)
robot.connect()

joint_order = [m.strip() for m in rc.motors_csv.split(",") if m.strip()]

# Enable torque
for m in joint_order:
    try:
        robot.bus.enable_torque(m, num_retry=2)
    except Exception:
        pass

# Read current position
obs = robot.get_observation()
current = {}
for name in joint_order:
    key = f"{name}.pos"
    if key in obs:
        current[name] = float(obs[key])

print("=== Current joint positions ===")
for name in joint_order:
    print(f"  {name}: {current.get(name, '???'):.2f}°")

neutral = dict(rc.neutral_joints_deg) if rc.neutral_joints_deg else dict(current)
print("\n=== Neutral joints (from config) ===")
for name in joint_order:
    print(f"  {name}: {neutral.get(name, 0):.2f}°")

# Go to neutral first
print("\n>>> Moving to neutral... ", end="", flush=True)
action = {f"{name}.pos": neutral[name] for name in joint_order}
for _ in range(30):
    robot.send_action(action)
    time.sleep(0.05)
print("done.")

DELTA = 10.0  # degrees
HOLD = 2.0    # seconds

test_motors = ["shoulder_lift", "elbow_flex", "wrist_flex"]

for motor in test_motors:
    input(f"\n>>> Press ENTER to move {motor} by +{DELTA}° from neutral...")

    target = dict(neutral)
    target[motor] = neutral[motor] + DELTA

    print(f"    Moving {motor}: {neutral[motor]:.1f}° → {target[motor]:.1f}° (+{DELTA}°)")
    action = {f"{name}.pos": target[name] for name in joint_order}
    t0 = time.monotonic()
    while time.monotonic() - t0 < HOLD:
        robot.send_action(action)
        time.sleep(0.05)

    # Read achieved
    obs = robot.get_observation()
    achieved = float(obs.get(f"{motor}.pos", 0))
    print(f"    Achieved: {achieved:.2f}°")
    print(f"    ==> OBSERVE: Did the spoon tip move UP, DOWN, FORWARD, or BACKWARD?")

    print(f"    Returning to neutral...", end="", flush=True)
    action = {f"{name}.pos": neutral[name] for name in joint_order}
    t0 = time.monotonic()
    while time.monotonic() - t0 < HOLD:
        robot.send_action(action)
        time.sleep(0.05)
    print(" done.")

# Also test shoulder_pan
input(f"\n>>> Press ENTER to move shoulder_pan by +{DELTA}° from neutral...")
target = dict(neutral)
target["shoulder_pan"] = neutral["shoulder_pan"] + DELTA
print(f"    Moving shoulder_pan: {neutral['shoulder_pan']:.1f}° → {target['shoulder_pan']:.1f}° (+{DELTA}°)")
action = {f"{name}.pos": target[name] for name in joint_order}
t0 = time.monotonic()
while time.monotonic() - t0 < HOLD:
    robot.send_action(action)
    time.sleep(0.05)
print(f"    ==> OBSERVE: Did the base rotate LEFT or RIGHT (from arm's perspective)?")
action = {f"{name}.pos": neutral[name] for name in joint_order}
t0 = time.monotonic()
while time.monotonic() - t0 < HOLD:
    robot.send_action(action)
    time.sleep(0.05)
print("    Returned to neutral.")

# Release torque
for m in joint_order:
    try:
        robot.bus.disable_torque(m, num_retry=1)
    except Exception:
        pass
robot.disconnect()

print("\n=== Done! ===")
print("Tell me the direction each motor moved the spoon tip when given +10°:")
print("  shoulder_lift +10° → ?  (up/down/forward/backward)")
print("  elbow_flex +10°    → ?  (up/down/forward/backward)")
print("  wrist_flex +10°    → ?  (up/down/forward/backward)")
print("  shoulder_pan +10°  → ?  (left/right)")
