# Feeding Assistant v1

This is a minimal software scaffold for your science fair feeding arm:

- Wait until mouth is open continuously for `open_hold_seconds` (default `0.5s`)
- Start approach toward mouth target
- If mouth closes at any point during approach/hold, immediately retreat to neutral

## Quick start

From `project/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_feeding.py
```

Press `q` in the camera window to stop.

## Backend switching

Default is fake arm backend for testing logic.

```bash
python run_feeding.py --backend fake
python run_feeding.py --backend lerobot
```

`lerobot` backend now supports a custom 4-motor SO-101 follower arm.
Configure port/id/motors in `configs/feeding_default.yaml` under `lerobot`.
For your build, keep:

```yaml
motors_csv: shoulder_pan,shoulder_lift,elbow_flex,wrist_flex
robot_id: synopsys2026
```

The backend sets `LEROBOT_SO_FOLLOWER_MOTORS` automatically at runtime.

## Simulator And Digital Twin

Run from `project/`:

```bash
python run_simulator.py --mode sim
python run_simulator.py --mode live_view
python run_simulator.py --mode command_live
```

`live_view` now supports drag-to-command directly (same as `command_live`).

Controls:

- Left-click + drag in the plot to move target
- `n` reset target to neutral
- `q` quit
- `g` toggle command streaming (in `command_live`)

Each run writes a CSV log in `logs/`.
Generate presentation plots:

```bash
python plot_sim_log.py --csv logs/<your_log_file>.csv
```

## Main files

- `run_feeding.py`: entry point
- `feeding/controller.py`: state machine (`WAIT_OPEN -> APPROACH -> HOLD -> RETREAT`)
- `feeding/perception/mediapipe_mouth.py`: webcam + mouth open detection
- `feeding/arm/fake_arm.py`: simulated arm backend
- `configs/feeding_default.yaml`: tunable settings

## Tuning knobs

- `vision.open_threshold`: mouth open ratio threshold
- `vision.open_hold_seconds`: required continuous open duration before approach
- `control.hold_at_target_seconds`: brief hold before retreat
- `control.rearm_cooldown_seconds`: wait time before next possible approach
- `control.require_close_before_rearm`: require mouth-close event before next cycle
- `safety.*`: speed and workspace limits
- `lerobot.approach_gains_deg`: per-joint response strength (start small)

## Notes for your SO-101 custom build

- Current default gains keep `shoulder_pan` fixed to avoid sudden base rotation while tuning.
- If disconnect throws overload on torque-disable, keep `lerobot.disable_torque_on_disconnect: false`.
