# Feeding Assistant (Clean Base)

This folder now contains only the core code for:
- mouth detection
- feeding runtime control
- robot/fake arm backends
- distance calibration utility

## Run Feeding

From `project/`:

```bash
pip install -r requirements.txt
python run_feeding.py --backend lerobot
```

For camera-only testing:

```bash
python run_feeding.py --backend fake
```

Press `q` in the camera window to quit.

## Calibrate Distance

Use this when you want a fresh landmark-to-distance fit:

```bash
python calibrate_mouth_distance.py --distances-cm 0,5,10,15,20 --feature eye --output configs/spoon_mouth_calibration.yaml
```

## Main Files

- `run_feeding.py`: app entry point
- `feeding/controller.py`: state machine
- `feeding/perception/mediapipe_mouth.py`: mouth detection
- `feeding/arm/lerobot_arm.py`: real hardware backend
- `feeding/arm/fake_arm.py`: fake backend
- `configs/feeding_default.yaml`: runtime config
- `calibrate_mouth_distance.py`: distance calibration tool
