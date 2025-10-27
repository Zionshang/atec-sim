# ATEC Simulation Quickstart

## Setup
```bash
conda env update -f environment.yml
conda activate atec-sim
```

## Run the Demo
```bash
python main.py
```
This launches two windows:
1. The standard MuJoCo viewer for the full scene.
2. An OpenCV window titled **End-effector Camera** showing the wrist-mounted camera feed.

## Keyboard Controls (focus the camera window first)
- `w / s`: Translate forward / backward in the robot's local frame.
- `a / d`: Translate left / right.
- `r / f`: Translate up / down.
- `j / l`: Yaw rotation (left / right) of the floating base.
- `0`: Reset the floating base pose to its initial state.
- `q`: Quit the simulation loop.

The `robot_control.py` helper applies these moves by directly modifying the floating base ("god mode"), which is why no MuJoCo actuator is required.
