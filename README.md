# ATEC Simulation

## Setup
```
conda env update -f environment.yml
conda activate atec-sim
```

## Run
```
python main.py
```
This starts the MuJoCo viewer and two OpenCV camera windows (end-effector + base). After the scripted actions run, close the viewer to stop the simulator.

### Call Actions
Inside `main.py` the simulator runs in a separate process. Create an `ActionExecutor` with the provided context and call:
```python
executor.set_position({"delta_x": 0.3, "delta_y": 0.0, "delta_yaw": 0.2})
executor.set_velocity({"vx": 0.1, "vy": -0.05, "yaw_rate": -0.1, "duration": 1.5})
```
Velocities are expressed in the local frame
