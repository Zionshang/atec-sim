import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


class RobotControl:
    """Utility helpers to directly reposition a floating-base robot."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, body_name: str = "floating_base_link"):
        self.model = model
        self.data = data
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in the model.")
        self._initial_pos = np.array(model.body_pos[self.body_id], dtype=np.float64)
        mj_quat = np.array(model.body_quat[self.body_id], dtype=np.float64)
        self._initial_quat = mj_quat.copy()
        self._rot = Rotation.from_quat(mj_quat[[1, 2, 3, 0]])

    def move_relative(self, delta_pos: np.ndarray) -> None:
        local_step = np.asarray(delta_pos, dtype=np.float64)
        world_step = self._rot.apply(local_step)
        self.model.body_pos[self.body_id] = self.model.body_pos[self.body_id] + world_step
        mujoco.mj_forward(self.model, self.data)

    def reset_base_pose(self) -> None:
        self.model.body_pos[self.body_id] = self._initial_pos
        self.model.body_quat[self.body_id] = self._initial_quat
        self._rot = Rotation.from_quat(self._initial_quat[[1, 2, 3, 0]])
        mujoco.mj_forward(self.model, self.data)

    def yaw_relative(self, delta_yaw: float) -> None:
        self._rot = Rotation.from_euler('z', delta_yaw) * self._rot
        quat_xyzw = self._rot.as_quat()
        self.model.body_quat[self.body_id] = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )
        mujoco.mj_forward(self.model, self.data)
