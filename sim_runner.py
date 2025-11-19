import queue
import time
from multiprocessing import Process, Queue
from typing import Dict

import cv2
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation


CAMERAS = ["ee_camera", "fb_camera"]
FLOATING_BASE_BODY = "floating_base_link"


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _update_camera_views(renderers: Dict[str, mujoco.Renderer], data: mujoco.MjData) -> Dict[str, np.ndarray]:
    rgb_frames: Dict[str, np.ndarray] = {}
    for name in CAMERAS:
        renderer = renderers[name]
        renderer.update_scene(data, camera=name)
        rgb_frame = renderer.render()
        rgb_frames[name] = rgb_frame.copy()
        cv2.imshow(name, rgb_to_bgr(rgb_frame))
    cv2.waitKey(1)
    return rgb_frames


class FloatingBaseController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, body_name: str):
        self.model = model
        self.data = data
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in the model.")
        self._initial_pos = np.array(model.body_pos[self.body_id], dtype=np.float64)
        mj_quat = np.array(model.body_quat[self.body_id], dtype=np.float64)
        self._initial_quat = mj_quat.copy()
        self._rot = Rotation.from_quat(mj_quat[[1, 2, 3, 0]])
        self._linear_velocity = np.zeros(3, dtype=np.float64)
        self._yaw_rate = 0.0
        self._last_update_time = float(self.data.time)

    def command_velocity(self, vx: float, vy: float, yaw_rate: float) -> None:
        self._linear_velocity[0] = float(vx)
        self._linear_velocity[1] = float(vy)
        self._yaw_rate = float(yaw_rate)

    def stop(self) -> None:
        self.command_velocity(0.0, 0.0, 0.0)

    def update_motion(self) -> None:
        current_time = float(self.data.time)
        dt = current_time - self._last_update_time
        if dt <= 0.0:
            return
        world_velocity = self._rot.apply(self._linear_velocity)
        self.model.body_pos[self.body_id] = self.model.body_pos[self.body_id] + world_velocity * dt
        if self._yaw_rate != 0.0:
            delta_rot = Rotation.from_euler('z', self._yaw_rate * dt)
            self._rot = delta_rot * self._rot
            quat_xyzw = self._rot.as_quat()
            self.model.body_quat[self.body_id] = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                dtype=np.float64,
            )
        self._last_update_time = current_time
        mujoco.mj_forward(self.model, self.data)

class SimulatorClient:
    def __init__(self, command_queue: Queue, image_queue: Queue):
        self._command_queue = command_queue
        self._image_queue = image_queue

    def set_velocity(self, vx: float, vy: float, yaw_rate: float) -> None:
        self._command_queue.put((
            float(vx),
            float(vy),
            float(yaw_rate),
        ))

    def get_rgb_frames(self, timeout: float | None = None) -> Dict[str, np.ndarray] | None:
        try:
            return self._image_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class SimulatorProcess(Process):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._cmd_queue: Queue = Queue()
        self._image_queue: Queue = Queue(maxsize=1)
        self.dt: float | None = None
        self.client = SimulatorClient(self._cmd_queue, self._image_queue)

    def run(self) -> None:
        model = mujoco.MjModel.from_xml_path("asset/scene.xml")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        dt = model.opt.timestep

        for camera_name in CAMERAS:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
                raise ValueError(f"Camera '{camera_name}' not found in the model.")

        base_control = FloatingBaseController(model, data, FLOATING_BASE_BODY)
        renderers = {name: mujoco.Renderer(model) for name in CAMERAS}
        for name in CAMERAS:
            renderer = renderers[name]
            renderer.update_scene(data, camera=name)
            frame = rgb_to_bgr(renderer.render())
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(name, frame)
        cv2.waitKey(1)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                loop_start = time.perf_counter()
                try:
                    while True:
                        cmd = self._cmd_queue.get_nowait()
                        vx, vy, yaw_rate = cmd
                        base_control.command_velocity(vx, vy, yaw_rate)
                except queue.Empty:
                    pass

                mujoco.mj_step(model, data)
                base_control.update_motion()
                frames = _update_camera_views(renderers, data)
                self._publish_frames(frames)
                viewer.sync()
                elapsed = time.perf_counter() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        for name in CAMERAS:
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(name)
        cv2.waitKey(1)

    def start(self) -> None:
        if self.is_alive():
            return
        temp_model = mujoco.MjModel.from_xml_path("asset/scene.xml")
        self.dt = float(temp_model.opt.timestep)
        super().start()

    def wait_for_stop(self) -> None:
        if not self.is_alive():
            return
        super().join()
        self.dt = None

    def stop(self) -> None:
        if not self.is_alive():
            return
        self.terminate()
        super().join()
        self.dt = None

    def _publish_frames(self, frames: Dict[str, np.ndarray]) -> None:
        # keep only the most recent frame set
        try:
            while True:
                self._image_queue.get_nowait()
        except queue.Empty:
            pass
        self._image_queue.put(frames)
