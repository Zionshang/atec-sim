import time

import cv2
import numpy as np
import mujoco
import mujoco.viewer

from robot_control import RobotControl


CAMERA_NAME = "ee_camera"
WINDOW_TITLE = "End-effector Camera"
FLOATING_BASE_BODY = "floating_base_link"


def rgb_to_bgr(image, cv2_module):
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255).astype(np.uint8)
    return cv2_module.cvtColor(image, cv2_module.COLOR_RGB2BGR)


def simulate():
    cv2_module = cv2

    model = mujoco.MjModel.from_xml_path('asset/scene.xml')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if camera_id < 0:
        raise ValueError(f"Camera '{CAMERA_NAME}' not found in the model.")

    base_control = RobotControl(model, data, FLOATING_BASE_BODY)
    key_mapping = {
        ord('w'): np.array([0.05, 0.0, 0.0]),
        ord('s'): np.array([-0.05, 0.0, 0.0]),
        ord('a'): np.array([0.0, 0.05, 0.0]),
        ord('d'): np.array([0.0, -0.05, 0.0]),
        ord('r'): np.array([0.0, 0.0, 0.05]),
        ord('f'): np.array([0.0, 0.0, -0.05]),
    }
    yaw_map = {
        ord('q'): 0.1,
        ord('e'): -0.1,
    }

    renderer = mujoco.Renderer(model)
    renderer.update_scene(data, camera=CAMERA_NAME)

    initial_image = renderer.render()
    bgr_frame = rgb_to_bgr(initial_image, cv2_module)

    cv2_module.namedWindow(WINDOW_TITLE, cv2_module.WINDOW_AUTOSIZE)
    cv2_module.imshow(WINDOW_TITLE, bgr_frame)
    cv2_module.waitKey(1)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            loop_start = time.perf_counter()
 
            mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=CAMERA_NAME)
            camera_frame = renderer.render()
            bgr_frame = rgb_to_bgr(camera_frame, cv2_module)
            cv2_module.imshow(WINDOW_TITLE, bgr_frame)
            key = cv2_module.waitKey(1) & 0xFF
            if key == ord('0'):
                base_control.reset_base_pose()
            elif key in key_mapping:
                base_control.move_relative(key_mapping[key])
            elif key in yaw_map:
                base_control.yaw_relative(yaw_map[key])
            if cv2_module.getWindowProperty(WINDOW_TITLE, cv2_module.WND_PROP_VISIBLE) < 1:
                break

            viewer.sync()

            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    cv2_module.destroyWindow(WINDOW_TITLE)
    cv2_module.waitKey(1)
 
if __name__ == "__main__":
    simulate()