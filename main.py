import time

import cv2
import numpy as np
import mujoco
import mujoco.viewer

from robot_control import RobotControl


CAMERAS = [("ee_camera", "End-effector Camera"), ("fb_camera", "Base Camera")]
FLOATING_BASE_BODY = "floating_base_link"


def rgb_to_bgr(image):
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def simulate():
    model = mujoco.MjModel.from_xml_path("asset/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep

    for camera_name, _ in CAMERAS:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
            raise ValueError(f"Camera '{camera_name}' not found in the model.")

    base_control = RobotControl(model, data, FLOATING_BASE_BODY)
    x_velocity_map = {
        ord("w"): 0.05,
        ord("s"): -0.05,
    }
    y_velocity_map = {
        ord("a"): 0.05,
        ord("d"): -0.05,
    }
    yaw_map = {
        ord("q"): 0.1,
        ord("e"): -0.1,
    }

    renderers = {name: mujoco.Renderer(model) for name, _ in CAMERAS}
    for name, title in CAMERAS:
        renderer = renderers[name]
        renderer.update_scene(data, camera=name)
        frame = rgb_to_bgr(renderer.render())
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(title, frame)
    cv2.waitKey(1)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            loop_start = time.perf_counter()

            mujoco.mj_step(model, data)
            base_control.update_motion()

            for name, title in CAMERAS:
                renderer = renderers[name]
                renderer.update_scene(data, camera=name)
                camera_frame = renderer.render()
                bgr_frame = rgb_to_bgr(camera_frame)
                cv2.imshow(title, bgr_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("0"):
                base_control.reset_base_pose()
            elif key == ord(" "):
                base_control.set_x_velocity(0.0)
                base_control.set_y_velocity(0.0)
                base_control.set_yaw_velocity(0.0)
            elif key in x_velocity_map:
                base_control.set_x_velocity(x_velocity_map[key])
            elif key in y_velocity_map:
                base_control.set_y_velocity(y_velocity_map[key])
            elif key in yaw_map:
                base_control.set_yaw_velocity(yaw_map[key])
            if all(cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1 for _, title in CAMERAS):
                break

            viewer.sync()

            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    for _, title in CAMERAS:
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(title)
    cv2.waitKey(1)


if __name__ == "__main__":
    simulate()
