from actions import ActionContext, ActionExecutor
from sim_runner import SimulatorProcess


def main():
    simulator = SimulatorProcess()
    simulator.start()
    if simulator.dt is None:
        raise RuntimeError("Simulator failed to report timestep.")

    ctx = ActionContext(simulator.client, simulator.dt)
    executor = ActionExecutor(ctx)

    print("Running action 'set_position'")
    print(executor.set_position({"delta_x": 3, "delta_y": 4, "delta_yaw": 0}))
    print("Running action 'set_velocity'")
    print(executor.set_velocity({"vx": 0.1, "vy": -0.05, "yaw_rate": -0.1, "duration": 2.5}))

    print("All actions executed. Close the viewer window to exit.")
    simulator.wait_for_stop()


if __name__ == "__main__":
    main()
