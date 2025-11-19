from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

from sim_runner import SimulatorClient


DEFAULT_LINEAR_SPEED = 0.5  # m/s along each planar axis
DEFAULT_YAW_RATE = 0.5  # rad/s


@dataclass
class ActionContext:
    simulator: SimulatorClient
    dt: float


def _integrate_for_duration(ctx: ActionContext, duration: float) -> None:
    if duration <= 0:
        return
    end_time = time.perf_counter() + duration
    while True:
        remaining = end_time - time.perf_counter()
        if remaining <= 0:
            break
        time.sleep(min(ctx.dt, remaining))


def _position_step_handler(ctx: ActionContext, params: Dict[str, Any]) -> Dict[str, Any]:
    dx = float(params.get("delta_x", 0.0))
    dy = float(params.get("delta_y", 0.0))
    delta_yaw = float(params.get("delta_yaw", 0.0))

    planar_displacement = max(abs(dx), abs(dy))
    linear_duration = planar_displacement / DEFAULT_LINEAR_SPEED if planar_displacement > 0 else 0.0
    angular_duration = abs(delta_yaw) / DEFAULT_YAW_RATE if delta_yaw != 0.0 else 0.0
    duration = max(linear_duration, angular_duration, ctx.dt)

    if duration <= 0.0:
        return {"status": "ok", "note": "No movement requested."}

    ctx.simulator.set_velocity(dx / duration, dy / duration, delta_yaw / duration)
    _integrate_for_duration(ctx, duration)
    ctx.simulator.set_velocity(0.0, 0.0, 0.0)

    return {
        "status": "ok",
        "note": "Reached target offset via time integration.",
    }


def _velocity_command_handler(ctx: ActionContext, params: Dict[str, Any]) -> Dict[str, Any]:
    vx = float(params.get("vx", 0.0))
    vy = float(params.get("vy", 0.0))
    yaw_rate = float(params.get("yaw_rate", 0.0))
    ctx.simulator.set_velocity(vx, vy, yaw_rate)

    duration = float(params.get("duration", 0.0))
    if duration > 0:
        _integrate_for_duration(ctx, duration)
        ctx.simulator.set_velocity(0.0, 0.0, 0.0)
    return {"status": "ok"}


class ActionExecutor:
    def __init__(self, ctx: ActionContext):
        self.ctx = ctx

    def set_position(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return _position_step_handler(self.ctx, params)

    def set_velocity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return _velocity_command_handler(self.ctx, params)
