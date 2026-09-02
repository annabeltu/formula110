"""Behavior-cloned controller trained from the bundled baseline expert."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path

from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Baseline Imitation"
RACING_COLOR: str = "#F2C14E"

_MODEL_PATH = Path(__file__).with_name("imitation_model.json")


def _range(value: float) -> float:
    return min(50.0, max(0.0, value)) if isfinite(value) else 50.0


def features(sensors: RobotSensors) -> tuple[float, ...]:
    """Convert public sensors into the fixed feature vector used in training."""
    offsets = sensors.camera.lookahead_offsets_m
    near = offsets[0] if offsets else sensors.camera.center_offset_m
    far = offsets[-1] if offsets else near
    front = _range(sensors.lidar.front_m)
    left = _range(sensors.lidar.front_left_m)
    right = _range(sensors.lidar.front_right_m)
    heading = sensors.camera.heading_error_degrees
    center = sensors.camera.center_offset_m
    # Nonlinear basis terms let a small linear policy learn braking and turn
    # slowdown while keeping inference transparent and dependency-free.
    return (
        1.0,
        sensors.odometry.speed_mps / 10.0,
        heading / 60.0,
        center / 8.0,
        near / 8.0,
        far / 8.0,
        front / 50.0,
        left / 50.0,
        right / 50.0,
        min(front, 7.0) / 7.0,
        min(left, right, 4.0) / 4.0,
        abs(heading) / 60.0,
        abs(far) / 8.0,
        float(sensors.camera.visible),
        min(sensors.contact.any_contact, 1.0),
        (left - right) / 50.0,
    )


class ImitationController:
    """CPU-only linear behavior-cloning policy loaded once per race car."""

    def __init__(self) -> None:
        model = json.loads(_MODEL_PATH.read_text(encoding="utf-8"))
        self._throttle = tuple(float(value) for value in model["throttle_weights"])
        self._steer = tuple(float(value) for value in model["steer_weights"])

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        inputs = features(sensors)
        throttle = sum(
            weight * value for weight, value in zip(self._throttle, inputs, strict=True)
        )
        steer = sum(
            weight * value for weight, value in zip(self._steer, inputs, strict=True)
        )
        return RobotCommand(
            throttle=max(-1.0, min(1.0, throttle)),
            steer=max(-1.0, min(1.0, steer)),
        )


def create_controller() -> ImitationController:
    return ImitationController()


_default_controller: ImitationController | None = None


def control(sensors: RobotSensors) -> RobotCommand:
    """Compatibility entry point for callers that do not use the factory."""
    global _default_controller
    if _default_controller is None:
        _default_controller = create_controller()
    return _default_controller(sensors)
