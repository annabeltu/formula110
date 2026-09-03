"""Autonomous controller that follows the track and avoids obstacles."""

from math import isfinite

from controllers.reactive_params import (
    BASE_SPEED,
    CENTER_WEIGHT,
    FAR_LOOKAHEAD_WEIGHT,
    FRONT_SLOW_DISTANCE,
    FRONT_SPEED_SCALE,
    HEADING_DIVISOR,
    MAX_THROTTLE,
    RECOVERY_MAX_SPEED_MPS,
    STEERING_GAIN,
    THROTTLE_DEADBAND_MPS,
    THROTTLE_GAIN,
    TURN_SLOWDOWN,
)
from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Reactive"
RACING_COLOR: str = "#C8249C"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _range(distance_m: float, fallback_m: float = 50.0) -> float:
    """Turn a no-hit (infinite) LiDAR reading into a useful finite range."""
    return distance_m if isfinite(distance_m) else fallback_m


def control(sensors: RobotSensors) -> RobotCommand:
    """Choose steering and throttle from the current sensor snapshot."""
    speed = max(0.0, sensors.odometry.speed_mps)
    front = _range(sensors.lidar.front_m)
    front_left = _range(sensors.lidar.front_left_m)
    front_right = _range(sensors.lidar.front_right_m)

    # If track geometry is unavailable, cautiously aim toward the open side.
    if not sensors.camera.visible:
        steer = _clamp((front_left - front_right) / 8.0, -0.65, 0.65)
        return RobotCommand(throttle=0.12 if front > 1.5 else -0.25, steer=steer)

    camera = sensors.camera
    offsets = camera.lookahead_offsets_m
    near_offset = offsets[0] if offsets else camera.center_offset_m
    far_offset = offsets[-1] if offsets else near_offset

    # Follow the center line and use the far point to begin turns early.
    raw_steer = (
        camera.heading_error_degrees / HEADING_DIVISOR
        + camera.center_offset_m * CENTER_WEIGHT
        + near_offset * 0.055
        + far_offset * FAR_LOOKAHEAD_WEIGHT
    )

    # Add a last-moment correction away from a close wall, car, or blocker.
    obstacle_distance = min(front_left, front_right)
    if obstacle_distance < 4.0:
        avoidance_strength = (4.0 - obstacle_distance) / 4.0
        open_side = -1.0 if front_left > front_right else 1.0
        raw_steer += open_side * 0.45 * avoidance_strength
    steer = _clamp(raw_steer * STEERING_GAIN, -1.0, 1.0)

    # Slow down as the required turn becomes sharper.
    turn_demand = max(
        abs(steer),
        min(1.0, abs(camera.heading_error_degrees) / 55.0),
        min(1.0, abs(far_offset) / 7.0),
    )

    target_speed = BASE_SPEED - TURN_SLOWDOWN * turn_demand

    # Leave room to stop for anything directly ahead.
    if front < FRONT_SLOW_DISTANCE:
        target_speed = min(
            target_speed,
            max(0.0, (front - 0.8) * FRONT_SPEED_SCALE)
        )

    # Always calculate throttle.
    speed_error = target_speed - speed
    # Coast near the target instead of alternating between throttle and brake
    # when sensor readings move by a small amount from one tick to the next.
    throttle = 0.0 if abs(speed_error) < THROTTLE_DEADBAND_MPS else _clamp(
        speed_error * THROTTLE_GAIN,
        -0.65,
        MAX_THROTTLE,
    )

    # Reverse only when sustained contact has actually left the car stuck.
    # At racing speed, contact is usually a brief side-by-side car collision;
    # reversing then feels like a random brake and creates a larger crash.
    if sensors.contact.any_contact > 0.25 and speed < RECOVERY_MAX_SPEED_MPS:
        open_side = -0.7 if front_left > front_right else 0.7
        return RobotCommand(throttle=-0.35, steer=open_side)

    return RobotCommand(throttle=throttle, steer=steer)
