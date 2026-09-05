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
    SIDE_WALL_CLEARANCE_M,
    SIDE_WALL_STEER_GAIN,
    STEERING_GAIN,
    THROTTLE_DEADBAND_MPS,
    THROTTLE_GAIN,
    TURN_SLOWDOWN,
    TURN_SPEED_EXPONENT,
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
    signed_speed = sensors.odometry.speed_mps
    speed = max(0.0, signed_speed)
    front = _range(sensors.lidar.front_m)
    front_left = _range(sensors.lidar.front_left_m)
    front_right = _range(sensors.lidar.front_right_m)
    wall_left = _range(sensors.wall_lidar.left_m)
    wall_right = _range(sensors.wall_lidar.right_m)
    wall_front = _range(sensors.wall_lidar.front_m)

    # If track geometry is unavailable, cautiously aim toward the open side.
    if not sensors.camera.visible:
        steer = _clamp((front_left - front_right) / 8.0, -0.65, 0.65)
        throttle = 0.12 if signed_speed >= 0.0 and front > 1.5 else 0.0
        return RobotCommand(throttle=throttle, steer=steer)

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

    # Begin a gentle pass before a slower car becomes an emergency. Camera
    # competitors are filtered to an ahead cone so cars beside or behind us do
    # not create steering noise.
    for competitor in camera.competitors:
        if (
            competitor.distance_m < 8.0
            and abs(competitor.angle_degrees) < 28.0
            and competitor.closing_speed_mps > 0.5
        ):
            pass_strength = (8.0 - competitor.distance_m) / 8.0
            pass_direction = 1.0 if competitor.angle_degrees <= 0.0 else -1.0
            raw_steer += pass_direction * 0.28 * pass_strength
            break

    # Side beams provide a final steering-only guard near a barrier. Keeping
    # this separate from target speed avoids unnecessary braking in turns.
    if wall_left < SIDE_WALL_CLEARANCE_M:
        raw_steer += SIDE_WALL_STEER_GAIN * (1.0 - wall_left / SIDE_WALL_CLEARANCE_M)
    if wall_right < SIDE_WALL_CLEARANCE_M:
        raw_steer -= SIDE_WALL_STEER_GAIN * (1.0 - wall_right / SIDE_WALL_CLEARANCE_M)
    steer = _clamp(raw_steer * STEERING_GAIN, -1.0, 1.0)

    # Slow down as the required turn becomes sharper.
    turn_demand = max(
        abs(steer),
        min(1.0, abs(camera.heading_error_degrees) / 55.0),
        min(1.0, abs(far_offset) / 7.0),
    )

    # Preserve full speed on straights, but shed speed early when even a
    # moderate bend appears. The nonlinear profile is smoother and safer than
    # waiting for steering demand to become extreme before braking.
    straight_fraction = (1.0 - turn_demand) ** TURN_SPEED_EXPONENT
    corner_speed = BASE_SPEED - TURN_SLOWDOWN
    target_speed = corner_speed + TURN_SLOWDOWN * straight_fraction

    severe_turn = abs(camera.heading_error_degrees) > 55.0 or abs(far_offset) > 9.0
    if severe_turn:
        target_speed = min(target_speed, 9.0)

    # Leave room to stop for anything directly ahead.
    if front < FRONT_SLOW_DISTANCE:
        target_speed = min(target_speed, max(0.0, (front - 0.8) * FRONT_SPEED_SCALE))

    # Always calculate throttle.
    speed_error = target_speed - speed
    # Coast near the target instead of alternating between throttle and brake
    # when sensor readings move by a small amount from one tick to the next.
    throttle = (
        0.0
        if speed_error < THROTTLE_DEADBAND_MPS
        else _clamp(
            speed_error * THROTTLE_GAIN,
            0.0,
            MAX_THROTTLE,
        )
    )

    # Positive throttle would count as braking while the car is still rolling
    # backward. Coast until forward motion resumes so no command ever opposes
    # the current direction of travel.
    if signed_speed < 0.0:
        throttle = 0.0

    # On an unfamiliar track, lift early when the wall-only forward beam sees
    # the end of the available straight. This is coasting, not braking.
    coast_distance = 5.0 + speed * 0.6
    if min(front, wall_front) < coast_distance:
        throttle = 0.0

    # This final clamp is a deliberate invariant: future changes to any rule
    # above cannot accidentally introduce a negative-throttle brake command.
    return RobotCommand(throttle=_clamp(throttle, 0.0, MAX_THROTTLE), steer=steer)
