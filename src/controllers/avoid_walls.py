from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Avoid Walls"
RACING_COLOR: str = "#1EB4FF"


def control(sensors: RobotSensors) -> RobotCommand:
    throttle: float = 0.4
    steer: float = 0.0
    return RobotCommand(throttle=throttle, steer=steer)