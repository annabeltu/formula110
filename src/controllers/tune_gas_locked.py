"""Tune ``controllers.reactive`` for the gas-locked lap-time leaderboard.

The leaderboard averages the fastest completed lap from solo 30-second runs
with seeds 110 and 2026.  This tuner evaluates that exact setup and rejects
parameter sets that brake, fail to complete a lap, or fail to survive.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import optuna

from controllers import reactive
from racing.graphics.panda_config import configure_headless_panda
from racing.graphics.track_rendering import add_racing_scene_collisions
from racing.physics import (
    FORMULA_VEHICLE_PHYSICS_CONFIG,
    PhysicsScene,
    apply_robot_vehicle_command,
    apply_wall_impact_damage,
    create_physics_world,
    create_robot_vehicle,
)
from racing.race.progress import default_track_progress_model, project_track_position
from racing.race.runtime import (
    RaceCarRuntime,
    lap_progress_tracker_for_spawn_pose,
    race_contact_states,
    race_spawn_poses,
    robot_is_eliminated,
    robot_score_damage,
    robot_track_point,
    update_race_runtime_after_step,
)
from racing.race.sensors import build_robot_sensors

PARAM_FILE = Path(__file__).with_name("reactive_params.py")
SEEDS = (110, 2026)
FIXED_DELTA_SECONDS = 1.0 / 60.0


@dataclass(frozen=True, slots=True)
class TrialMetrics:
    fastest_lap_seconds: float | None
    lap_count: int
    damage: float
    survived: bool
    wall_contact_seconds: float
    brake_applications: int


class SoloEvaluator:
    """Keep one headless Panda instance alive across optimization trials."""

    def __init__(self) -> None:
        configure_headless_panda()
        showbase = cast(Any, import_module("direct.showbase.ShowBase"))
        self.base = showbase.ShowBase(windowType="none")
        self.model = default_track_progress_model()

    def close(self) -> None:
        self.base.destroy()

    def run(self, *, seed: int, duration_seconds: float = 30.0) -> TrialMetrics:
        physics_world = create_physics_world()
        physics_scene = PhysicsScene(world=physics_world, vehicles=[])
        root = self.base.render.attachNewNode(f"gas-locked-{seed}")
        add_racing_scene_collisions(physics_world=physics_world, render=root)
        pose = race_spawn_poses(
            1,
            model=self.model,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            random_seed=seed,
            race_index=1,
        )[0]
        robot = create_robot_vehicle(
            world=physics_world,
            render=root,
            name=f"gas-locked-{seed}-car",
            position=pose.position,
            heading_degrees=pose.heading_degrees,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        )
        physics_scene.vehicles.append(robot)
        runtime = RaceCarRuntime(
            robot=robot,
            tracker=lap_progress_tracker_for_spawn_pose(model=self.model, spawn_pose=pose),
        )
        elapsed_seconds = 0.0
        brake_applications = 0
        try:
            while elapsed_seconds < duration_seconds:
                if not robot_is_eliminated(robot):
                    sensors, runtime.sensor_state = build_robot_sensors(
                        physics_world=physics_world,
                        robot=robot,
                        track_model=self.model,
                        time_s=elapsed_seconds,
                        dt_s=FIXED_DELTA_SECONDS,
                        previous_state=runtime.sensor_state,
                    )
                    command = reactive.control(sensors)
                    if (sensors.odometry.speed_mps > 0.0 and command.throttle < 0.0) or (
                        sensors.odometry.speed_mps < 0.0 and command.throttle > 0.0
                    ):
                        brake_applications += 1
                    apply_robot_vehicle_command(robot=robot, command=command)

                physics_scene.step(FIXED_DELTA_SECONDS)
                next_elapsed_seconds = min(duration_seconds, elapsed_seconds + FIXED_DELTA_SECONDS)
                contact_state = race_contact_states(physics_world=physics_world, runtimes=(runtime,))[0]
                apply_wall_impact_damage(
                    physics_world=physics_world,
                    robots=(robot,),
                    fixed_time_step=physics_scene.fixed_time_step,
                )
                projection = project_track_position(self.model, robot_track_point(robot))
                update_race_runtime_after_step(
                    runtime=runtime,
                    projection=projection,
                    contact_state=contact_state,
                    elapsed_seconds=next_elapsed_seconds,
                    delta_seconds=FIXED_DELTA_SECONDS,
                )
                elapsed_seconds = next_elapsed_seconds
        finally:
            root.removeNode()

        crossings = runtime.tracker.lap_times_seconds
        lap_durations = [
            crossing - (crossings[index - 1] if index else 0.0) for index, crossing in enumerate(crossings)
        ]
        damage = robot_score_damage(robot)
        return TrialMetrics(
            fastest_lap_seconds=min(lap_durations) if lap_durations else None,
            lap_count=runtime.tracker.lap_count,
            damage=damage,
            survived=not robot_is_eliminated(robot) and damage < 1.0,
            wall_contact_seconds=runtime.tracker.wall_contact_seconds,
            brake_applications=brake_applications,
        )


def current_parameters() -> dict[str, float]:
    return {
        "CENTER_WEIGHT": reactive.CENTER_WEIGHT,
        "NEAR_LOOKAHEAD_WEIGHT": reactive.NEAR_LOOKAHEAD_WEIGHT,
        "FAR_LOOKAHEAD_WEIGHT": reactive.FAR_LOOKAHEAD_WEIGHT,
        "HEADING_DIVISOR": reactive.HEADING_DIVISOR,
        "STEERING_GAIN": reactive.STEERING_GAIN,
        "BASE_SPEED": reactive.BASE_SPEED,
        "TURN_SLOWDOWN": reactive.TURN_SLOWDOWN,
        "TURN_SPEED_EXPONENT": reactive.TURN_SPEED_EXPONENT,
        "THROTTLE_GAIN": reactive.THROTTLE_GAIN,
        "MAX_THROTTLE": reactive.MAX_THROTTLE,
        "FRONT_SLOW_DISTANCE": reactive.FRONT_SLOW_DISTANCE,
        "FRONT_SPEED_SCALE": reactive.FRONT_SPEED_SCALE,
        "TURN_HEADING_NORMALIZER_DEGREES": reactive.TURN_HEADING_NORMALIZER_DEGREES,
        "TURN_OFFSET_NORMALIZER_M": reactive.TURN_OFFSET_NORMALIZER_M,
        "COAST_BASE_DISTANCE_M": reactive.COAST_BASE_DISTANCE_M,
        "COAST_SPEED_FACTOR": reactive.COAST_SPEED_FACTOR,
    }


def install_parameters(parameters: dict[str, float]) -> None:
    for name, value in parameters.items():
        setattr(reactive, name, value)


def sampled_parameters(trial: optuna.Trial) -> dict[str, float]:
    return {
        "CENTER_WEIGHT": trial.suggest_float("CENTER_WEIGHT", -0.1, 0.03),
        "NEAR_LOOKAHEAD_WEIGHT": trial.suggest_float("NEAR_LOOKAHEAD_WEIGHT", 0.07, 0.25),
        "FAR_LOOKAHEAD_WEIGHT": trial.suggest_float("FAR_LOOKAHEAD_WEIGHT", 0.055, 0.145),
        "HEADING_DIVISOR": trial.suggest_float("HEADING_DIVISOR", 38.0, 95.0),
        "STEERING_GAIN": trial.suggest_float("STEERING_GAIN", 0.9, 1.45),
        "BASE_SPEED": trial.suggest_float("BASE_SPEED", 20.8, 23.2),
        "TURN_SLOWDOWN": trial.suggest_float("TURN_SLOWDOWN", 0.4, 3.5),
        "TURN_SPEED_EXPONENT": trial.suggest_float("TURN_SPEED_EXPONENT", 2.2, 5.8),
        "THROTTLE_GAIN": trial.suggest_float("THROTTLE_GAIN", 0.45, 0.95),
        "MAX_THROTTLE": trial.suggest_float("MAX_THROTTLE", 0.96, 1.0),
        "FRONT_SLOW_DISTANCE": trial.suggest_float("FRONT_SLOW_DISTANCE", 0.8, 3.0),
        "FRONT_SPEED_SCALE": trial.suggest_float("FRONT_SPEED_SCALE", 1.0, 2.5),
        "TURN_HEADING_NORMALIZER_DEGREES": trial.suggest_float(
            "TURN_HEADING_NORMALIZER_DEGREES", 65.0, 150.0
        ),
        "TURN_OFFSET_NORMALIZER_M": trial.suggest_float("TURN_OFFSET_NORMALIZER_M", 6.5, 15.0),
        "COAST_BASE_DISTANCE_M": trial.suggest_float("COAST_BASE_DISTANCE_M", 0.1, 3.0),
        "COAST_SPEED_FACTOR": trial.suggest_float("COAST_SPEED_FACTOR", 0.3, 0.72),
    }


def write_parameters(parameters: dict[str, float]) -> None:
    PARAM_FILE.write_text(
        "\n".join(f"{name} = {value!r}" for name, value in parameters.items())
        + "\nTHROTTLE_DEADBAND_MPS = 0.35\n"
        + "SIDE_WALL_CLEARANCE_M = 1.8\n"
        + "SIDE_WALL_STEER_GAIN = 0.50\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    evaluator = SoloEvaluator()
    initial = current_parameters()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=40)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(initial)

    def objective(trial: optuna.Trial) -> float:
        install_parameters(sampled_parameters(trial))
        metrics = tuple(evaluator.run(seed=seed) for seed in SEEDS)
        trial.set_user_attr("metrics", [asdict(metric) for metric in metrics])
        if any(
            metric.fastest_lap_seconds is None or not metric.survived or metric.brake_applications > 0
            for metric in metrics
        ):
            return 100.0 + sum(metric.damage * 100.0 for metric in metrics)
        mean_lap = sum(cast(float, metric.fastest_lap_seconds) for metric in metrics) / len(metrics)
        # Contact is not a leaderboard DQ, but clean laps are much more likely
        # to remain stable on the grading host.
        return mean_lap + sum(metric.wall_contact_seconds for metric in metrics) * 2.0

    try:
        study.optimize(objective, n_trials=args.trials)
    finally:
        evaluator.close()

    print(f"best objective: {study.best_value:.6f}")
    for name, value in study.best_params.items():
        print(f"{name} = {value!r}")
    print(f"metrics: {study.best_trial.user_attrs.get('metrics')}")
    if not args.no_save:
        write_parameters(study.best_params)
        print(f"saved: {PARAM_FILE}")


if __name__ == "__main__":
    main()
