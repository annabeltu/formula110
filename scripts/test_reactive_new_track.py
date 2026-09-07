"""Audit the reactive controller on a different technical track.

Run headless metrics by default, or pass ``--watch`` for a solo live view.
"""

from __future__ import annotations

import argparse
from functools import partial
from importlib import import_module
from typing import Any, cast

import racing.game.app as game_app
import racing.race.head_to_head as head_to_head
from controllers import reactive
from racing import RobotSensors, load_student_submission
from racing.game.config import GameConfig
from racing.graphics.panda_config import configure_headless_panda
from racing.graphics.track_rendering import add_mugello_short_track, add_racing_scene_collisions
from racing.physics import FORMULA_VEHICLE_PHYSICS_CONFIG, PhysicsScene, create_physics_world, create_robot_vehicle
from racing.race.progress import build_track_progress_model, default_track_progress_model
from racing.race.runtime import (
    RaceCarRuntime,
    lap_progress_tracker_for_spawn_pose,
    race_spawn_poses,
    robot_score_damage,
)
from racing.track.world import TrackPoint, sampled_track_centerline

# Unlike the assignment track, this course alternates tight and open bends and
# includes a short S-section. Coordinates are deliberately track-relative;
# the controller receives no knowledge of this list.
TECHNICAL_TRACK_POINTS = (
    TrackPoint(0.0, -18.0, "start"),
    TrackPoint(13.0, -18.0),
    TrackPoint(22.0, -12.0),
    TrackPoint(23.0, -2.0),
    TrackPoint(16.0, 5.0),
    TrackPoint(5.0, 2.0),
    TrackPoint(-3.0, 8.0),
    TrackPoint(2.0, 17.0),
    TrackPoint(-8.0, 22.0),
    TrackPoint(-20.0, 17.0),
    TrackPoint(-24.0, 6.0),
    TrackPoint(-18.0, -3.0),
    TrackPoint(-8.0, -5.0),
    TrackPoint(-10.0, -13.0),
)


def technical_track_model():
    samples = sampled_track_centerline(TECHNICAL_TRACK_POINTS, samples_per_segment=10)
    return build_track_progress_model(samples)


def install_technical_track() -> None:
    """Route existing race builders to the test model and matching walls."""
    model = technical_track_model()
    track_renderer = partial(add_mugello_short_track, samples=model.points)
    collision_builder = partial(add_racing_scene_collisions, samples=model.points)
    game_app.default_track_progress_model = lambda: model
    game_app.add_mugello_short_track = track_renderer
    game_app.add_racing_scene_collisions = collision_builder
    head_to_head.default_track_progress_model = lambda: model
    head_to_head.add_racing_scene_collisions = collision_builder


def run_metrics(*, track_name: str, seed: int, races: int, round_seconds: float) -> None:
    brake_applications = 0

    def audited_control(sensors: RobotSensors):
        nonlocal brake_applications
        command = reactive.control(sensors)
        speed = sensors.odometry.speed_mps
        if (speed > 0.0 and command.throttle < 0.0) or (speed < 0.0 and command.throttle > 0.0):
            brake_applications += 1
        return command

    configure_headless_panda()
    showbase = cast(Any, import_module("direct.showbase.ShowBase"))
    base = showbase.ShowBase(windowType="none")
    runtimes: list[RaceCarRuntime] = []
    model = head_to_head.default_track_progress_model()
    try:
        for race_index in range(1, races + 1):
            physics_world = create_physics_world()
            physics_scene = PhysicsScene(world=physics_world, vehicles=[])
            root = base.render.attachNewNode(f"solo-reactive-{race_index}")
            head_to_head.add_racing_scene_collisions(physics_world=physics_world, render=root)
            pose = race_spawn_poses(
                1,
                model=model,
                config=FORMULA_VEHICLE_PHYSICS_CONFIG,
                random_seed=seed,
                race_index=race_index,
            )[0]
            robot = create_robot_vehicle(
                world=physics_world,
                render=root,
                name=f"solo-reactive-{race_index}",
                position=pose.position,
                heading_degrees=pose.heading_degrees,
                config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            )
            physics_scene.vehicles.append(robot)
            runtime = RaceCarRuntime(
                robot=robot,
                tracker=lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=pose),
            )
            run_runtime = head_to_head._run_headless_student_runtime_for_duration  # pyright: ignore[reportPrivateUsage]
            run_runtime(
                model=model,
                physics_world=physics_world,
                physics_scene=physics_scene,
                entries=(head_to_head.HeadToHeadRaceEntry(role="challenger", copy_index=0),),
                controllers=(audited_control,),
                runtimes=(runtime,),
                duration_seconds=round_seconds,
                fixed_delta_seconds=1 / 60,
                recovery_config=None,
            )
            runtimes.append(runtime)
            root.removeNode()
    finally:
        base.destroy()

    fastest_lap_times = tuple(
        min(
            crossing_time - (runtime.tracker.lap_times_seconds[index - 1] if index else 0.0)
            for index, crossing_time in enumerate(runtime.tracker.lap_times_seconds)
        )
        for runtime in runtimes
        if runtime.tracker.lap_times_seconds
    )
    print(f"track: {track_name}, controller: controllers.reactive, seed: {seed}, races: {races}")
    print(f"completed laps: {sum(runtime.tracker.lap_count for runtime in runtimes)}")
    print(
        f"mean fastest lap: {sum(fastest_lap_times) / len(fastest_lap_times):.3f} s"
        if fastest_lap_times
        else "mean fastest lap: none"
    )
    print(f"max speed: {max(runtime.max_speed_mps for runtime in runtimes) * 2.23694:.1f} mph")
    print(f"wall contact: {sum(runtime.tracker.wall_contact_seconds for runtime in runtimes):.3f} s")
    print(f"maximum damage: {max(robot_score_damage(runtime.robot) for runtime in runtimes):.4f}")
    print(f"actual brake applications: {brake_applications}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--original", action="store_true", help="use the original assignment track")
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--races", type=int, default=7)
    parser.add_argument("--round-seconds", type=float, default=30.0)
    args = parser.parse_args()
    track_name = "original" if args.original else "technical-test"
    if args.original:
        model = default_track_progress_model()
        game_app.default_track_progress_model = lambda: model
    else:
        install_technical_track()
    if args.watch:
        submission = load_student_submission("controllers.reactive")
        game_app.create_app(GameConfig(student_controller=submission.controller, random_seed=args.seed)).run()
    else:
        run_metrics(track_name=track_name, seed=args.seed, races=args.races, round_seconds=args.round_seconds)


if __name__ == "__main__":
    main()
