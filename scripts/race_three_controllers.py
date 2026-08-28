"""Run baseline, reactive, and imitation together in one headless race."""

from __future__ import annotations

import argparse
from importlib import import_module
from typing import Any, cast

from controllers import baseline, imitation, reactive
from racing.graphics.panda_config import configure_headless_panda
from racing.graphics.track_rendering import add_racing_scene_collisions
from racing.physics import FORMULA_VEHICLE_PHYSICS_CONFIG, PhysicsScene, create_physics_world, create_robot_vehicle
from racing.race.head_to_head import _run_headless_student_runtime_for_duration
from racing.race.progress import default_track_progress_model
from racing.race.runtime import (
    RaceCarRuntime,
    RaceRecoveryConfig,
    lap_progress_tracker_for_spawn_pose,
    race_scored_distance_m,
    race_spawn_poses,
)
from racing.student.api import RobotController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--races", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=110)
    args = parser.parse_args()
    if args.races < 1 or args.seconds <= 0.0:
        parser.error("--races and --seconds must be positive")

    names = ("Baseline", "Reactive", "Imitation")
    prototypes: tuple[RobotController, ...] = (baseline.control, reactive.control, imitation.create_controller())
    totals = [0.0, 0.0, 0.0]
    configure_headless_panda()
    showbase = cast(Any, import_module("direct.showbase.ShowBase"))
    base = showbase.ShowBase(windowType="none")
    try:
        for race_index in range(1, args.races + 1):
            model = default_track_progress_model()
            world = create_physics_world()
            scene = PhysicsScene(world=world, vehicles=[])
            root = base.render.attachNewNode(f"three-way-{race_index}")
            add_racing_scene_collisions(physics_world=world, render=root)
            poses = race_spawn_poses(3, model=model, random_seed=args.seed, race_index=race_index)
            runtimes: list[RaceCarRuntime] = []
            try:
                for index, pose in enumerate(poses):
                    robot = create_robot_vehicle(
                        world=world,
                        render=root,
                        name=f"three-way-{race_index}-{index}",
                        position=pose.position,
                        heading_degrees=pose.heading_degrees,
                        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
                    )
                    scene.vehicles.append(robot)
                    runtimes.append(
                        RaceCarRuntime(
                            robot=robot,
                            tracker=lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=pose),
                        )
                    )
                # The shared race runtime is controller-agnostic. Entry values
                # are used only by an optional sampling callback, so labels are
                # sufficient for this three-way evaluation.
                _run_headless_student_runtime_for_duration(
                    model=model,
                    physics_world=world,
                    physics_scene=scene,
                    entries=cast(Any, names),
                    controllers=prototypes,
                    runtimes=tuple(runtimes),
                    duration_seconds=args.seconds,
                    fixed_delta_seconds=1 / 60,
                    recovery_config=RaceRecoveryConfig(stuck_seconds=1.5, distance_penalty_m=5.0, cooldown_seconds=2.0),
                )
                distances = tuple(race_scored_distance_m(runtime) for runtime in runtimes)
                for index, distance in enumerate(distances):
                    totals[index] += distance
                result = " | ".join(
                    f"{name}: {distance:.1f} m" for name, distance in zip(names, distances, strict=True)
                )
                print(f"Race {race_index}: {result}")
            finally:
                root.removeNode()
    finally:
        base.destroy()

    print("\nTotals")
    for place, index in enumerate(sorted(range(3), key=totals.__getitem__, reverse=True), start=1):
        print(f"{place}. {names[index]}: {totals[index]:.1f} m")


if __name__ == "__main__":
    main()
