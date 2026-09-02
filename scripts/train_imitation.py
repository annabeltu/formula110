"""Collect baseline demonstrations and fit the dependency-free imitation policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from controllers import baseline
from controllers.imitation import features
from racing import RobotSensors
from racing.race.head_to_head import HeadToHeadRaceEntry, run_headless_head_to_head


def _solve_ridge(rows: list[tuple[float, ...]], targets: list[float], ridge: float) -> list[float]:
    size = len(rows[0])
    matrix = [[0.0] * (size + 1) for _ in range(size)]
    for row, target in zip(rows, targets, strict=True):
        for column, value in enumerate(row):
            matrix[column][-1] += value * target
            for other, other_value in enumerate(row):
                matrix[column][other] += value * other_value
    for index in range(size):
        matrix[index][index] += ridge
    for pivot in range(size):
        best = max(range(pivot, size), key=lambda row_index: abs(matrix[row_index][pivot]))
        matrix[pivot], matrix[best] = matrix[best], matrix[pivot]
        divisor = matrix[pivot][pivot]
        if abs(divisor) < 1e-12:
            raise ValueError("demonstrations do not span the feature space")
        matrix[pivot] = [value / divisor for value in matrix[pivot]]
        for row_index in range(size):
            if row_index == pivot:
                continue
            factor = matrix[row_index][pivot]
            matrix[row_index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(matrix[row_index], matrix[pivot], strict=True)
            ]
    return [matrix[index][-1] for index in range(size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--races", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "src/controllers/imitation_model.json",
    )
    args = parser.parse_args()
    rows: list[tuple[float, ...]] = []
    throttle: list[float] = []
    steer: list[float] = []
    seen = 0

    def record(_entry: HeadToHeadRaceEntry, sensors: RobotSensors) -> None:
        nonlocal seen
        seen += 1
        if seen % args.sample_every:
            return
        # The expert labels exactly the state observed by the simulator. The
        # race uses two baseline cars to include overtaking and obstruction.
        command = baseline.control(sensors)
        rows.append(features(sensors))
        throttle.append(command.throttle)
        steer.append(command.steer)

    run_headless_head_to_head(
        challenger_controller=baseline.control,
        incumbent_controller=baseline.control,
        challenger_name="Baseline expert A",
        incumbent_name="Baseline expert B",
        race_count=args.races,
        round_seconds=args.seconds,
        random_seed=args.seed,
        sensor_sample_callback=record,
    )
    model = {
        "schema_version": 1,
        "expert": "controllers.baseline.control",
        "samples": len(rows),
        "seed": args.seed,
        "throttle_weights": _solve_ridge(rows, throttle, args.ridge),
        "steer_weights": _solve_ridge(rows, steer, args.ridge),
    }
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} from {len(rows):,} baseline demonstrations")


if __name__ == "__main__":
    main()
