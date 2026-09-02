"""Open a graphical race with baseline, reactive, and imitation cars."""

from __future__ import annotations

import argparse

from controllers import baseline, imitation, reactive
from racing.game.app import create_head_to_head_viewer_app
from racing.game.config import CameraView, HeadToHeadViewerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--races", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=110)
    args = parser.parse_args()
    create_head_to_head_viewer_app(
        HeadToHeadViewerConfig(
            title="Formula 110: Baseline vs Reactive vs Imitation",
            camera_view=CameraView.DRONE,
            challenger_name="Baseline + Reactive",
            incumbent_name="Imitation",
            challenger_controller=baseline.control,
            challenger_copy_controllers=(baseline.control, reactive.control),
            incumbent_controller=imitation.create_controller(),
            challenger_copies=2,
            incumbent_copies=1,
            challenger_copy_names=("Baseline", "Reactive"),
            challenger_copy_colors=(
                (0.14, 0.78, 0.46, 1.0),
                (0.78, 0.14, 0.61, 1.0),
            ),
            incumbent_team_color=(0.95, 0.76, 0.31, 1.0),
            race_count=args.races,
            round_seconds=args.seconds,
            random_seed=args.seed,
        )
    ).run()


if __name__ == "__main__":
    main()
