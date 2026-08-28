from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from racing.game.app import (
    _format_individual_winner_banner,
    _head_to_head_round_finished,
    active_scene_camera_lens,
    build_scene,
    head_to_head_car_label_layout,
    head_to_head_damage_hud_text,
)
from racing.game.config import (
    CameraView,
    CarShowcaseConfig,
    GameConfig,
    HeadToHeadViewerConfig,
    parse_color_rgba,
    parse_window_size,
    positive_float,
)
from racing.graphics.colors import (
    DEFAULT_CHALLENGER_TEAM_COLOR,
    DEFAULT_FORMULA_TEAM_COLOR,
    DEFAULT_INCUMBENT_TEAM_COLOR,
    UNC_CAROLINA_BLUE,
    UNC_FORDHAM_FOUNTAIN,
)
from racing.race.head_to_head import HeadToHeadRaceEntry
from racing.race.runtime import DEFAULT_RACE_RANDOM_SEED, RaceCarRuntime
from racing.student.api import default_student_controller


def test_individual_winner_banner_names_only_first_place_car_and_time() -> None:
    config = HeadToHeadViewerConfig(
        challenger_name="Baseline + Reactive",
        incumbent_name="Imitation",
        challenger_copies=2,
        incumbent_copies=1,
        challenger_copy_names=("Baseline", "Reactive"),
    )
    entries = (
        HeadToHeadRaceEntry(role="challenger", copy_index=0),
        HeadToHeadRaceEntry(role="challenger", copy_index=1),
        HeadToHeadRaceEntry(role="incumbent", copy_index=0),
    )
    runtimes = tuple(
        cast(RaceCarRuntime, SimpleNamespace(tracker=SimpleNamespace(best_distance_m=distance), marshal_penalty_m=0.0))
        for distance in (15.0, 22.5, 18.0)
    )

    assert _format_individual_winner_banner(
        config=config, entries=entries, runtimes=runtimes, elapsed_seconds=60.0
    ) == "1ST PLACE: Reactive\nTIME: 60.0s"


def test_viewed_race_finishes_when_first_car_completes_a_lap() -> None:
    runtimes = tuple(
        cast(RaceCarRuntime, SimpleNamespace(tracker=SimpleNamespace(completed_lap=completed)))
        for completed in (False, True, False)
    )

    assert _head_to_head_round_finished(
        runtimes=runtimes, elapsed_seconds=18.5, time_limit_seconds=60.0
    )


def test_viewed_race_uses_time_limit_when_no_car_finishes() -> None:
    runtimes = (
        cast(RaceCarRuntime, SimpleNamespace(tracker=SimpleNamespace(completed_lap=False))),
    )

    assert not _head_to_head_round_finished(
        runtimes=runtimes, elapsed_seconds=59.9, time_limit_seconds=60.0
    )
    assert _head_to_head_round_finished(
        runtimes=runtimes, elapsed_seconds=60.0, time_limit_seconds=60.0
    )


def test_parse_window_size_accepts_width_by_height() -> None:
    assert parse_window_size("1440x900") == (1440, 900)


def test_parse_color_rgba_accepts_hex_and_decimal_channels() -> None:
    assert parse_color_rgba("#ff8000") == (1.0, 128 / 255, 0.0, 1.0)
    assert parse_color_rgba("0.1, 0.2, 0.3, 0.4") == (0.1, 0.2, 0.3, 0.4)


def test_parse_color_rgba_rejects_out_of_range_channels() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_color_rgba("1.2, 0, 0")


def test_positive_float_rejects_zero() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("0")


def test_game_config_has_no_vehicle_selector() -> None:
    assert not hasattr(GameConfig(), "vehicle_model")


def test_racing_views_default_to_drone_camera() -> None:
    assert GameConfig().camera_view is CameraView.DRONE
    assert HeadToHeadViewerConfig().camera_view is CameraView.DRONE


def test_racing_modes_share_default_random_seed() -> None:
    assert GameConfig().random_seed == DEFAULT_RACE_RANDOM_SEED
    assert HeadToHeadViewerConfig().random_seed == DEFAULT_RACE_RANDOM_SEED


def test_head_to_head_races_default_to_thirty_seconds() -> None:
    assert HeadToHeadViewerConfig().round_seconds == 30.0


def test_head_to_head_damage_label_only_shows_car_name_and_distance() -> None:
    text = head_to_head_damage_hud_text(
        config=HeadToHeadViewerConfig(challenger_name="Candidate"),
        entry=HeadToHeadRaceEntry(role="challenger", copy_index=0),
        distance_m=12.34,
    )

    assert text == "Candidate 1  12.3 m"


def test_overview_camera_views_use_smaller_floating_car_labels() -> None:
    top_down = head_to_head_car_label_layout(CameraView.TOP_DOWN)
    drone = head_to_head_car_label_layout(CameraView.DRONE)

    assert top_down.width < drone.width
    assert top_down.height < drone.height
    assert top_down.text_scale < drone.text_scale


def test_active_scene_camera_lens_reads_the_lens_installed_on_panda_camera() -> None:
    active_lens = object()
    camera_node = SimpleNamespace(getLens=lambda: active_lens)
    ursina = SimpleNamespace(
        application=SimpleNamespace(
            base=SimpleNamespace(
                cam=SimpleNamespace(node=lambda: camera_node),
            )
        )
    )

    assert active_scene_camera_lens(ursina) is active_lens


def test_default_formula_car_uses_fordham_fountain() -> None:
    assert DEFAULT_FORMULA_TEAM_COLOR == UNC_FORDHAM_FOUNTAIN
    assert GameConfig().team_color == DEFAULT_FORMULA_TEAM_COLOR
    assert CarShowcaseConfig().team_color == DEFAULT_FORMULA_TEAM_COLOR


def test_playable_scene_rejects_human_recording_with_student_controller(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manual control"):
        build_scene(
            GameConfig(
                student_controller=default_student_controller,
                human_recording_path=tmp_path / "invalid.jsonl",
            )
        )


def test_default_head_to_head_colors_use_fordham_fountain_versus_carolina_blue() -> None:
    assert DEFAULT_CHALLENGER_TEAM_COLOR == UNC_FORDHAM_FOUNTAIN
    assert DEFAULT_INCUMBENT_TEAM_COLOR == UNC_CAROLINA_BLUE
    assert HeadToHeadViewerConfig().challenger_team_color == DEFAULT_CHALLENGER_TEAM_COLOR
    assert HeadToHeadViewerConfig().incumbent_team_color == DEFAULT_INCUMBENT_TEAM_COLOR
    assert HeadToHeadViewerConfig().challenger_copy_colors == ()
