"""Tune the reactive controller using Optuna."""

from __future__ import annotations

import importlib
from pathlib import Path

import optuna


from racing import load_student_submission, run_headless_head_to_head


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

PARAM_FILE = Path("src/controllers/reactive_params.py")

# Number of Optuna parameter combinations to try.
NUM_TRIALS = 30

# Number of races used to evaluate each combination.
RACES_PER_TRIAL = 5

# Keeps the experiment reproducible.
RACING_SEED = 110


# --------------------------------------------------
# WRITE PARAMETERS
# --------------------------------------------------

def write_parameters(
    center_weight,
    far_lookahead_weight,
    base_speed,
    turn_slowdown,
    throttle_gain,
    max_throttle,
    front_slow_distance,
    front_speed_scale,
):
    PARAM_FILE.write_text(
        f"""CENTER_WEIGHT = {center_weight}
FAR_LOOKAHEAD_WEIGHT = {far_lookahead_weight}
BASE_SPEED = {base_speed}
TURN_SLOWDOWN = {turn_slowdown}
THROTTLE_GAIN = {throttle_gain}
MAX_THROTTLE = {max_throttle}
FRONT_SLOW_DISTANCE = {front_slow_distance}
FRONT_SPEED_SCALE = {front_speed_scale}
"""
    )


# --------------------------------------------------
# FIND NUMERIC VALUES INSIDE RESULT DICTIONARY
# --------------------------------------------------

def find_values(data, words):
    """
    Search a nested result dictionary for numeric values whose
    key path contains all of the requested words.
    """

    matches = []

    def search(value, path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                search(item, f"{path}.{key}".lower())

        elif isinstance(value, list):
            for index, item in enumerate(value):
                search(item, f"{path}[{index}]".lower())

        elif isinstance(value, (int, float)):
            if all(word in path for word in words):
                matches.append(float(value))

    search(data)

    return matches


# --------------------------------------------------
# SCORE ONE HEAD-TO-HEAD RESULT
# --------------------------------------------------

def score_result(record: dict) -> float:
    """
    Higher score = better reactive controller.

    First try to use challenger scored distance.
    If that field cannot be found, try challenger distance.
    """

    scored_distances = find_values(
        record,
        ["challenger", "scored", "distance"],
    )

    if scored_distances:
        return sum(scored_distances)

    distances = find_values(
        record,
        ["challenger", "distance"],
    )

    if distances:
        return sum(distances)

    # If Formula 110's result structure is different,
    # print it so we can see the exact field names.
    print("\nCould not automatically locate challenger distance.")
    print("Result dictionary:")
    print(record)

    raise RuntimeError(
        "Could not find a distance field in Formula 110 results."
    )


# --------------------------------------------------
# LOAD BASELINE ONCE
# --------------------------------------------------

baseline = load_student_submission("controllers.baseline")


# --------------------------------------------------
# OPTUNA OBJECTIVE
# --------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    """
    Optuna calls this function once per trial.

    Each trial:
        1. chooses parameters
        2. writes them to reactive_params.py
        3. reloads reactive controller
        4. runs races
        5. returns performance score
    """

    # --------------------------------------------------
    # 1. OPTUNA CHOOSES PARAMETERS
    # --------------------------------------------------

    center_weight = trial.suggest_float(
        "CENTER_WEIGHT",
        0.05,
        0.30,
    )

    far_lookahead_weight = trial.suggest_float(
        "FAR_LOOKAHEAD_WEIGHT",
        0.005,
        0.05,
    )

    # Allow faster target speeds.
    base_speed = trial.suggest_float(
        "BASE_SPEED",
        6.0,
        12.0,
    )

    # Allow the car to slow down less aggressively.
    turn_slowdown = trial.suggest_float(
        "TURN_SLOWDOWN",
        0.5,
        5.0,
    )

    # How aggressively it tries to reach target speed.
    throttle_gain = trial.suggest_float(
        "THROTTLE_GAIN",
        0.20,
        0.60,
    )

    # Formula 110 allows throttle up to 1.
    max_throttle = trial.suggest_float(
        "MAX_THROTTLE",
        0.70,
        1.0,
    )

    # How early to slow for an obstacle.
    front_slow_distance = trial.suggest_float(
        "FRONT_SLOW_DISTANCE",
        4.0,
        8.0,
    )

    # How restrictive front-distance speed control is.
    front_speed_scale = trial.suggest_float(
        "FRONT_SPEED_SCALE",
        0.60,
        1.50,
    )

    # --------------------------------------------------
    # 2. WRITE PARAMETERS
    # --------------------------------------------------

    write_parameters(
        center_weight,
        far_lookahead_weight,
        base_speed,
        turn_slowdown,
        throttle_gain,
        max_throttle,
        front_slow_distance,
        front_speed_scale,
    )

    # --------------------------------------------------
    # 3. RELOAD CONTROLLER
    # --------------------------------------------------

    import controllers.reactive_params as reactive_params
    import controllers.reactive as reactive

    importlib.reload(reactive_params)
    importlib.reload(reactive)

    # --------------------------------------------------
    # 4. RUN RACES
    # --------------------------------------------------

    result = run_headless_head_to_head(
        challenger_controller=reactive.control,
        incumbent_controller=baseline.controller,
        challenger_name="Reactive",
        incumbent_name="Baseline",
        race_count=RACES_PER_TRIAL,
        random_seed=RACING_SEED,
    )

    record = result.to_dict()

    # --------------------------------------------------
    # 5. SCORE RESULT
    # --------------------------------------------------

    score = score_result(record)

    print()
    print("----------------------------------------")
    print(f"Trial: {trial.number}")
    print(f"CENTER_WEIGHT:        {center_weight:.4f}")
    print(f"FAR_LOOKAHEAD_WEIGHT: {far_lookahead_weight:.4f}")
    print(f"BASE_SPEED:           {base_speed:.4f}")
    print(f"TURN_SLOWDOWN:        {turn_slowdown:.4f}")
    print(f"THROTTLE_GAIN:        {throttle_gain:.4f}")
    print(f"MAX_THROTTLE:         {max_throttle:.4f}")
    print(f"FRONT_SLOW_DISTANCE:  {front_slow_distance:.4f}")
    print(f"FRONT_SPEED_SCALE:    {front_speed_scale:.4f}")
    print("----------------------------------------")

    return score


# --------------------------------------------------
# RUN OPTUNA
# --------------------------------------------------

def main():

    # TPE starts by exploring and then uses previous
    # trials to search more intelligently.
    sampler = optuna.samplers.TPESampler(seed=110)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=NUM_TRIALS,
    )

    # --------------------------------------------------
    # DISPLAY BEST RESULT
    # --------------------------------------------------

    print()
    print("========================================")
    print("BEST PARAMETERS")
    print("========================================")

    for name, value in study.best_params.items():
        print(f"{name}: {value}")

    print()
    print(f"Best score: {study.best_value}")

    # --------------------------------------------------
    # SAVE BEST PARAMETERS
    # --------------------------------------------------

    best = study.best_params
    write_parameters(
        best["CENTER_WEIGHT"],
        best["FAR_LOOKAHEAD_WEIGHT"],
        best["BASE_SPEED"],
        best["TURN_SLOWDOWN"],
        best["THROTTLE_GAIN"],
        best["MAX_THROTTLE"],
        best["FRONT_SLOW_DISTANCE"],
        best["FRONT_SPEED_SCALE"],
    )

    print()
    print("Best parameters saved to:")
    print(PARAM_FILE)


if __name__ == "__main__":
    main()