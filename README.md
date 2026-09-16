---
author: Kris Jordan
---

# Formula 110

Formula 110 is a deterministic racing simulator for designing, testing, and
comparing autonomous car controllers. At each 60 Hz simulation tick, your
controller maps a public sensor snapshot to signed throttle and steering
commands. The controller can be rules, an MLP, an evolved policy, a search
procedure, or another method.

Start here:

- [Getting Started](GETTING_STARTED.md): installation, manual driving, and your
  first controller
- [Sensor Reference](SENSORS.md): every input field, type, unit, range, and
  sentinel value
- [Autograder Guide](autograder/README.md): building and operating the isolated
  Gradescope evaluator

## Install and run

Formula 110 uses Python 3.11 and `uv`. From the repository root, install the
locked dependencies once:

```bash
uv sync --managed-python
```

Run one controller on the graphical assignment track:

```bash
uv run racing --student-module controllers.reactive --seed 110
uv run racing --student-module controllers.imitation --seed 110
uv run racing --student-module controllers.baseline --seed 110
```

Replace `110` with any integer to change the starting position. Reusing a seed
reproduces that position, which makes controller comparisons meaningful.

Watch two controllers race (remove `--watch` for the faster, terminal-only
headless version):

```bash
uv run racing h2h --watch \
  --challenger-module controllers.reactive \
  --incumbent-module controllers.imitation \
  --seed 110 --races 1 --round-seconds 30
```

Compare all three supplied strategies in one headless simulation:

```bash
uv run python scripts/race_three_controllers.py \
  --races 7 --seconds 30 --seed 110
```

### Original and technical-test tracks

The normal `racing` commands use the original assignment track. The technical
track is an extra test course with tighter bends and an S-section; it is
installed at runtime by the audit script and is not a normal CLI track choice.

```bash
# Reactive on the technical-test track (headless metrics)
uv run python scripts/test_reactive_new_track.py \
  --seed 110 --races 7 --round-seconds 30

# The same audit on the original track
uv run python scripts/test_reactive_new_track.py \
  --original --seed 110 --races 7 --round-seconds 30

# Watch either track (omit --original for the technical track)
uv run python scripts/test_reactive_new_track.py --watch --seed 110
uv run python scripts/test_reactive_new_track.py --watch --original --seed 110
```

## Controller approaches

### Reactive controller

`controllers.reactive` is a hand-written, sensor-driven policy. It does not
learn while racing and has no model state: every 60 Hz tick it computes a
command directly from the current sensor snapshot. Its main rules are:

- combine heading error, center offset, and near/mid/far lookahead offsets to
  steer along the centerline;
- use ordinary and wall-only LiDAR to avoid walls, blockers, and nearby cars;
- use camera competitor readings to begin passes and avoid cars alongside;
- reduce target speed as turn demand or a front obstacle increases; and
- coast near the target speed and before an approaching wall.

The final throttle is clamped to `[0, MAX_THROTTLE]`. It never requests
negative throttle, and it coasts while rolling backward, so its command never
opposes its current motion. This is the project's **gas-locked-in** behavior
(also written `gas_locked_in` in notes): the car may accelerate or coast, but
does not actively brake. There is no Python symbol named `gas_locked_in`; the
invariant is implemented in `reactive.py`, while the dedicated evaluator is
`tune_gas_locked.py`.

`src/controllers/reactive_params.py` contains the numeric policy constants.
Keeping these values separate from the rules makes them easy to optimize or
edit without rewriting the controller. The weights control steering inputs;
the speed, turn, throttle, front-distance, and coasting values control the
target-speed profile; and the side-wall values control the last-resort wall
guard.

### Imitation controller

`controllers.imitation` uses behavior cloning. Instead of applying explicit
driving rules, it converts each sensor snapshot into 16 normalized and
nonlinear features, then predicts throttle and steering with two fitted linear
models. The weights live in `src/controllers/imitation_model.json`, are loaded
once per controller instance, and require no ML framework at race time.

The training script races two copies of `controllers.baseline`, records the
expert action for sampled sensor states, and fits ridge-regression weights. Two
expert cars provide some traffic and overtaking examples as well as ordinary
track following:

```bash
uv run python scripts/train_imitation.py \
  --races 12 --seconds 30 --seed 110
```

Change `--seed` to change the deterministic sequence of training starts. Other
useful options are `--sample-every`, `--ridge`, and `--output`; run the script
with `--help` for their defaults. Training overwrites
`src/controllers/imitation_model.json` unless `--output` is supplied.

The important limitation is distribution shift: the model learns what the
baseline does in states the baseline visits. It has fewer examples of severe
mistakes, recovery, and genuinely different tracks, so it can drift when it
encounters unfamiliar situations.

## Tuning the reactive controller with Optuna

The primary tuner for the current reactive controller is
`src/controllers/tune_gas_locked.py`. It uses Optuna's seeded TPE sampler to
search the values consumed by `reactive.py` while keeping its gas-locked-in
rule intact:

```bash
# Run 250 trials and save the best values to reactive_params.py
uv run python src/controllers/tune_gas_locked.py --trials 250 --seed 110

# Short experiment without modifying reactive_params.py at the end
uv run python src/controllers/tune_gas_locked.py \
  --trials 30 --seed 2026 --no-save
```

Each trial is evaluated in four 30-second solo scenarios: leaderboard starts
for seeds `110` and `2026`, plus stress-test race indices for both seeds. A
trial receives a large penalty if any scenario fails to complete a lap, is
eliminated, applies a brake, touches a wall, or takes damage. Valid trials
minimize mean fastest-lap time, with an additional wall-contact penalty. The
current values are enqueued as the first trial so Optuna has a known baseline.

The tuner's `--seed` controls **Optuna's sampling sequence**. It does not change
the four evaluation scenarios, which are intentionally fixed in `SCENARIOS`
for fair comparison. To evaluate different racing starts, edit `SCENARIOS` in
the tuner or use the track-audit command with a different `--seed`.

The tuner changes parameters in memory during the search and writes only the
best set at the end. Nevertheless, commit or copy a known-good
`reactive_params.py` before a long run. Use `--no-save` when exploring.

`scripts/tune_reactive.py` is an older head-to-head tuning experiment against
`controllers.baseline`. Its search space and parameter writer predate the
current expanded parameter set, so `tune_gas_locked.py` is the maintained tuner
for the current controller.

## Essential files

| Path | Purpose |
| --- | --- |
| `src/controllers/reactive.py` | Hand-written reactive rules and the gas-locked-in throttle invariant |
| `src/controllers/reactive_params.py` | Tuned constants imported by the reactive controller |
| `src/controllers/tune_gas_locked.py` | Maintained Optuna solo-lap tuner and safety evaluator |
| `scripts/tune_reactive.py` | Older Optuna head-to-head experiment against baseline |
| `src/controllers/imitation.py` | Feature extraction and dependency-free linear imitation inference |
| `src/controllers/imitation_model.json` | Fitted throttle and steering weights used by imitation inference |
| `scripts/train_imitation.py` | Collects baseline demonstrations and refits the imitation artifact |
| `src/controllers/baseline.py` | Rule-based reference/expert used for training and comparisons |
| `scripts/race_three_controllers.py` | Headless Baseline/Reactive/Imitation three-way comparison |
| `scripts/test_reactive_new_track.py` | Reactive audit on the original or runtime-installed technical track |
| `SENSORS.md` | Complete public observation fields, units, ranges, and meanings |
| `GETTING_STARTED.md` | Setup, manual driving, and first-controller tutorial |
| `src/racing/student/api.py` | Public `RobotSensors`, `RobotCommand`, loading, and controller contracts |
| `src/racing/race/head_to_head.py` | Headless race runner, scoring, and result types |
| `src/racing/game/cli.py` | Implementation of the `racing` command and its options |
| `formula110-submission.json` | Submission manifest identifying the controller files/functions to grade |
| `pyproject.toml` / `uv.lock` | Python version, dependencies, scripts, and reproducible dependency lock |
| `tests/` | Physics, sensors, race, rendering, CLI, and controller regression tests |

## Runtime contract

A controller receives an immutable `RobotSensors` snapshot and returns one
`RobotCommand`:

```python
from racing import RobotCommand, RobotSensors


def control(sensors: RobotSensors) -> RobotCommand:
    return RobotCommand(throttle=0.2, steer=0.0)
```

Command ranges are:

| Field | Range | Meaning |
| --- | --- | --- |
| `throttle` | `-1.0` to `1.0` | Reverse to forward drive request |
| `steer` | `-1.0` to `1.0` | Full left to full right |

When signed throttle opposes the car's current motion, the simulator brakes
before applying drive in the new direction. `0.0` coasts. Values outside the
normalized ranges are clamped; `NaN` and infinite command values are rejected.

`RobotSensors` exposes:

| Group | Available information |
| --- | --- |
| `imu` | Heading, turn rate, pitch, roll, and acceleration |
| `odometry` | Signed speed and accumulated travel distance |
| `lidar` | Ranges that detect walls, cars, and blockers |
| `wall_lidar` | Wall-only ranges |
| `camera` | Processed track geometry and nearby competitors |
| `contact` | Current contact durations and accumulated damage |

The processed camera values are geometry, not raw pixels. Controllers do not
receive the mutable physics world, official race progress, future state, or
another controller's private state. See [SENSORS.md](SENSORS.md) for the full
field-by-field contract.

## Deterministic starting positions

Both single-car racing and head-to-head racing accept `--seed`. The seed chooses
a random position along the track using the same deterministic spawn algorithm
in both modes:

```bash
uv run racing --seed 110

uv run racing h2h \
  --challenger-module controllers.candidate \
  --incumbent-module controllers.baseline \
  --seed 110
```

The same seed reproduces the same single-car start and the same head-to-head
race sequence. For multiple head-to-head races, the race index deterministically
selects the next position in that sequence. Programmatic single-car callers can
use `GameConfig.random_seed`; explicit `spawn_position`,
`spawn_heading_degrees`, and `spawn_progress_distance_m` values take precedence
over their corresponding seeded defaults.

The simulator seed controls simulator placement only. It does not seed PyTorch,
NumPy, a genetic algorithm, or stochastic controller inference.

## Baseline imitation learning

`controllers.imitation` is a dependency-free behavior-cloning policy trained
from actions produced by `controllers.baseline`. Its checked-in model was fit
from 8,005 observations gathered while two baseline cars raced, which adds
traffic and obstacle states to ordinary track-following demonstrations.

Retrain the artifact after changing the expert or feature representation:

```bash
uv run python scripts/train_imitation.py --races 12 --seconds 30
```

Race baseline, reactive, and imitation simultaneously in the same headless
physics simulation:

```bash
uv run python scripts/race_three_controllers.py --races 7 --seconds 30 --seed 110
```

Useful demonstrations cover representative speeds, curves, traffic, errors,
and recovery states. The current data varies race start and nearby traffic on
the bundled track, but it does not provide genuinely different tracks and has
few severe mistakes. This means the policy can imitate normal baseline driving
well while still drifting in unfamiliar states. Collecting corrections from
the imitation car's own states (DAgger) is the natural next improvement.

## Packaging a controller

A simple function is the smallest supported controller shape. Keep function
controllers stateless because a module-level object may otherwise be shared by
controller copies during a local multi-car run.

A model-backed or otherwise stateful controller should expose
`create_controller()`. The runtime calls the factory for every car and repeated
race so each receives independent state:

```python
from racing import RobotCommand, RobotSensors

RACING_NAME = "My Controller"
RACING_COLOR = "#4C8DFF"


class Controller:
    def __init__(self) -> None:
        # Load fixed parameters or a trained artifact here, on CPU.
        ...

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        # Convert public sensor values to the policy's representation.
        ...
        return RobotCommand(throttle=0.0, steer=0.0)


def create_controller() -> Controller:
    return Controller()
```

The callable object and function forms implement the same public
`RobotController` protocol. `RACING_NAME` and `RACING_COLOR` are optional
display metadata. Keep model files and controller helper modules under
`src/controllers/` so a controller can move without private simulator files.

## CPU and memory boundary

Submitted inference must run on CPU. CUDA, MPS, ROCm, and other accelerators
must not be required or selected. The complete controller process—including
Python, imported libraries, model parameters, temporary tensors, caches, and
controller state—must remain at or below **512 MiB of resident memory**.

For PyTorch, load artifacts onto CPU, enter evaluation mode, and use inference
mode:

```python
import torch

model = build_your_model()
state = torch.load(model_path, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.to("cpu")
model.eval()

with torch.inference_mode():
    output = model(inputs)
```

Avoid retaining computation graphs, growing history buffers without bounds, or
creating a model on every control tick. The official isolated controller worker
hides common accelerator backends and stops a process tree that exceeds the
memory boundary. Local in-process races do not provide that security sandbox.

## Dependencies and controller artifacts

Add libraries for training or inference with:

```bash
uv add PACKAGE_NAME
uv sync --managed-python
```

Commit both `pyproject.toml` and `uv.lock` when dependency versions change.
Prefer CPU-capable packages and include imported library memory in the 512 MiB
limit. Training-only libraries do not need to be imported by the runtime
controller.

Keep inference artifacts small, read-only, and addressed relative to the
controller module rather than the current working directory. Export the full
controller package when it uses non-Python files or dynamically loaded helpers:

```bash
uv run python scripts/export_student_controllers.py --all-controllers
```

The archive packages `src/controllers/`. Dependency declarations remain in
`pyproject.toml` and `uv.lock`. Do not package training-only datasets, virtual
environments, or experiment logs with the runtime controller.

## Capturing human demonstrations

Manual keyboard and gamepad driving can be captured as observation/action pairs:

```bash
uv run racing \
  --seed 110 \
  --record-human artifacts/human-driving.jsonl
```

The destination is append-only JSON Lines. Each physics tick produces one
independently parseable record:

```json
{
  "schema_version": 2,
  "record_type": "human_control_step",
  "session_id": "...",
  "simulation_time_s": 0.016666666666666666,
  "sensors": {
    "dt_s": 0.016666666666666666,
    "tick": 0,
    "imu": {},
    "odometry": {},
    "lidar": {},
    "wall_lidar": {},
    "camera": {},
    "contact": {}
  },
  "command": {"throttle": 1.0, "steer": 0.0}
}
```

The empty sensor objects only keep this example compact; actual records contain
every public field. A row captures the state immediately before its command is
applied, so the result of the action appears in the next row. Recording stops
when the car is eliminated or the app exits.

Commands contain normalized simulator controls rather than raw input events.
Infinite LiDAR no-hit values are serialized as JSON `null`. Each launch appends
with a new `session_id`; split trajectories on that ID rather than treating the
first row of a new session as following the previous session.

`--record-human` is limited to single-car manual mode and cannot be combined
with `--student-module` or `h2h`. The recording format does not prescribe an
observation vector, normalization strategy, imitation objective, or train/test
split.

## Comparing controllers

Use a watched race when you need to understand behavior:

```bash
uv run racing h2h --watch \
  --challenger-module controllers.candidate \
  --incumbent-module controllers.baseline \
  --seed 110 \
  --races 1 \
  --round-seconds 30
```

One side can use keyboard control in a watched race:

```bash
uv run racing h2h --watch \
  --challenger-keyboard \
  --incumbent-module controllers.baseline \
  --seed 110 \
  --camera follow
```

Keyboard head-to-head requires `--watch`, and a headless race requires automated
controllers on both sides. Run several headless comparisons when you need
faster evidence:

```bash
uv run racing h2h \
  --challenger-module controllers.candidate \
  --incumbent-module controllers.baseline \
  --seed 110 \
  --races 7 \
  --round-seconds 30
```

Races default to 30 seconds. On the starting grid, the car in the outside lane
starts ahead of the car in the inside lane. Scored distance is forward track
progress minus marshal penalties. Wall and car contact continue to count toward
progress; contact and damage are reported separately and do not multiply or
otherwise reduce distance. At the end of a watched race, the simulation pauses
on the final positions and keeps the window open with a winner banner and both
sides' scored distances.

Add `--json` for a versioned machine-readable result. The Python API exposes the
same runner:

```python
from racing import load_student_submission, run_headless_head_to_head

candidate = load_student_submission("controllers.candidate")
baseline = load_student_submission("controllers.baseline")

result = run_headless_head_to_head(
    challenger_controller=candidate.controller,
    incumbent_controller=baseline.controller,
    challenger_name=candidate.display_name or "candidate",
    incumbent_name=baseline.display_name or "baseline",
    race_count=7,
    random_seed=110,
)
record = result.to_dict()
```

`run_headless_head_to_head` also accepts `fixed_delta_seconds`, copy counts,
race rules, and a `sensor_sample_callback` observation hook. Formula 110 does
not define a replay buffer, reward, fitness function, optimizer, or training
loop.

## Reproducibility and fair comparison

Record at least the controller version, seed, race count, round duration,
timestep, and race rules. One seed or opponent is weak evidence; evaluate
across several seeds and retain a baseline controller for regressions.

Head-to-head outcomes can depend on traffic and contact, so solo distance is not
a substitute for racing against an opponent. Results include scored and raw
distance, laps, damage, contact, speed, off-track time, marshal activity, and
per-race winners.

Keep working controller versions and compare them directly:

```bash
cp src/controllers/candidate.py src/controllers/baseline.py
```

Improve the candidate, then evaluate both from identical seeds. A change that
looks better in one watched run can still lose distance or take more damage over
a multi-seed suite.

## Project map

Stay on the public surface unless a task specifically changes the simulator:

| Path | Purpose |
| --- | --- |
| `GETTING_STARTED.md` | Installation, first drive, and first controller |
| `SENSORS.md` | Complete sensor types, units, ranges, and semantics |
| `src/controllers/` | Student controllers, helpers, and model artifacts |
| `src/racing/student/api.py` | Sensor, command, loading, and controller contracts |
| `src/racing/game/recording.py` | Human JSONL schema and serializers |
| `src/racing/race/head_to_head.py` | Public headless runner and result types |
| `src/racing/race/rules.py` | Competitive scoring and marshal rules |
| `autograder/` | Isolated Gradescope packaging and runtime |
| `tests/` | Simulator contract and regression tests |

For controller work, import friendly names from `racing`. Do not couple a policy
to private underscore-prefixed functions, Panda3D nodes, or mutable physics
objects. Before implementation, identify the observation representation,
controller packaging choice, evaluation suite, CPU behavior, and expected
memory footprint.

## Intentional non-goals

Formula 110 does not prescribe or provide a neural-network architecture,
genetic algorithm, observation vector, normalization scheme, reward, fitness
function, replay buffer, optimizer, training schedule, hyperparameters, or
experiment tracker. Those are controller-design decisions. The stable handoff
point is a CPU controller that fits within 512 MiB and maps the documented
sensor snapshot to a valid command.
