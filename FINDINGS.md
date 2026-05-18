# Simulator Findings

## Executive Summary
This document explains, in detail, how the GPS/IMU kart simulator in this repository was designed and built, what design choices were made along the way, how the simulator currently works, what tuning and validation steps were performed, and what the overall findings are.

The short version is:

- A runner-compatible simulator was successfully built around the existing Donkey `Vehicle` graph.
- The simulator can launch the real `cte_runner.py` control stack without hardware by replacing only the hardware-edge parts.
- The simulator architecture is good enough to support continued iteration.
- Midline-style, shorter-window comparisons became fairly good after sensor-model tuning.
- Full-run / full-lap comparisons against the recent raceline logs (`data9`, `data011`) still show substantial mismatch.
- The main unresolved issue is not repository plumbing anymore. It is model fidelity, especially speed-dependent lateral behavior and path / loop handling under high-speed raceline conditions.

This means the simulator project moved from "infrastructure missing" to "modeling and validation gap remains."

## Original Goal
The goal was to create a pseudo-simulator for the kart such that existing GPS/IMU runners could be run in simulation, ideally through an invocation of the form:

```bash
python sim.py "cte_runner.py gps_paths/path.csv"
```

The intent was not to build a separate toy controller sandbox. The intent was to run the real runner, with the real estimator and controller stack, against simulated hardware and dynamics.

That goal drove almost every design choice.

## Constraints and Scope
The project evolved under the following constraints:

1. The initial target was GPS/IMU runners only, not camera-based runners.
2. Small runner refactors were allowed.
3. Analysis and calibration had to use the `aks/` logs and avoid the `manual*` logs.
4. Later tuning explicitly focused on `data5`, `data7`, `data9`, and `data011`.
5. The simulator needed to use real observed kart behavior rather than hand-wavy constants.

The scope was intentionally narrowed to make meaningful progress:

- In scope:
  - `cte_runner.py`
  - simulated GPS
  - simulated BNO086-style IMU
  - simulated UART sink
  - simulated heartbeat
  - parameter fitting from logs
  - validation against logged behavior

- Out of scope:
  - camera / segmentation / pursuit runners
  - photorealistic rendering
  - tire-force / slip-angle / full dynamic vehicle model
  - exact hardware serial protocol emulation
  - generic support for all runners immediately

## What Was Already in the Repo
Before any changes, the repo already contained several useful ingredients:

1. `cte_runner.py`
   - The main GPS/IMU path-following runner.
   - Uses `GPS`, `BNO086`, `GPS_to_xy`, `HeadingFusion`, `CTEController`, and `UART_backup_driver`.

2. `cte_pseudo_sim.py`
   - A standalone pseudo simulator.
   - Useful as a source of path loading, track geometry, and simple actuator/dynamics logic.
   - Not runner-compatible.

3. `parts/fake_gps.py` and `parts/fake_imu.py`
   - Useful as evidence that simulated parts were conceptually acceptable.
   - Not appropriate as-is because each sensor advanced its own motion independently.

4. `parts/gps_to_xy.py`
   - Important because it converts lat/lon to the local Cartesian frame used by the controller.

5. `parts/heading_fusion.py`
   - Important because it fuses GPS and IMU heading and position into the state actually used by the controller.

6. `parts/cte_controller.py`
   - The production path-following controller.

The most important realization early on was this:

The repo already had enough machinery to make a simulator practical, but not by running `cte_pseudo_sim.py` directly. The correct move was to preserve the `Vehicle` graph and replace only hardware-boundary parts.

## First Major Design Decision: Simulate the Hardware Boundary, Not the Controller
This was the single most important architectural choice.

### Option A: Keep `cte_pseudo_sim.py` and adapt it
Pros:
- Already had a path loader
- Already had a simple bicycle model
- Already had noise injection

Cons:
- Bypassed the real runner
- Bypassed the real estimator
- Did not exercise the same control graph as the physical kart
- Used its own internal control flow rather than the Donkey `Vehicle` loop

### Option B: Keep the real runner and fake only sensors and actuators
Pros:
- Reuses the real `GPS_to_xy`
- Reuses the real `HeadingFusion`
- Reuses the real `CTEController`
- Preserves runner behavior
- Makes validation much more meaningful

Cons:
- More integration work up front
- Requires a shared truth state and multiple fake parts

Option B was chosen.

This decision led directly to the final architecture.

## Resulting Architecture
The simulator is now built around the following idea:

1. The runner still builds a Donkey `Vehicle`.
2. The parts that normally touch hardware are replaced by simulated parts.
3. Those simulated parts are all driven by one shared truth state.
4. The controller still sees realistic channels like GPS lat/lon, compass heading, IMU accel, gyro, etc.
5. The controller output still flows to a UART-like sink, but that sink writes into the simulated vehicle state instead of a serial port.

In effect:

- Real runner logic stays.
- Hardware I/O gets virtualized.

That architecture is implemented through:

- `cte_runner.py`
- `sim.py`
- `sim/shared_truth.py`
- `parts/sim_gps.py`
- `parts/sim_bno086.py`
- `parts/sim_uart.py`
- `parts/sim_health_check.py`

## Files Added or Changed
### New files
- `FINDINGS.md`
- `sim.py`
- `sim/__init__.py`
- `sim/track_loader.py`
- `sim/kart_dynamics.py`
- `sim/shared_truth.py`
- `sim/calibration.py`
- `sim/fit_from_logs.py`
- `sim/validation.py`
- `sim/kart_params.json`
- `parts/noop_part.py`
- `parts/sim_health_check.py`
- `parts/sim_uart.py`
- `parts/sim_gps.py`
- `parts/sim_bno086.py`

### Existing files changed
- `cte_runner.py`
- `parts/logger_gps.py`

### Environment changes made during implementation
Two Python packages had to be installed so the vendored `donkeycar` package would import correctly:

- `pyfiglet`
- `prettytable`

Without those, even importing `donkeycar` failed.

## Making `cte_runner.py` Runner-Compatible With Simulation
`cte_runner.py` originally instantiated hardware parts directly at import/runtime and immediately executed the drive loop. That made it hard to simulate.

It was refactored to expose:

- `build_arg_parser()`
- `parse_args()`
- `build_vehicle(args, part_overrides=None, runtime_overrides=None)`
- `main()`

This was a small but extremely important refactor.

### Why this matters
Without a builder seam, the simulator would have needed brittle monkeypatching.

With the seam:

- the simulator can inject `SimGPS` instead of `GPS`
- inject `SimBNO086` instead of `BNO086`
- inject `SimUARTBackupDriver` instead of `UART_backup_driver`
- inject `SimHealthCheck`
- disable telemetry cleanly

At the same time, normal CLI behavior of `cte_runner.py` was preserved.

### Additional improvement in `cte_runner.py`
The file now also bootstraps the vendored DonkeyCar package by inserting:

- `static_donkeycar/donkeycar`

into `sys.path` before importing `donkeycar`.

This was necessary because the repo relies on a vendored DonkeyCar rather than a globally installed package.

### Lazy-loading hardware parts
Another important change was making hardware-only imports lazy.

Why?

Because simply importing `cte_runner.py` was trying to import modules like:

- `parts.gps`
- `parts.bno086`

which in turn depend on hardware / external GNSS libraries.

That broke the simulator before it could override anything.

So those real hardware parts are now instantiated lazily through helper factories, only if needed.

## Extracting Reusable Simulator Pieces
`cte_pseudo_sim.py` contained useful logic, but it was trapped inside a standalone script. The useful pieces were extracted conceptually into reusable modules.

### `sim/track_loader.py`
Purpose:
- Load path CSVs in either lat/lon or XY form.
- Convert geodetic paths into local XY.
- Build heading and arc-length arrays.
- Generate controller-friendly `_xy` and `_xy_throttle` files.

Key design choices:

1. Support both geodetic and XY inputs.
   - Needed because repo path files are inconsistent.

2. Automatically generate controller CSVs.
   - `cte_runner.py` expects a `_xy_throttle` file.
   - Many such files did not exist.
   - The simulator therefore had to generate them.

3. Drop duplicated loop endpoint if the last point equals the first.
   - This became important later when raceline loops showed seam problems.

This file is one of the most useful long-term outputs of the project because it normalized path handling.

### `sim/kart_dynamics.py`
Purpose:
- Define the simulated kart state and dynamics parameters.
- Advance the truth model each time step.

It currently models:

1. Speed target as a linear function of throttle command.
2. First-order lag on speed response.
3. Steering command saturation / rate limit.
4. Steering command delay.
5. First-order lag on steering effectiveness.
6. Yaw-rate based on curvature:

```text
yaw_rate = speed * curvature
curvature = c0 + c1 * steering + c3 * steering^3
```

This is still fundamentally a kinematic model, not a dynamic tire model.

### `sim/shared_truth.py`
Purpose:
- Hold the authoritative state of the simulated kart.
- Run the truth dynamics in a background thread.
- Receive actuator commands from the fake UART part.
- Provide consistent truth snapshots to the fake GPS and fake IMU.

This was necessary because fake sensors must all observe the same vehicle.

That was a critical improvement over the repo’s older `fake_gps` / `fake_imu` approach.

### `sim/calibration.py`
Purpose:
- Load and save the simulation parameter artifact.

This keeps parameter loading simple and centralized.

### `sim/fit_from_logs.py`
Purpose:
- Reproducibly extract parameters from the allowed `aks/data*.csv` logs.

This file is where most of the empirical modeling logic lives.

### `sim/validation.py`
Purpose:
- Summarize real and simulated logger CSVs.
- Compare them.
- Compute lap-aware windows when possible.
- Estimate track-relative CTE.

Validation became more sophisticated over time as simple fixed-window checks proved misleading.

## Simulated Hardware Parts
### `parts/sim_uart.py`
Purpose:
- Emulate the actuator sink.
- Receive `(throttle, steering, alive)` just like the real UART part.
- Forward those commands into the shared truth state.

Important behavior preserved:
- startup warmup loops
- startup throttle clipping

That was deliberate because startup behavior matters for trajectory evolution.

### `parts/sim_gps.py`
Purpose:
- Emulate the legacy GPS tuple returned by `parts.gps.GPS`.

It outputs:

```text
(lat, lon, alt, fix, diff_age, hdop, num_sv, course_deg, speed_mps)
```

Important modeling choices:

1. Fix states are probabilistic.
2. GPS position noise depends on fix state.
3. GPS course and speed are noisy observations derived from truth.
4. GPS lat/lon are produced from local XY through `GPS_to_xy.to_latlon`.

Later, this part was upgraded from simple white position noise to correlated noise plus a white component, because pure white GPS noise made `GPS_to_xy` heading estimates unrealistically jittery.

### `parts/sim_bno086.py`
Purpose:
- Emulate the Donkey-format BNO086 part.

It outputs:

```text
[heading, accuracy_deg, (lin_ax, lin_ay, lin_az), (gx, gy, gz)]
```

The heading is returned in compass format to match what `HeadingFusion` expects.

This was a very important convention detail. If this had been wrong, all yaw comparisons would have been corrupted.

### `parts/sim_health_check.py`
Purpose:
- Always return `True`.

Very simple, but necessary because the runner expects a heartbeat.

### `parts/noop_part.py`
Purpose:
- Disable optional side-effect parts like telemetry cleanly.

## The Simulation Entry Point: `sim.py`
`sim.py` is the top-level wrapper.

### What it does
1. Parse a quoted runner command.
2. Load the runner module.
3. Load calibrated simulator parameters.
4. Load the requested track.
5. Construct the shared truth state.
6. Inject simulated parts into the runner.
7. Run the real `Vehicle` loop.
8. Write:
   - a truth log
   - the normal runner logger CSV
   - a JSON summary

### Current support
Right now `sim.py` only supports:

- `cte_runner.py`

This was intentional. Broad generic support was deferred until the first runner actually worked.

### Important CLI controls
`sim.py` now supports:

- `--controller-throttle`
- `--max-time-s`
- `--stop-at-lap`
- `--reference-log`
- `--params`
- `--seed`

`--controller-throttle` became especially important once it became clear that one simulated throttle regime was not enough to judge generalization.

## Initial Empirical Calibration
Before building the simulator, the logs were analyzed to extract basic relationships.

### Steering-to-curvature relationship
From the allowed logs, the approximate relationship was estimated as something like:

```text
curvature ~= c0 + c1 * steering
```

with a nonzero steering trim / bias.

That gave an initial lateral-response model.

### Command-to-speed relationship
A linear command-to-speed model was also fitted:

```text
speed ~= a + b * throttle
```

This was not assumed from first principles. It was fitted from observed straight-ish plateaus in the logs.

### Fitted parameter artifact
These results were persisted in:

- `sim/kart_params.json`

This file is the simulator’s parameter source of truth.

## Why the First Validation Looked Promising
Early on, after the infrastructure was in place, the simulator looked fairly good in short midline-style checks.

One important tuned run produced:

- simulated steering mean very close to a reference log
- GPS speed mean also reasonably close

That improvement came from realizing the following:

### Key finding: white GPS noise was overdriving the controller
Initially, the simulator injected independent GPS position noise at every fix.

That meant:

- successive GPS positions jittered unrealistically
- `GPS_to_xy` converted that into erratic GPS heading
- `HeadingFusion` had to fuse that noisy signal
- `CTEController` responded with too much steering

Once the GPS model was changed to use mostly correlated noise, with only a small white component, the steering magnitude dropped dramatically and matched the real logs much better in those short-window tests.

This was one of the most important findings of the whole effort.

## Correlated GPS Noise Model
The GPS model was changed to:

1. keep a slow-varying bias in east and north
2. update that bias with exponential correlation
3. add a smaller white noise term on top

Parameters added:

- `gps_noise_correlation_s`
- `gps_white_noise_fraction`

This made the GPS signal behave more like a real RTK stream: noisy, but not violently incoherent from fix to fix.

## Why the "Raw Sensor Noise" in the Parameter Fit Is Misleading
One important caveat:

The `fit_report.raw_sensor_noise.gps_position_sigma_m` values currently stored in `sim/kart_params.json` are not physically trustworthy as absolute RTK RMS error estimates.

Why?

Because those estimates were derived by comparing logged geodetic position against fused local XY across logs that are not all guaranteed to share a perfectly consistent local frame / origin alignment.

That means the raw absolute position residual can absorb frame mismatch, not just sensor noise.

This is why the simulator currently uses a manually tuned "effective GPS innovation" instead of blindly trusting the raw fitted noise.

This is not ideal, but it was the right pragmatic decision.

## Extending Validation Beyond the First 500 Ticks
At first, many comparisons were effectively short-window checks because the simulator was often run with a fixed loop count.

That turned out to be misleading.

The user correctly pointed out that the simulator should work across:

- `data5`
- `data7`
- `data9`
- `data011`

and across the whole run, not just an initial snippet.

This caused two major validation improvements:

1. `sim.py` gained `--stop-at-lap`
2. `sim/validation.py` gained lap-window extraction

### Lap extraction strategy
The validator now tries to detect a lap by:

1. computing traveled distance along the trajectory
2. waiting until at least about `0.75 * path_length`
3. checking whether the trajectory returns near its own starting point

If no such segment exists, validation falls back to the full available run.

This was much more robust than naive nearest-path index wrap logic.

## Matching Real Logs to the Correct Paths
Another big discovery was that not all logs should be compared against the same path file.

By checking nearest-distance statistics, the following approximate mapping emerged:

- `data5` matched best with `gps_raceline_run2_spline_aligned_to_old.csv`
- `data7` matched reasonably with `gps_raceline_edited_v2.csv`
- `data9` matched best with `gps_raceline_run2_lstsq_aligned_to_old.csv`
- `data011` also matched best with `gps_raceline_run2_lstsq_aligned_to_old.csv`

This matters a lot because comparing a raceline log to a midline path can make the controller look worse than it really is.

## Additional Loop / Seam Issue Discovered
The raceline loop files repeated the first point as the last point.

That sounds harmless, but in practice it can create a degenerate seam segment for the controller’s nearest-waypoint / tangent logic.

This was addressed in `sim/track_loader.py` by dropping the duplicated loop endpoint during load.

That removed one class of seam artifact, though it did not solve the higher-speed fidelity problem by itself.

## Generalization Testing Across Throttle Regimes
Once `sim.py` supported `--controller-throttle`, the simulator was tested against multiple operating points.

### What happened
Short-window checks suggested:

- `2500` was fairly good
- `3000` was in the ballpark
- `4500` was not good enough

But once the validation became full-lap / full-run aware, the real picture became harsher:

- `data9` full-lap behavior still diverged substantially
- `data011` full-run behavior diverged even more

This is an important overall finding:

The current simulator does not yet generalize well across all throttle regimes when judged on full trajectories.

## Current Parameter Artifact
The current `sim/kart_params.json` captures the best working state reached during this effort.

Notable values include:

- `speed_intercept_mps`: `0.09437329478172252`
- `speed_slope_mps_per_erpm`: `0.0007807196721023582`
- `speed_lag_s`: `0.47614097595214844`
- `curvature_intercept_inv_m`: `0.012480314031610977`
- `curvature_gain_inv_m`: `-0.15298009663471046`
- `steering_delay_s`: `0.6799999999999997`
- effective GPS innovation:
  - `RTK FIXED`: `0.003`
  - `RTK FLOAT`: `0.08`
  - `3D`: `0.30`
  - `NO FIX`: `1.0`

These values should be interpreted as "best current empirical working values," not final physical truths.

## Major Findings
### Finding 1: Infrastructure is no longer the bottleneck
The repository now has a working simulation framework that:

- launches the real runner
- injects simulated sensors / actuators
- preserves the real estimator and controller stack
- emits comparable logs
- supports parameter loading
- supports lap-aware validation

This is a major success.

### Finding 2: Sensor modeling mattered more than expected
The jump from white GPS noise to correlated GPS noise had a huge effect on controller behavior.

This means "realistic sensor temporal structure" matters, not just noise magnitude.

### Finding 3: Short-window validation can be misleading
It is possible to get a simulator that looks pretty good over an initial 10-second slice while still failing badly over a whole lap.

This was directly observed here.

### Finding 4: Full-run raceline fidelity is still poor
On the more recent and more demanding runs (`data9`, `data011`), the current simulator still oversteers and accumulates too much cross-track error over long trajectories.

That remained true even after:

- better path matching
- better loop handling
- correlated GPS noise
- throttle-specific validation

### Finding 5: The remaining problem is probably structural, not just scalar tuning
At this point, the remaining mismatch is unlikely to be solved by simply twiddling one or two scalar constants.

The simulator probably needs at least one of:

1. speed-dependent steering effectiveness / understeer
2. speed-dependent actuator lag or saturation
3. a better lateral model than a fixed linear curvature gain
4. a start-state initialization that depends on the chosen track and recorded run
5. more careful path preprocessing for looped racelines

## What the Simulator Does Well Right Now
1. Runs `cte_runner.py` without physical hardware.
2. Preserves the `GPS_to_xy -> HeadingFusion -> CTEController` pipeline.
3. Produces logger CSVs and truth logs.
4. Supports reproducible parameter fitting.
5. Supports quick experimentation and debugging.
6. Supports throttle override and lap stopping.

This is already useful for:

- code iteration
- controller debugging
- estimator debugging
- sensitivity experiments
- regression testing basic behavior

## What the Simulator Does Poorly Right Now
1. Full-lap high-speed raceline fidelity.
2. Generalization across all operating points using one fixed lateral model.
3. Deriving absolute GPS noise directly from the current logs.
4. Exact matching of row counts / duration against the real logs.
5. Physical realism beyond a kinematic / quasi-kinematic abstraction.

## Why This Still Counts as Progress
Even though the full-run fidelity is not yet where it needs to be, the project made meaningful progress in several ways:

1. It converted a collection of disconnected scripts into a coherent simulator framework.
2. It identified the major current sources of mismatch.
3. It exposed that the next stage of work is model fidelity, not repo surgery.
4. It produced reusable infrastructure that future simulator improvements can build on.

Without this groundwork, every next modeling attempt would still have been blocked on integration plumbing.

Now that plumbing exists.

## Recommended Next Steps
If this simulator is to be pushed further, the next best steps are:

1. Add a speed-dependent lateral model.
   - For example, let effective curvature gain depend on speed.
   - This is the most likely next improvement.

2. Initialize from recorded state where possible.
   - Instead of always starting at the path start, allow start pose / speed to be specified or inferred.

3. Separate path validation modes:
   - midline mode
   - raceline mode

4. Improve raw noise fitting.
   - Fit in a more consistent local frame.
   - Avoid mixing coordinate alignment error into "sensor noise."

5. Add structured parameter sweeps.
   - Especially for speed-dependent lateral coefficients.

6. Add a reproducible validation harness that compares:
   - short-window metrics
   - lap metrics
   - full-run metrics

## Current Bottom-Line Assessment
The simulator project is now in this state:

### Successfully achieved
- Real-runner-compatible simulation for `cte_runner.py`
- Shared-truth multi-sensor simulation
- Reproducible parameter artifact
- Reasonable short-window / midline behavior
- Good platform for continued tuning

### Not yet achieved
- Full-fidelity multi-throttle full-lap match across `data5`, `data7`, `data9`, and `data011`
- Robust high-speed raceline realism

Put bluntly:

The simulator now exists and works structurally, but it is not yet trustworthy as a high-fidelity "extreme fidelity to the actual thing" replacement across all regimes.

That is not because the implementation failed.
It is because the remaining gap is now genuinely a modeling problem.

## Final Conclusion
The work accomplished here should be viewed as Phase 1 of a serious simulator effort:

- Phase 1: runner-compatible simulation infrastructure, empirical first-pass tuning, validation tooling
- Phase 2: speed-dependent lateral model and better high-speed raceline fidelity

Phase 1 is complete enough to be useful.
Phase 2 is required before claiming strong fidelity across all of the recent logged runs.

