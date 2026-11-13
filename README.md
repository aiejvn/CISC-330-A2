# CISC/CMPE 330 Assignment 2: CT-guided Bone Biopsy Planning

Code for Task 1 (tool calibration) and Task 2 (registration). Tasks 3 and 4 will be
added to this folder.

## Running the code

Needs Python 3 with numpy and matplotlib (`python3 -m pip install numpy matplotlib` if
they are missing). From inside this folder:

```
bash run_tests.sh                       # all tests and the Task 1 noisy calibration, about 1 minute
python3 tests/test_task1_calibration.py # Task 1 with exact marker positions
python3 run_task1_calibration.py        # Task 1 with noisy markers, about 50 s
python3 tests/test_task2_registration.py
```

Each script prints its results and saves them in `outputs/`.

## Files

| File | Contents |
|---|---|
| `config.py` | shared constants: the tool model, the marker noise, the motion settings |
| `geometry.py` | vectors, rotations and 4x4 transformations |
| `frames.py` | the marker frame built from three markers |
| `fitting.py` | least-squares plane fit |
| `noise.py` | Gaussian marker noise and error statistics |
| `simulate.py` | the simulated pivot and spin motions for Task 1 |
| `calibration.py` | Task 1: tool-tip calibration (pivot) and tool-axis calibration (spin) |
| `noisy_calibration.py` | Task 1: repeated calibrations with noisy markers |
| `registration.py` | Task 2: `register_tool`, the tool tip and axis in the CT frame |
| `distance_point_line.py` | distance of a point from a line |
| `intersect_line_sphere.py` | where a line enters and leaves a sphere |
| `biopsy_plan.py` | Task 3: `plan_biopsy`, the targeting error, tissue-core length and insertion depth |
| `figures_task3.py` | the three Task 3 figures (`python3 figures_task3.py`) |
| `figures.py` | the two Task 1 figures |
| `run_task1_calibration.py` | Task 1 noisy results and figures |
| `tests/` | tests of the utilities, the noise level, Task 1, Task 2 and Task 3 (`checks.py` is the shared PASS/FAIL helper) |
| `run_tests.sh` | runs all of the above |

## Conventions

- Lengths are in mm and angles in degrees. The assignment gives the tool model without
  units. We read its numbers as cm, so `config.py` stores them in mm, e.g. M1 = (20, 180, 0).
- Frames: `mod` is the tool model frame F_mod, `tracker` the optical tracker frame (the
  same as F_mod in the Task 1 simulation), `marker` the frame built from M1, M2, M3, and
  `ct` the CT frame.
- `F_a_from_b` is the 4x4 transformation from frame b to frame a.
- Markers are always in the order M1, M2, M3. Marker positions over many poses have the
  shape (n_poses, 3, 3).
- Marker noise: zero-mean Gaussian with sigma = 0.358 mm on each of x, y, z, so that 95% of
  the 3-D marker errors are shorter than 1.0 mm.

## Reusing Task 2 in Tasks 3 and 4

```python
from registration import register_tool
P_ct_mm, V_ct, F_ct_from_mod = register_tool(M1_tool_mm, M2_tool_mm, M3_tool_mm, P_tool_mm, v_tool,
                                             M1_ct_mm, M2_ct_mm, M3_ct_mm)
```

For noisy CT markers, use `noise.add_marker_noise(points_mm, config.MARKER_SIGMA_mm, rng)`.

## Outputs

- `test_utilities.txt`, `test_noise.txt`, `test_task1_calibration.txt`, `test_task2_registration.txt`:
  the test results, with each predicted value next to the computed one.
- `task1_calibration_results.txt`: the Task 1 errors with noisy markers.
- `task1_motions.png`, `task1_motion_sweeps.png`: the Task 1 figures.
- `test_task3_biopsy_plan.txt`: the Task 3 test results.
- `task3_geometry.png`, `task3_test_scenes.png`, `task3_scene_3d.png`: the Task 3 figures.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
