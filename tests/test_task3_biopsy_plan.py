"""
Task 3 test: targeting error, tissue-core length and insertion depth of a biopsy plan.

The tumor is a unit sphere at C = (10, 20, 30). The needle points along -y (like the tool
axis) with its tip 5 mm above the tumor center in y, and is moved sideways in x to intersect,
touch and miss the tumor. The expected values are worked out on paper (written below).

Run from the A2 folder:   python3 tests/test_task3_biopsy_plan.py

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checks import CheckList, value_text
from biopsy_plan import plan_biopsy

TOL = 1e-9
C_tum_mm, R_tum_mm = [10, 20, 30], 1.0
v_ct = [0, -1, 0]

TEST_CASES = [
    {"name": "Case 1: needle intersects the tumor, through its center",
     "on_paper": ["Tip P = (10,25,30) is right above C, so the axis passes through C: error 0.",
                  "The needle enters at y = 21 (t = 4) and leaves at y = 19 (t = 6).",
                  "Core = 6 - 4 = 2 (the diameter), depth = 6."],
     "P_ct_mm": [10, 25, 30], "error_mm": 0.0, "core_mm": 2.0, "depth_mm": 6.0},

    {"name": "Case 2: needle touches the tumor",
     "on_paper": ["Tip P = (11,25,30) is 1 mm off C in x, so the axis is R = 1 from C: error 1.",
                  "It touches the sphere only at (11,20,30), at t = 5.",
                  "Core = 0, depth = 5 (to the touch point)."],
     "P_ct_mm": [11, 25, 30], "error_mm": 1.0, "core_mm": 0.0, "depth_mm": 5.0},

    {"name": "Case 3: needle misses the tumor",
     "on_paper": ["Tip P = (12,25,30) is 2 mm off C in x, so the axis is 2 > R from C: error 2.",
                  "The axis never reaches the sphere.",
                  "Core = 0, depth = None (no depth reaches the tumor)."],
     "P_ct_mm": [12, 25, 30], "error_mm": 2.0, "core_mm": 0.0, "depth_mm": None},
]

checks = CheckList("Task 3 test: biopsy plan (targeting error, core length, insertion depth)")
checks.write(f"Tumor: C = {value_text(C_tum_mm)} mm, R = {R_tum_mm:g} mm.  Needle direction v = {value_text(v_ct)}.")
summary_rows = []
for case in TEST_CASES:
    checks.write()
    checks.write(case["name"])
    for line in case["on_paper"]:
        checks.write("  on paper: " + line)
    error_mm, core_mm, depth_mm = plan_biopsy(C_tum_mm, R_tum_mm, case["P_ct_mm"], v_ct)
    checks.check("targeting error", case["error_mm"], error_mm, TOL, "mm")
    checks.check("tissue-core length", case["core_mm"], core_mm, TOL, "mm")
    if case["depth_mm"] is None:
        checks.check_true("insertion depth is None", depth_mm is None, f"computed {depth_mm}")
    else:
        checks.check("insertion depth", case["depth_mm"], depth_mm, TOL, "mm")
    summary_rows.append((case["name"].split(":")[0], case, error_mm, core_mm, depth_mm))

checks.write()
checks.write("Summary: predicted on paper vs computed (mm)")
checks.write(f"  {'case':<8s} {'error pred/comp':<18s} {'core pred/comp':<18s} depth pred/comp")
depth_text = lambda d: "None" if d is None else value_text(d)
for name, case, error_mm, core_mm, depth_mm in summary_rows:
    checks.write(f"  {name:<8s} {value_text(case['error_mm']) + ' / ' + value_text(error_mm):<18s} "
                 f"{value_text(case['core_mm']) + ' / ' + value_text(core_mm):<18s} "
                 f"{depth_text(case['depth_mm'])} / {depth_text(depth_mm)}")

checks.finish("test_task3_biopsy_plan")
