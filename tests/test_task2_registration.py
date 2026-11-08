"""
Task 2 test: registration of the tool tip and tool axis to the CT frame.

The ground-truth tool is placed in three known poses in the CT frame: a pure translation,
a pure rotation, and both. For each pose the CT markers and the expected Pct and Vct are
worked out on paper (written below), and the registration must reproduce them.

Run from the A2 folder:   python3 tests/test_task2_registration.py

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checks import CheckList, value_text
from registration import register_tool
from frames import frame_from_three_points
from geometry import apply_transform, is_rigid

TOL = 1e-9
M1_tool_mm, M2_tool_mm, M3_tool_mm = [2, 18, 0], [8, 18, 0], [2, 24, 0]
P_tool_mm, v_tool = [0, 0, 0], [0, -1, 0]
# In its marker frame (origin (4,20,0), axes parallel to F_mod) the tip is at (-4,-20,0)
# and the axis is (0,-1,0), so Pct = O - 4 i - 20 j and Vct = -j in every case.

TEST_CASES = [
    {"name": "Case 1: pure translation by d = (10, 20, 30)",
     "on_paper": ["Every point moves by d: M1ct = (12,38,30), M2ct = (18,38,30), M3ct = (12,44,30).",
                  "Marker frame in CT: O = (14,40,30), i, j, k = x, y, z.",
                  "Pct = Ptool + d = (10,20,30), and Vct = vtool = (0,-1,0)."],
     "M_ct_mm": [[12, 38, 30], [18, 38, 30], [12, 44, 30]],
     "O": [14, 40, 30], "i": [1, 0, 0], "j": [0, 1, 0], "k": [0, 0, 1],
     "F_ct_from_mod": [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]],
     "P_ct_mm": [10, 20, 30], "V_ct": [0, -1, 0]},

    {"name": "Case 2: pure rotation, 90 deg about z: (x, y, z) -> (-y, x, z)",
     "on_paper": ["M1ct = (-18,2,0), M2ct = (-18,8,0), M3ct = (-24,2,0).",
                  "Marker frame in CT: i = (0,1,0), k = (0,0,1), j = k x i = (-1,0,0), O = (-20,4,0).",
                  "Pct = O - 4 i - 20 j = (0,0,0), and Vct = -j = (1,0,0)."],
     "M_ct_mm": [[-18, 2, 0], [-18, 8, 0], [-24, 2, 0]],
     "O": [-20, 4, 0], "i": [0, 1, 0], "j": [-1, 0, 0], "k": [0, 0, 1],
     "F_ct_from_mod": [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
     "P_ct_mm": [0, 0, 0], "V_ct": [1, 0, 0]},

    {"name": "Case 3: 90 deg about x, then 90 deg about z, then translation by d = (-40, 60, 100)",
     "on_paper": ["The two turns give (x, y, z) -> (z, x, y): M1ct = (-40,62,118), M2ct = (-40,68,118), M3ct = (-40,62,124).",
                  "Marker frame in CT: i = (0,1,0), k = (1,0,0), j = k x i = (0,0,1), O = (-40,64,120).",
                  "Pct = O - 4 i - 20 j = (-40,60,100), and Vct = -j = (0,0,-1)."],
     "M_ct_mm": [[-40, 62, 118], [-40, 68, 118], [-40, 62, 124]],
     "O": [-40, 64, 120], "i": [0, 1, 0], "j": [0, 0, 1], "k": [1, 0, 0],
     "F_ct_from_mod": [[0, 0, 1, -40], [1, 0, 0, 60], [0, 1, 0, 100], [0, 0, 0, 1]],
     "P_ct_mm": [-40, 60, 100], "V_ct": [0, 0, -1]},
]

checks = CheckList("Task 2 test: registration of the tool tip and tool axis to the CT frame")
summary_rows = []
for case in TEST_CASES:
    checks.write()
    checks.write(case["name"])
    for line in case["on_paper"]:
        checks.write("  on paper: " + line)
    M1_ct_mm, M2_ct_mm, M3_ct_mm = case["M_ct_mm"]

    F_ct_from_marker = frame_from_three_points(M1_ct_mm, M2_ct_mm, M3_ct_mm)
    P_ct_mm, V_ct, F_ct_from_mod = register_tool(M1_tool_mm, M2_tool_mm, M3_tool_mm, P_tool_mm, v_tool,
                                                 M1_ct_mm, M2_ct_mm, M3_ct_mm)
    checks.check("marker frame in CT: origin O", case["O"], F_ct_from_marker[:3, 3], TOL, "mm")
    checks.check("marker frame in CT: i", case["i"], F_ct_from_marker[:3, 0], TOL)
    checks.check("marker frame in CT: j", case["j"], F_ct_from_marker[:3, 1], TOL)
    checks.check("marker frame in CT: k", case["k"], F_ct_from_marker[:3, 2], TOL)
    checks.check("F_ct_from_mod (4x4)", case["F_ct_from_mod"], F_ct_from_mod, TOL)
    checks.check("Pct, tool tip in CT", case["P_ct_mm"], P_ct_mm, TOL, "mm")
    checks.check("Vct, tool axis in CT", case["V_ct"], V_ct, TOL, "",
                 f"moving vtool like a point would give {value_text(apply_transform(F_ct_from_mod, v_tool))}")
    markers_mapped_mm = apply_transform(F_ct_from_mod, [M1_tool_mm, M2_tool_mm, M3_tool_mm])
    checks.check("model markers mapped into CT vs M1ct, M2ct, M3ct", 0.0,
                 np.linalg.norm(markers_mapped_mm - np.array(case["M_ct_mm"]), axis=1).max(), TOL, "mm")
    checks.check_true("F_ct_from_mod is rigid", is_rigid(F_ct_from_mod))
    summary_rows.append((case["name"].split(":")[0], case["P_ct_mm"], P_ct_mm, case["V_ct"], V_ct))

checks.write()
checks.write("Summary: predicted on paper vs computed")
checks.write(f"  {'case':<8s} {'Pct predicted (mm)':<20s} {'Pct computed (mm)':<20s} {'Vct predicted':<15s} Vct computed")
for name, P_predicted, P_computed, V_predicted, V_computed in summary_rows:
    checks.write(f"  {name:<8s} {value_text(P_predicted):<20s} {value_text(P_computed):<20s} "
                 f"{value_text(V_predicted):<15s} {value_text(V_computed)}")

checks.finish("test_task2_registration")
