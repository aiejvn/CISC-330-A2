"""
biopsy_plan.py - Task 3: the salient parameters of a biopsy plan.

The physician uses these three numbers to judge the current needle trajectory: accept it,
or adjust the tool, re-image and plan again (iterative forward planning).

The tumor is a sphere (C_tum, R_tum) and the needle is the line L(t) = P_ct + t v_ct, all in
the CT frame (P_ct and v_ct come from the Task 2 registration). The plan pushes the needle
forward along its current trajectory from the tip, so only t >= 0 can be reached.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
from distance_point_line import distance_point_from_line
from intersect_line_sphere import intersect_line_sphere


def plan_biopsy(C_tum_mm, R_tum_mm, P_ct_mm, v_ct):
    """
    Returns (targeting_error_mm, core_length_mm, insertion_depth_mm).
    - Targeting error: distance of the needle axis from the tumor center. 0 means the needle
      is aimed straight at the center.
    - Core length: how far the needle is submerged in the tumor, i.e. the length of its
      forward path inside the sphere, from max(t_in, 0) to t_out.
    - Insertion depth: distance from the tip to where the needle leaves the tumor (t_out).
      Pushing the needle this far harvests the whole core.
    Edge cases:
    - Miss (error > R), or tumor entirely behind the tip: core 0, depth None, because no
      forward depth reaches the tumor.
    - Tangent (error = R): core 0, depth = distance to the touch point.
    - Tip already inside the tumor: the core counts only from the tip (t = 0) to t_out.
    """
    # Targeting error: the needle axis is a line, the tumor center a point.
    targeting_error_mm = distance_point_from_line(C_tum_mm, P_ct_mm, v_ct)
    # Where the needle axis meets the tumor, as distances (mm) from the tip along the needle.
    t_values = intersect_line_sphere(P_ct_mm, v_ct, C_tum_mm, R_tum_mm)
    # Miss, or the whole tumor is behind the tip (even the exit point has t < 0).
    if not t_values or t_values[-1] < 0:
        return targeting_error_mm, 0.0, None
    # Tangent: one touch point, no tissue inside the needle path.
    if len(t_values) == 1:
        return targeting_error_mm, 0.0, t_values[0]
    # Through the tumor: the core runs from the entry point (or the tip, if the tip is
    # already inside, t_in < 0) to the exit point, and the needle must reach the exit point.
    t_in, t_out = t_values
    return targeting_error_mm, t_out - max(t_in, 0.0), t_out
