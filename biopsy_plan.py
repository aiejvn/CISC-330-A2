"""
biopsy_plan.py - Task 3: the salient parameters of a biopsy plan.

The tumor is a sphere (C_tum, R_tum) and the needle is the line L(t) = P_ct + t v_ct, all in
the CT frame. The needle is pushed forward along its current trajectory (t >= 0) from the tip.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
from distance_point_line import distance_point_from_line
from intersect_line_sphere import intersect_line_sphere


def plan_biopsy(C_tum_mm, R_tum_mm, P_ct_mm, v_ct):
    """
    Returns (targeting_error_mm, core_length_mm, insertion_depth_mm).
    - Targeting error: distance of the needle axis from the tumor center.
    - Core length: how far the needle is submerged in the tumor, i.e. the length of its
      forward path inside the sphere, from max(t_in, 0) to t_out.
    - Insertion depth: distance from the tip to where the needle leaves the tumor (t_out),
      which harvests the whole core.
    Edge cases:
    - Tangent: core 0, depth = distance to the touch point.
    - Miss, or tumor entirely behind the tip: core 0, depth None (no depth reaches the tumor).
    - Tip already inside the tumor: the core counts only from the tip (t = 0) to t_out.
    """
    targeting_error_mm = distance_point_from_line(C_tum_mm, P_ct_mm, v_ct)
    t_values = intersect_line_sphere(P_ct_mm, v_ct, C_tum_mm, R_tum_mm)
    if not t_values or t_values[-1] < 0:
        return targeting_error_mm, 0.0, None
    if len(t_values) == 1:
        return targeting_error_mm, 0.0, t_values[0]
    t_in, t_out = t_values
    return targeting_error_mm, t_out - max(t_in, 0.0), t_out
