"""
registration.py - Task 2: register the tool tip and tool axis to the CT frame.

The three markers define the same marker frame in the tool model frame and in the CT
frame. Going from F_mod into the marker frame and from there into F_ct gives the
registration F_ct_from_mod = F_ct_from_marker F_marker_from_mod.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
from geometry import invert_rigid, apply_transform, transform_direction
from frames import frame_from_three_points


def rigid_transform_from_markers(markers_a_mm, markers_b_mm):
    """
    F_b_from_a, the rigid transformation that maps three markers given in frame a onto the
    same markers given in frame b. Both inputs are (3, 3), rows M1, M2, M3.
    """
    F_marker_from_a = invert_rigid(frame_from_three_points(*markers_a_mm))
    F_b_from_marker = frame_from_three_points(*markers_b_mm)
    return F_b_from_marker @ F_marker_from_a


def register_tool(M1_tool_mm, M2_tool_mm, M3_tool_mm, P_tool_mm, v_tool, M1_ct_mm, M2_ct_mm, M3_ct_mm):
    """
    Tool tip and tool axis in the CT frame.
    Inputs: the three markers, the tip Ptool and the axis vtool in the tool model frame,
    and the same three markers in the CT frame (all (3,), mm).
    Returns (P_ct_mm, V_ct, F_ct_from_mod). The tip is a point, so the whole transformation
    acts on it. The axis is a direction, so only the rotation part acts on it.
    """
    F_ct_from_mod = rigid_transform_from_markers([M1_tool_mm, M2_tool_mm, M3_tool_mm],
                                                 [M1_ct_mm, M2_ct_mm, M3_ct_mm])
    P_ct_mm = apply_transform(F_ct_from_mod, P_tool_mm)
    V_ct = transform_direction(F_ct_from_mod, v_tool)
    return P_ct_mm, V_ct, F_ct_from_mod
