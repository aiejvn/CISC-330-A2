"""
figures_task3.py - the Task 3 figures: the biopsy-plan geometry, the three test scenes, and a
3-D view of the test scene.

Every point and length drawn here is computed by the Task 3 code (distance_point_from_line,
intersect_line_sphere, plan_biopsy), so the figures show what the program computes, not a
sketch. The test scene is the one of tests/test_task3_biopsy_plan.py.

Run from the A2 folder:   python3 figures_task3.py      (saves the PNGs in outputs/)

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import config
# Importing figures also sets the shared plot style (Agg backend, rcParams).
from figures import _equal_3d_axes, DARK_BLUE, LIGHT_BLUE, DULL_PURPLE, MED_GRAY, LIGHT_GRAY, DARK_RED
from geometry import normalize
from distance_point_line import distance_point_from_line
from intersect_line_sphere import intersect_line_sphere
from biopsy_plan import plan_biopsy

# The test scene: unit tumor at C, needle along -y with its tip 5 mm above C in y.
# The scene lies in the plane z = 30, so the 2-D figures plot x and y directly.
C_tum_mm, R_tum_mm = np.array([10.0, 20.0, 30.0]), 1.0
v_ct = np.array([0.0, -1.0, 0.0])
TEST_TIPS_mm = {"Case 1: intersects": [10, 25, 30],
                "Case 2: touches": [11, 25, 30],
                "Case 3: misses": [12, 25, 30]}
GEOMETRY_TIP_mm = np.array([10.6, 25.0, 30.0])   # 0.6 mm off center: d = 0.6, h = 0.8 (a 3-4-5 triangle)
NEEDLE_END_T_mm = 7.5                             # how far past the tip the needle line is drawn


def _point(P, v, t):
    """L(t) = P + t v, with v normalized as in the utilities, so t is in mm."""
    return np.asarray(P, dtype=float) + t * normalize(v)


def _draw_tumor_2d(ax):
    """The tumor sphere cut by the plane z = 30: a circle of radius R around C."""
    ax.add_patch(plt.Circle(C_tum_mm[:2], R_tum_mm, facecolor=LIGHT_BLUE, edgecolor=DARK_BLUE, alpha=0.45))
    ax.plot(*C_tum_mm[:2], "+", color=DARK_BLUE, markersize=10)


def _draw_needle_2d(ax, P, t_values):
    """The needle from its tip forward, the tip, and the core (from max(t_in, 0) to t_out) in red."""
    end = _point(P, v_ct, NEEDLE_END_T_mm)
    ax.plot([P[0], end[0]], [P[1], end[1]], color=MED_GRAY, linewidth=1.5, linestyle="--")
    ax.plot(P[0], P[1], "v", color=DULL_PURPLE, markersize=9)
    if len(t_values) == 2:
        core_start, core_end = _point(P, v_ct, max(t_values[0], 0.0)), _point(P, v_ct, t_values[1])
        ax.plot([core_start[0], core_end[0]], [core_start[1], core_end[1]], color=DARK_RED, linewidth=4)
    elif len(t_values) == 1:
        touch = _point(P, v_ct, t_values[0])
        ax.plot(*touch[:2], "o", color=DARK_RED, markersize=6)


def plot_geometry(path):
    """
    The construction of the plan for one off-center needle: the closest point L(t_mid), the
    targeting error d, the half-chord h, the entry and exit points, the core and the depth.
    """
    P = GEOMETRY_TIP_mm
    d = distance_point_from_line(C_tum_mm, P, v_ct)
    t_in, t_out = intersect_line_sphere(P, v_ct, C_tum_mm, R_tum_mm)
    t_mid = (t_in + t_out) / 2                     # the closest point lies halfway between them
    error_mm, core_mm, depth_mm = plan_biopsy(C_tum_mm, R_tum_mm, P, v_ct)
    closest, entry, exit_ = (_point(P, v_ct, t) for t in (t_mid, t_in, t_out))

    fig, ax = plt.subplots(figsize=(6.2, 6.8))
    _draw_tumor_2d(ax)
    _draw_needle_2d(ax, P, [t_in, t_out])
    # Targeting error: the perpendicular from C to the needle axis.
    ax.plot([C_tum_mm[0], closest[0]], [C_tum_mm[1], closest[1]], color=DARK_BLUE, linewidth=1.5)
    ax.text((C_tum_mm[0] + closest[0]) / 2 - 0.05, C_tum_mm[1] + 0.07, f"d = {d:.1f}", color=DARK_BLUE,
            ha="center", va="bottom")
    # Radius to the exit point: the hypotenuse of the right triangle (d, h, R).
    ax.plot([C_tum_mm[0], exit_[0]], [C_tum_mm[1], exit_[1]], color=DARK_BLUE, linewidth=1, linestyle=":")
    ax.text(C_tum_mm[0] - 0.05, (C_tum_mm[1] + exit_[1]) / 2 - 0.15, f"R = {R_tum_mm:g}", color=DARK_BLUE, ha="right")
    # Half-chords on either side of the closest point, labeled just right of the needle.
    for point in (entry, exit_):
        ax.text(closest[0] + 0.08, (closest[1] + point[1]) / 2, f"h = {np.sqrt(R_tum_mm**2 - d**2):.1f}",
                color=DARK_RED, va="center")
    # The named points.
    for point, label in ((closest, "closest point $L(t_{mid})$"), (entry, f"entry $L(t_{{in}})$, $t_{{in}}$ = {t_in:.1f}"),
                         (exit_, f"exit $L(t_{{out}})$, $t_{{out}}$ = {t_out:.1f}")):
        ax.plot(*point[:2], "o", color="black", markersize=4)
        ax.text(point[0] + 0.55, point[1], label, va="center", fontsize=9)
    ax.text(P[0] + 0.12, P[1], "needle tip $P_{ct}$ ($t$ = 0)", va="center", fontsize=9)
    # Insertion depth: from the tip to the exit point, drawn beside the needle.
    x_depth = P[0] - 1.6
    ax.annotate("", xy=(x_depth, exit_[1]), xytext=(x_depth, P[1]),
                arrowprops=dict(arrowstyle="<->", color=DULL_PURPLE))
    ax.text(x_depth - 0.08, (P[1] + exit_[1]) / 2, f"insertion depth\n= $t_{{out}}$ = {depth_mm:.1f}",
            color=DULL_PURPLE, ha="right", va="center", fontsize=9)
    ax.text(exit_[0] + 0.55, exit_[1] - 0.45, f"core = $t_{{out}} - t_{{in}}$ = {core_mm:.1f}", color=DARK_RED,
            fontsize=10, fontweight="bold", va="center")
    ax.annotate("", xy=(P[0], P[1] - 1.0), xytext=(P[0], P[1] - 0.2),
                arrowprops=dict(arrowstyle="->", color=DULL_PURPLE))
    ax.text(P[0] + 0.12, P[1] - 0.6, "$v_{ct}$", color=DULL_PURPLE, va="center")

    ax.set_xlim(7.4, 13.6)
    ax.set_ylim(18.0, 25.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x CT (mm)")
    ax.set_ylabel("y CT (mm)")
    ax.set_title(f"Biopsy plan geometry (targeting error {error_mm:.1f} mm, plane z = 30)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_test_scenes(path):
    """The three test cases side by side, each titled with the values plan_biopsy computes."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
    for ax, (name, P) in zip(axes, TEST_TIPS_mm.items()):
        P = np.asarray(P, dtype=float)
        error_mm, core_mm, depth_mm = plan_biopsy(C_tum_mm, R_tum_mm, P, v_ct)
        _draw_tumor_2d(ax)
        _draw_needle_2d(ax, P, intersect_line_sphere(P, v_ct, C_tum_mm, R_tum_mm))
        depth_text = "none" if depth_mm is None else f"{depth_mm:g} mm"
        ax.set_title(f"{name}\nerror {error_mm:g} mm, core {core_mm:g} mm, depth {depth_text}", fontsize=10)
        ax.set_xlim(8.0, 13.5)
        ax.set_ylim(17.5, 25.8)
        ax.set_aspect("equal")
        ax.set_xlabel("x CT (mm)")
    axes[0].set_ylabel("y CT (mm)")
    # One legend for all panels.
    handles = [plt.Line2D([], [], color=DULL_PURPLE, marker="v", linestyle="", label="needle tip $P_{ct}$"),
               plt.Line2D([], [], color=MED_GRAY, linestyle="--", label="needle axis"),
               plt.Line2D([], [], color=DARK_RED, linewidth=4, label="tissue core"),
               plt.Line2D([], [], color=DARK_RED, marker="o", linestyle="", label="touch point")]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scene_3d(path):
    """
    The unit tumor and the three test needles in the CT frame, drawn with y CT vertical
    (up, as in the assignment's CT frame), so the needles point from top to bottom.
    matplotlib always draws its third axis vertically, so the CT axes are passed in the
    order (z, x, y). This is a cyclic reorder, so the drawn frame stays right-handed.
    """
    PLOT_ORDER = [2, 0, 1]                         # plot axes 1, 2, 3 = CT z, x, y
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(projection="3d")
    # The tumor sphere as a wireframe.
    u, w = np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 16))
    sphere = [C_tum_mm[i] + R_tum_mm * f for i, f in
              enumerate((np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w)))]
    ax.plot_wireframe(*[sphere[i] for i in PLOT_ORDER], color=LIGHT_BLUE, linewidth=0.6)
    colors = [DARK_RED, DULL_PURPLE, MED_GRAY]
    ends = []
    for color, (name, P) in zip(colors, TEST_TIPS_mm.items()):
        P = np.asarray(P, dtype=float)
        end = _point(P, v_ct, NEEDLE_END_T_mm)
        ax.plot(*np.column_stack([P, end])[PLOT_ORDER], color=color, linewidth=2, label=name)
        ax.scatter(*P[PLOT_ORDER], color=color, marker="o", s=40)   # a dot, so it is not read as an arrowhead
        ends += [P, end]
    points = np.vstack([ends, C_tum_mm - R_tum_mm, C_tum_mm + R_tum_mm])
    _equal_3d_axes(ax, points[:, PLOT_ORDER])
    ax.set_xlabel("z CT (mm)")
    ax.set_ylabel("x CT (mm)")
    ax.set_zlabel("y CT (mm)")
    # Look along -z CT, slightly from above, so the sideways x offsets of the needles show.
    ax.view_init(elev=15, azim=-15)
    ax.set_title("Test scene: unit tumor at (10, 20, 30) mm and the three needles\n"
                 "(dots: needle tips at y = 25, needles point down along $-y$)", fontsize=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    for name, plot in (("task3_geometry.png", plot_geometry), ("task3_test_scenes.png", plot_test_scenes),
                       ("task3_scene_3d.png", plot_scene_3d)):
        plot(os.path.join(config.OUTPUT_DIR, name))
        print("saved", os.path.join("outputs", name))
