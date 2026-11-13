"""
figures.py - the Task 1 figures: the simulated motions, and the errors against the
motion settings.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from geometry import apply_transform

# Tool-tip results are teal, tool-axis results blue, and the 95th percentile dull purple.
DARK_TEAL = "#1f6f6f"
DARK_BLUE, MED_BLUE, LIGHT_BLUE = "#2066a8", "#3594cc", "#8cc5e3"
DULL_PURPLE, MED_GRAY, LIGHT_GRAY = "#5e4c5f", "#999999", "#b8b8b8"
DARK_RED = "#a00000"
MARKER_COLORS = [DARK_BLUE, MED_BLUE, LIGHT_BLUE]          # M1, M2, M3
SHAFT_LENGTH_mm = 300.0                                    # length of the drawn tool shaft

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
                     "legend.frameon": True, "legend.framealpha": 0.85,
                     "legend.facecolor": "white", "legend.edgecolor": "none"})


def _equal_3d_axes(ax, points_mm):
    """Same scale on x, y and z so spheres look round, and axis labels in mm."""
    points_mm = np.asarray(points_mm)
    center = (points_mm.max(axis=0) + points_mm.min(axis=0)) / 2
    half = (points_mm.max(axis=0) - points_mm.min(axis=0)).max() / 2 * 1.05
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x tracker (mm)")
    ax.set_ylabel("y tracker (mm)")
    ax.set_zlabel("z tracker (mm)")


def _shaft_end_points(F_tracker_from_mod):
    """Tip and top of the drawn shaft in every pose, each (n, 3), tracker frame."""
    tips = np.array([apply_transform(F, config.TIP_MOD_mm) for F in F_tracker_from_mod])
    tops = np.array([apply_transform(F, [0.0, SHAFT_LENGTH_mm, 0.0]) for F in F_tracker_from_mod])
    return tips, tops


def plot_motions(F_pivot, markers_pivot_mm, F_spin, markers_spin_mm, path):
    """(a) the pivot motion: marker positions, every 5th shaft and the pivot point; (b) the spin motion."""
    fig = plt.figure(figsize=(12, 5.4))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    tips, tops = _shaft_end_points(F_pivot)
    for marker, name in enumerate(config.MARKER_NAMES):
        radius_mm = np.linalg.norm(config.MARKERS_MOD_mm[marker] - config.TIP_MOD_mm)
        ax.scatter(*markers_pivot_mm[:, marker, :].T, s=9, color=MARKER_COLORS[marker],
                   label=f"{name} (sphere, radius {radius_mm:.1f} mm)")
    for tip, top in zip(tips[::5], tops[::5]):
        ax.plot(*np.column_stack([tip, top]), color=LIGHT_GRAY, linewidth=0.8)
    ax.plot([], [], color=LIGHT_GRAY, linewidth=0.8, label="tool shaft, every 5th pose")
    ax.scatter(*config.PIVOT_POINT_TRACKER_mm, color=DULL_PURPLE, marker="*", s=160, label="pivot point p_h (tool tip)")
    _equal_3d_axes(ax, np.vstack([markers_pivot_mm.reshape(-1, 3), tips, tops]))
    ax.set_title(f"(a) Pivot about the tool tip ({len(F_pivot)} poses)")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    tips, tops = _shaft_end_points(F_spin)
    for marker, name in enumerate(config.MARKER_NAMES):
        radius_mm = np.hypot(config.MARKERS_MOD_mm[marker][0], config.MARKERS_MOD_mm[marker][2])   # distance from the shaft
        ax.scatter(*markers_spin_mm[:, marker, :].T, s=9, color=MARKER_COLORS[marker],
                   label=f"{name} (circle, radius {radius_mm:.0f} mm)")
    ax.plot(*np.column_stack([tips[0], tops[0]]), color=DARK_RED, linewidth=2, label="tool axis (fixed line)")
    ax.scatter(*tips[0], color=DULL_PURPLE, marker="*", s=160, label="tool tip")
    _equal_3d_axes(ax, np.vstack([markers_spin_mm.reshape(-1, 3), tips[:1], tops[:1]]))
    ax.view_init(elev=20, azim=-20)       # side view, so the tilt of the axis shows
    ax.set_title(f"(b) Spin about the tool axis ({len(F_spin)} poses, one full turn)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _sweep_panel(ax, settings, stats_list, default, xlabel, ylabel, title, mean_color):
    """Mean and 95th percentile against one motion setting, with our setting marked."""
    ax.plot(settings, [s["mean"] for s in stats_list], "o-", color=mean_color, linewidth=2, label="mean")
    ax.plot(settings, [s["p95"] for s in stats_list], "s-", color=DULL_PURPLE, linewidth=1.5,
            markersize=5, label="95th percentile")
    ax.axvline(default, color=MED_GRAY, linewidth=1, linestyle=":", label=f"our setting ({default:g})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color=LIGHT_GRAY, alpha=0.4, linewidth=0.6)
    ax.legend()


def plot_motion_sweeps(cone_rows, spin_rows, poses_rows, path):
    """
    Three panels: tip error against the cone half-angle, axis error against the spin range,
    and tip error against the number of poses. Each row is (setting, statistics) or, for
    the poses, (number of poses, tip statistics, axis statistics).
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))

    settings = [row[0] for row in cone_rows]
    _sweep_panel(axes[0], settings, [row[1] for row in cone_rows], config.CONE_HALF_ANGLE_deg,
                 "pivot cone half-angle (deg)", "tool-tip error (mm)", "(a) Tip error vs pivot cone", DARK_TEAL)
    axes[0].set_xticks(settings)
    axes[0].set_ylim(bottom=0)

    settings = [row[0] for row in spin_rows]
    _sweep_panel(axes[1], settings, [row[1] for row in spin_rows], config.SPIN_RANGE_deg,
                 "spin range (deg)", "tool-axis error (deg)", "(b) Axis error vs spin range", DARK_BLUE)
    axes[1].set_yscale("log")
    axes[1].set_xticks(settings)

    settings = [row[0] for row in poses_rows]
    _sweep_panel(axes[2], settings, [row[1] for row in poses_rows], config.N_POSES,
                 "number of poses (pivot and spin)", "tool-tip error (mm)", "(c) Tip error vs number of poses",
                 DARK_TEAL)
    axes[2].set_xscale("log")
    axes[2].set_xticks(settings)
    axes[2].set_xticklabels([str(n) for n in settings])
    axes[2].minorticks_off()
    axes[2].set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
