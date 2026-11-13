import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Se till att moduler från rotmappen hittas
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import noise
from registration import register_tool


def compute_targeting_error(P_ct, V_ct, target_pt):
    """Vinkelrätt avstånd från det förskjutna målet till nållinjen."""
    v_unit = V_ct / np.linalg.norm(V_ct)
    diff = target_pt - P_ct
    return float(np.linalg.norm(np.cross(diff, v_unit)))


def simulate_targeting_error(target_sigma, n_trials=1000, insertion_depth_mm=100.0, rng=None):
    """Monte Carlo-simulering för ett givet target_sigma."""
    if rng is None:
        rng = np.random.default_rng(config.RANDOM_SEED)

    M1_mod = config.MARKERS_MOD_mm[0]
    M2_mod = config.MARKERS_MOD_mm[1]
    M3_mod = config.MARKERS_MOD_mm[2]
    P_mod = config.TIP_MOD_mm
    v_mod = config.AXIS_MOD

    nominal_markers_ct = config.MARKERS_MOD_mm.copy()
    true_target_ct = P_mod + insertion_depth_mm * v_mod

    errors = np.zeros(n_trials)

    for i in range(n_trials):
        # 1. Mätbrus på CT-markörerna
        noisy_markers_ct = noise.add_marker_noise(nominal_markers_ct, config.MARKER_SIGMA_mm, rng)

        # 2. Registrering med brus
        P_ct, V_ct, _ = register_tool(
            M1_mod, M2_mod, M3_mod, P_mod, v_mod,
            noisy_markers_ct[0], noisy_markers_ct[1], noisy_markers_ct[2]
        )

        # 3. Slumpmässig målförskjutning
        if target_sigma > 0.0:
            target_disp = rng.normal(loc=0.0, scale=target_sigma, size=3)
            displaced_target_ct = true_target_ct + target_disp
        else:
            displaced_target_ct = true_target_ct

        # 4. Beräkna siktfelet
        errors[i] = compute_targeting_error(P_ct, V_ct, displaced_target_ct)

    return errors


def generate_task4_plot(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(config.RANDOM_SEED)

    # Svep över target_sigma (0.0 till 2.0 mm med 21 steg)
    sigmas_to_test = np.linspace(0.0, 2.0, 21)
    mean_errors = []
    p95_errors = []

    print("Kör simulering för plotten...")
    for s in sigmas_to_test:
        errs = simulate_targeting_error(target_sigma=s, n_trials=1000, rng=rng)
        mean_errors.append(np.mean(errs))
        p95_errors.append(np.percentile(errs, 95))

    # --- Skapa grafen i rapportstil ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Färger och markörer som matchar figurerna i rapporten (mörkgrön för mean, skiffergrå för 95th)
    ax.plot(sigmas_to_test, p95_errors, color='#3f3f3f', marker='s', markersize=4,
            linewidth=1.3, label=r'95th percentile ($d_{95}$)')
    ax.plot(sigmas_to_test, mean_errors, color='#1b7837', marker='o', markersize=4,
            linewidth=1.3, label='mean error')

    # Klinisk toleransgräns (3.0 mm)
    ax.axhline(3.0, color='#b2182b', linestyle='--', linewidth=1.2, label='clinical tolerance (3.0 mm)')

    # Axelinställningar och text
    ax.set_xlabel(r'target displacement standard deviation $\sigma_{\mathrm{target}}$ (mm)', fontsize=9.5)
    ax.set_ylabel('targeting error at 100 mm depth (mm)', fontsize=9.5)
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(0.0, 32.0)

    # Rent och diskret rutnät
    ax.grid(True, linestyle=':', linewidth=0.6, color='#bbbbbb', alpha=0.8)
    ax.tick_params(direction='in', length=3.5, width=0.8, labelsize=9)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='#d0d0d0', fontsize=8.5)

    # Diskret pil och text för felet vid sigma = 0
    ax.annotate(
        r'$d_{95} \approx 25.0$ mm',
        xy=(0.0, p95_errors[0]),
        xytext=(0.25, 22.0),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9),
        fontsize=8.5,
        color="#222222"
    )

    plot_path = os.path.join(output_dir, "task4_targeting_error_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()

    print(f"Plot sparad till: {plot_path}")


if __name__ == "__main__":
    generate_task4_plot()