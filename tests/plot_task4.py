import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import noise
from registration import register_tool
import task4_simulation


def generate_task4_plot(output_dir=None):
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(config.RANDOM_SEED)

    # Sweep over target_sigma (0.0 to 2.0 mm with 21 steps)
    sigmas_to_test = np.linspace(0.0, 2.0, 21)
    mean_errors = []
    p95_errors = []

    print("Simulating the plot...")
    for s in sigmas_to_test:
        errs = task4_simulation.simulate_targeting_error(target_sigma=s, n_trials=1000, rng=rng)
        mean_errors.append(np.mean(errs))
        p95_errors.append(np.percentile(errs, 95))

    # --- Creating the graph ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(sigmas_to_test, p95_errors, color='#3f3f3f', marker='s', markersize=4,
            linewidth=1.3, label=r'95th percentile ($d_{95}$)')
    ax.plot(sigmas_to_test, mean_errors, color='#1b7837', marker='o', markersize=4,
            linewidth=1.3, label='mean error')

    # Clinical tolerance (3.0 mm)
    ax.axhline(3.0, color='#b2182b', linestyle='--', linewidth=1.2, label='clinical tolerance (3.0 mm)')

    
    ax.set_xlabel(r'target displacement standard deviation $\sigma_{\mathrm{target}}$ (mm)', fontsize=9.5)
    ax.set_ylabel('targeting error at 100 mm depth (mm)', fontsize=9.5)
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(0.0, 10.0)


    ax.grid(True, linestyle=':', linewidth=0.6, color='#bbbbbb', alpha=0.8)
    ax.tick_params(direction='in', length=3.5, width=0.8, labelsize=9)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='#d0d0d0', fontsize=8.5)

    
    ax.annotate(
        rf'$d_{{95}} = {p95_errors[0]:.2f}$ mm',
        xy=(0.0, p95_errors[0]),
        xytext=(0.25, p95_errors[0] + 1.3),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9),
        fontsize=8.5,
        color="#222222"
    )

    plot_path = os.path.join(output_dir, "Figure_6.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()

    print(f"Saving the plot to: {plot_path}")


if __name__ == "__main__":
    generate_task4_plot()
