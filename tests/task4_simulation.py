import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import noise
import distance_point_line 
from registration import register_tool



# Monte Carlo simulation for a given target_sigma
def simulate_targeting_error(target_sigma, n_trials=1000, insertion_depth_mm = 100.0, rng=None):
    # Slumptalsgenerator i numpy för att kunna applicera brus
    # rng är en lång lista med randomiserade tal, seed=x bestämmer vilken plats i listan vi börjar hämta tal från
    # target_sigma is the standard deviation (mm) per axis of the tumors displacement
    # n_trials is the number of insertions

    if rng is None:
        rng = np.random.default_rng(config.RANDOM_SEED)

    M1_mod = config.MARKERS_MOD_mm[0]
    M2_mod = config.MARKERS_MOD_mm[1]
    M3_mod = config.MARKERS_MOD_mm[2]
    P_mod = config.TIP_MOD_mm
    v_mod = config.AXIS_MOD

    nominal_markers_ct = config.MARKERS_MOD_mm.copy()
    # This is a reference where the tip is in origo of the CT

    P_ct_true, V_ct_true, _ = register_tool(
        M1_mod, M2_mod, M3_mod, P_mod, v_mod,
        nominal_markers_ct[0], nominal_markers_ct[1], nominal_markers_ct[2]
    )
    v_unit_true = V_ct_true / np.linalg.norm(V_ct_true)
    
    true_target_ct = P_mod + insertion_depth_mm * v_unit_true
    # true_target = [0,0,0] + 100*[0,-1,0] = [0,-100,0] mm

    errors = np.zeros(n_trials)
    # creates a array where all elements (1000) are 0

    for i in range(n_trials):
        # adding the Gaussian noise to M1, M2, M3
        noisy_markers_ct = noise.add_marker_noise(nominal_markers_ct, config.MARKER_SIGMA_mm, rng)
        

        # Re-register the tool based on noise in the markers
        P_ct, V_ct, _ = register_tool(M1_mod, M2_mod, M3_mod, P_mod, v_mod, noisy_markers_ct[0], noisy_markers_ct[1], noisy_markers_ct[2])
        # returns the calculated tip (P_ct) and the axis (V_ct)

        # tumor movement according to target_sigma
        if target_sigma > 0.0:
        # If there is a displacement from true location
            target_displacement = rng.normal(loc=0.0, scale=target_sigma, size=3)
            displaced_target_ct = true_target_ct + target_displacement
            
        else: 
            displaced_target_ct = true_target_ct

        # measure the distance between the line of the tip and the displaced target

        #errors[i] = compute_targeting_error(P_ct, V_ct, displaced_target_ct)
        errors[i] = distance_point_line.distance_point_from_line(displaced_target_ct, P_ct, V_ct)
    return errors


def run_task4_analysis():
    """
    increase target_sigma from 0.0 mm to decide sigma_max before 95th percentile > 3.0 mm
    """
    print("="*70)
    print("CISC/CMPE 330 - Task 4: Compound Targeting Error Simulation")
    print("="*70)
    print(f"Marker noise sigma: {config.MARKER_SIGMA_mm:.3f} mm (95% < 1.0 mm)")
    print(f"Insertion depth:   100.0 mm")
    print(f"Clinical tolerance: 95th percentile <= 3.0 mm")
    print("-"*70)

    rng = np.random.default_rng(config.RANDOM_SEED)

    sigmas_to_test = np.linspace(0.0, 2.0, 21)
    # from 0.0 mm to 2.0 mm in steps of 0.1 mm (21 times)

    sigma_max_allowed = None

    results_table = []

    print(f"{'Target Sigma (mm)':<20} | {'Mean Error (mm)':<18} | {'95th percentile (mm)':<20}")
    print("-"*70)

    for s in sigmas_to_test:
        errs = simulate_targeting_error(target_sigma=s, n_trials=1000, rng=rng)
        mean_err = float(np.mean(errs))
        p95_err = float(np.percentile(errs,95))
        status = "PASS (<= 3.0 mm)" if p95_err <= 3.0 else "FAIL (> 3.0 mm)"
        print(f"{s:<20.2f} | {mean_err:<18.3f} | {p95_err:<20.3f} | {status}")
        results_table.append((s,mean_err, p95_err))
        if p95_err <= 3.0:
            sigma_max_allowed = s
            
    print("-"*70)
    if sigma_max_allowed is not None:
        print(f"Maximal allowed tumor displacement (sigma_max): {sigma_max_allowed:.2f} mm")
    else:
        print(f"The requirement could not be met even at sigma = 0.0 mm.")
    print("="*70)

if __name__=="__main__":
    run_task4_analysis()

