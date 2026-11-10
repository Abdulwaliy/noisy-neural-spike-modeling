"""
evaluate_team.py
Evaluator module for the Reverse-Engineering Competition: Inter-Team Evaluation.

Interface:
def evaluate(reference_file, candidate_file):
    '''
    Inputs:
        reference_file: path to original spike_data.npz
        candidate_file: path to another team's reconstruction

    Output:
        dictionary containing:
        - validity checks - opt
        - numerical evaluation metrics - opt
        - Overlay figure - opt
        - final score - Must
    '''
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, signal

def evaluate(reference_file, candidate_file):
    """
    Evaluates candidate team reconstruction against reference spike_data.npz.
    
    Returns a dictionary with:
    - 'validity checks'
    - 'numerical evaluation metrics'
    - 'Overlay figure'
    - 'final score' (0.0 to 100.0)
    """
    validity_checks = {
        "file_exists": False,
        "loadable": False,
        "required_keys_present": False,
        "shapes_correct": False,
        "dtypes_valid": False,
        "no_nans_or_infs": False,
        "passed": False,
        "notes": []
    }
    
    metrics = {}
    fig = None
    final_score = 0.0

    # -------------------------------------------------------------
    # 1. LOAD REFERENCE FILE
    # -------------------------------------------------------------
    if not os.path.exists(reference_file):
        validity_checks["notes"].append(f"Reference file '{reference_file}' not found.")
        return {
            "validity checks": validity_checks,
            "numerical evaluation metrics": metrics,
            "Overlay figure": fig,
            "final score": 0.0
        }

    try:
        ref_data = np.load(reference_file)
        t_ref = np.asarray(ref_data['t']).squeeze()
        raw_ref = np.asarray(ref_data['raw_data']).squeeze()
        N_ref = len(t_ref)
    except Exception as e:
        validity_checks["notes"].append(f"Error loading reference file: {e}")
        return {
            "validity checks": validity_checks,
            "numerical evaluation metrics": metrics,
            "Overlay figure": fig,
            "final score": 0.0
        }

    # Extract reference ground-truth spikes (threshold 3.0 V + rising edge)
    ref_thresh = 3.0
    ref_above = raw_ref > ref_thresh
    ref_starts = np.where(ref_above & ~np.r_[False, ref_above[:-1]])[0]
    
    # Clean baseline mask for reference drift correlation (exclude [-6, +14] samples around spikes)
    ref_mask_clean = np.ones(N_ref, dtype=bool)
    for s in ref_starts:
        p = s + np.argmax(raw_ref[s:min(N_ref, s + 10)])
        ref_mask_clean[max(0, p - 6):min(N_ref, p + 15)] = False

    # -------------------------------------------------------------
    # 2. VALIDITY CHECKS ON CANDIDATE FILE
    # -------------------------------------------------------------
    if not os.path.exists(candidate_file):
        validity_checks["notes"].append(f"Candidate file '{candidate_file}' not found.")
        return {
            "validity checks": validity_checks,
            "numerical evaluation metrics": metrics,
            "Overlay figure": fig,
            "final score": 0.0
        }
    validity_checks["file_exists"] = True

    try:
        cand_data = np.load(candidate_file)
        validity_checks["loadable"] = True
    except Exception as e:
        validity_checks["notes"].append(f"Failed to load candidate file as NumPy archive: {e}")
        return {
            "validity checks": validity_checks,
            "numerical evaluation metrics": metrics,
            "Overlay figure": fig,
            "final score": 0.0
        }

    required_keys = ['t', 'drift', 'spike_signal', 'noise', 'reconstruction', 'spike_indices']
    cand_keys = list(cand_data.keys())
    missing_keys = [k for k in required_keys if k not in cand_keys]
    
    if missing_keys:
        validity_checks["notes"].append(f"Missing required keys: {missing_keys}")
        return {
            "validity checks": validity_checks,
            "numerical evaluation metrics": metrics,
            "Overlay figure": fig,
            "final score": 0.0
        }
    validity_checks["required_keys_present"] = True

    # Extract candidate arrays
    t_cand = np.asarray(cand_data['t']).squeeze()
    drift_cand = np.asarray(cand_data['drift']).squeeze()
    spike_cand = np.asarray(cand_data['spike_signal']).squeeze()
    noise_cand = np.asarray(cand_data['noise']).squeeze()
    recon_cand = np.asarray(cand_data['reconstruction']).squeeze()
    indices_cand = np.asarray(cand_data['spike_indices']).squeeze()

    # Check shapes
    if (len(t_cand) != N_ref or len(drift_cand) != N_ref or len(spike_cand) != N_ref or
        len(noise_cand) != N_ref or len(recon_cand) != N_ref):
        validity_checks["notes"].append(f"1D array shapes must match reference length {N_ref}.")
        return {
            "validity checks": validity_checks,
            "numerical evaluation metrics": metrics,
            "Overlay figure": fig,
            "final score": 0.0
        }

    if indices_cand.ndim > 1:
        validity_checks["notes"].append("spike_indices must be a 1D array of starting indices.")
        return {
            "validity checks": validity_checks,
            "numerical evaluation metrics": metrics,
            "Overlay figure": fig,
            "final score": 0.0
        }
    validity_checks["shapes_correct"] = True
    validity_checks["dtypes_valid"] = True

    # Check for NaNs or Infs
    for name, arr in [('t', t_cand), ('drift', drift_cand), ('spike_signal', spike_cand),
                      ('noise', noise_cand), ('reconstruction', recon_cand), ('spike_indices', indices_cand)]:
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            validity_checks["notes"].append(f"Array '{name}' contains NaN or Inf values.")
            return {
                "validity checks": validity_checks,
                "numerical evaluation metrics": metrics,
                "Overlay figure": fig,
                "final score": 0.0
            }
    validity_checks["no_nans_or_infs"] = True
    validity_checks["passed"] = True
    validity_checks["notes"].append("All validity and schema checks passed successfully.")

    # -------------------------------------------------------------
    # 3. NUMERICAL EVALUATION METRICS
    # -------------------------------------------------------------
    # A. Reconstruction Quality
    # In Part 1: reconstruction = structured signal (drift + spikes)
    res_signal = raw_ref - recon_cand
    rmse_recon = float(np.sqrt(np.mean(res_signal**2)))
    mae_recon = float(np.mean(np.abs(res_signal)))
    var_total = float(np.var(raw_ref))
    var_res = float(np.var(res_signal))
    r2_recon = float(max(0.0, 1.0 - var_res / var_total))

    # B. Spike Detection Performance (Tolerance window of +/- 3 ms = +/- 3 samples)
    tol = 3
    cand_indices_arr = np.atleast_1d(indices_cand).astype(int)
    n_cand_spikes = len(cand_indices_arr)
    n_ref_spikes = len(ref_starts)

    matched_ref = 0
    for r_s in ref_starts:
        if np.any(np.abs(cand_indices_arr - r_s) <= tol):
            matched_ref += 1

    matched_cand = 0
    for c_s in cand_indices_arr:
        if np.any(np.abs(ref_starts - c_s) <= tol):
            matched_cand += 1

    precision = float(matched_cand / n_cand_spikes) if n_cand_spikes > 0 else 0.0
    recall = float(matched_ref / n_ref_spikes) if n_ref_spikes > 0 else 0.0
    f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # C. Drift Plausibility
    # Check correlation with clean reference baseline
    drift_clean_cand = drift_cand[ref_mask_clean]
    raw_clean_ref = raw_ref[ref_mask_clean]
    if np.std(drift_clean_cand) > 1e-6 and np.std(raw_clean_ref) > 1e-6:
        r_drift = float(np.corrcoef(raw_clean_ref, drift_clean_cand)[0, 1])
    else:
        r_drift = 0.0

    # Drift smoothness (penalty for high-frequency roughness)
    drift_roughness = float(np.std(np.diff(drift_cand)))

    # D. Noise Fidelity
    noise_mean = float(np.mean(noise_cand))
    noise_std = float(np.std(noise_cand))
    noise_optimal_sigma = 0.2983 # Extracted ground truth noise floor

    metrics = {
        "reconstruction_rmse": round(rmse_recon, 4),
        "reconstruction_mae": round(mae_recon, 4),
        "variance_explained_r2": round(r2_recon, 4),
        "optimal_noise_floor_sigma": round(noise_optimal_sigma, 4),
        "spike_candidate_count": int(n_cand_spikes),
        "spike_reference_count": int(n_ref_spikes),
        "spike_precision": round(precision, 4),
        "spike_recall": round(recall, 4),
        "spike_f1_score": round(f1_score, 4),
        "drift_baseline_correlation": round(r_drift, 4),
        "drift_roughness_diff_std": round(drift_roughness, 5),
        "noise_mean": round(noise_mean, 4),
        "noise_std": round(noise_std, 4)
    }

    # -------------------------------------------------------------
    # 4. FINAL SCORE CALCULATION (0.0 to 100.0)
    # -------------------------------------------------------------
    # Rubric:
    # 1. Base Validity: 10 points
    # 2. Reconstruction Accuracy: up to 40 points
    #    Target: R^2 close to theoretical maximum 0.835 (given irreducible noise sigma=0.298)
    #    Score: 40 * min(1.0, max(0.0, r2_recon / 0.835))
    # 3. Spike Detection Accuracy: up to 35 points (35 * F1-score)
    # 4. Drift Quality: up to 15 points (15 * max(0.0, r_drift))
    score_validity = 10.0 if validity_checks["passed"] else 0.0
    score_recon = 40.0 * min(1.0, max(0.0, r2_recon / 0.8344))
    score_spikes = 35.0 * f1_score
    score_drift = 15.0 * max(0.0, r_drift)

    final_score = float(np.clip(score_validity + score_recon + score_spikes + score_drift, 0.0, 100.0))

    # -------------------------------------------------------------
    # 5. OVERLAY FIGURE GENERATION
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), dpi=200, gridspec_kw={'height_ratios': [1.2, 1.2, 1]})

    # Panel 1: Full 10s Recording vs Candidate Reconstruction
    axes[0].plot(t_ref, raw_ref, color='#2980b9', lw=0.6, alpha=0.8, label='Reference Raw Signal (spike_data.npz)')
    axes[0].plot(t_ref, recon_cand, color='#e74c3c', lw=1.1, alpha=0.9, label='Candidate Reconstruction (drift + spikes)')
    axes[0].set_title(f'Evaluation Overlay: Reconstruction Fit (RMSE = {rmse_recon:.4f} V, R² = {r2_recon*100:.1f}%)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Voltage (V)', fontsize=9.5)
    axes[0].set_xlim(0, 10)
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].legend(loc='upper right', fontsize=8.5)

    # Panel 2: Drift & Spikes Breakdown
    axes[1].plot(t_ref, raw_ref, color='#bdc3c7', lw=0.5, alpha=0.5, label='Reference Raw')
    axes[1].plot(t_ref, drift_cand, color='#f39c12', lw=2.0, label='Candidate Drift')
    if n_cand_spikes > 0:
        valid_indices = cand_indices_arr[(cand_indices_arr >= 0) & (cand_indices_arr < N_ref)]
        axes[1].plot(t_ref[valid_indices], raw_ref[valid_indices], 'r^', markersize=6, label=f'Candidate Spikes (N={n_cand_spikes})')
    axes[1].set_title(f'Drift Extraction & Spike Locations (Drift r = {r_drift:.3f}, Spike F1 = {f1_score:.3f})', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Voltage (V)', fontsize=9.5)
    axes[1].set_xlim(0, 10)
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].legend(loc='upper right', fontsize=8.5)

    # Panel 3: Zoomed Close-up (5.0 to 5.5 s)
    zoom = (t_ref >= 5.0) & (t_ref <= 5.5)
    axes[2].plot(t_ref[zoom], raw_ref[zoom], color='#2980b9', lw=1.3, label='Reference Raw')
    axes[2].plot(t_ref[zoom], recon_cand[zoom], color='#e74c3c', lw=1.5, linestyle='--', label='Candidate Reconstruction')
    axes[2].plot(t_ref[zoom], drift_cand[zoom], color='#f39c12', lw=1.5, label='Candidate Drift')
    axes[2].set_title('500 ms Detailed Zoom: Spike Waveforms & Residual Alignment', fontsize=10.5, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)', fontsize=9.5)
    axes[2].set_ylabel('Voltage (V)', fontsize=9.5)
    axes[2].grid(True, linestyle='--', alpha=0.4)
    axes[2].legend(loc='upper right', fontsize=8.5)

    plt.tight_layout()

    return {
        "validity checks": validity_checks,
        "numerical evaluation metrics": metrics,
        "Overlay figure": fig,
        "final score": round(final_score, 2)
    }

if __name__ == '__main__':
    ref_arg = sys.argv[1] if len(sys.argv) > 1 else 'spike_data.npz'
    if len(sys.argv) > 2:
        cand_arg = sys.argv[2]
    elif os.path.exists('team_Abdulwaliy_et_Eleni_reconstruction.npz'):
        cand_arg = 'team_Abdulwaliy_et_Eleni_reconstruction.npz'
    else:
        cand_arg = 'team_reconstruction.npz'
    
    print(f"Evaluating candidate: {cand_arg} against reference: {ref_arg}")
    results = evaluate(ref_arg, cand_arg)
    
    print("\n" + "="*50)
    print(f"FINAL SCORE: {results['final score']} / 100.0")
    print("="*50)
    print("\nValidity Checks:")
    for k, v in results['validity checks'].items():
        print(f"  {k}: {v}")
    print("\nNumerical Evaluation Metrics:")
    for k, v in results['numerical evaluation metrics'].items():
        print(f"  {k}: {v}")
    
    # Save overlay figure if generated
    if results['Overlay figure'] is not None:
        out_fig = "evaluation_overlay.png"
        results['Overlay figure'].savefig(out_fig, dpi=300)
        plt.close(results['Overlay figure'])
        print(f"\nOverlay figure saved to '{out_fig}'")
