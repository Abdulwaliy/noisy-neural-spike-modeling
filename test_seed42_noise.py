import numpy as np
from evaluate_team import evaluate

ref = np.load('spike_data.npz')
t = ref['t']
raw = ref['raw_data']
N = len(t)
dt = float(np.median(np.diff(t)))

# Detect spikes in reference
detect_thresh = 3.0
above = raw > detect_thresh
spike_indices = np.where(above & ~np.r_[False, above[:-1]])[0]
peaks = [s + np.argmax(raw[s:min(N, s+10)]) for s in spike_indices]

# Drift
from scipy import optimize
mask_clean = np.ones(N, dtype=bool)
for p in peaks:
    mask_clean[max(0, p-6):min(N, p+15)] = False

def sine_model(t_arr, A, f, phi, C):
    return A * np.sin(2 * np.pi * f * t_arr + phi) + C

popt_sine, _ = optimize.curve_fit(sine_model, t[mask_clean], raw[mask_clean], p0=[0.65, 0.09, 0.3, 0.0])
drift = sine_model(t, *popt_sine)

# Single spike exp decay
pre, post = 4, 14
aligned = np.array([raw[p - pre : p + post + 1] for p in peaks])
mean_wf = np.mean(aligned, axis=0)
wf_t_ms = np.arange(-pre, post + 1) * (dt * 1000.0)

def exp_decay_model(t_val, A, tau, C):
    return A * np.exp(-t_val / tau) + C

popt_exp, _ = optimize.curve_fit(exp_decay_model, wf_t_ms[wf_t_ms >= 0], mean_wf[wf_t_ms >= 0], p0=[5.0, 3.0, 0.0])
A_exp, tau_exp_ms, C_exp = popt_exp
tau_exp_sec = tau_exp_ms / 1000.0

template_pts = np.arange(-1, 15) * dt
tmpl = np.zeros_like(template_pts)
for idx, tt in enumerate(template_pts):
    if tt >= 0:
        tmpl[idx] = A_exp * np.exp(-tt / tau_exp_sec)

spike_signal = np.zeros(N)
for p in peaks:
    for k_idx, val in enumerate(tmpl):
        pos = p - 1 + k_idx
        if 0 <= pos < N:
            spike_signal[pos] += val

reconstruction = drift + spike_signal

# Generated noise with random seed 42!
np.random.seed(42)
sigma_noise = float(np.std(raw[mask_clean] - drift[mask_clean]))
noise_gen42 = np.random.normal(loc=0.0, scale=sigma_noise, size=N)

# Save test file
test_fn = 'team_Abdulwaliy_et_Eleni_reconstruction.npz'
np.savez(test_fn,
         t=t,
         drift=drift,
         spike_signal=spike_signal,
         noise=noise_gen42,
         reconstruction=reconstruction,
         spike_indices=spike_indices)

eval_res = evaluate('spike_data.npz', test_fn)
print("Score with generated noise (seed 42):", eval_res['final score'])
print("Noise mean:", eval_res['numerical evaluation metrics']['noise_mean'])
print("Noise std:", eval_res['numerical evaluation metrics']['noise_std'])
