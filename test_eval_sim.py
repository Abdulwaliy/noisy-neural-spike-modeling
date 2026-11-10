import numpy as np
from evaluate_team import evaluate

# Let's inspect what happens if candidate is simulated data (seed 42)
# Load simulated data components
ref = np.load('spike_data.npz')
raw_orig = ref['raw_data']
t = ref['t']
N = len(t)
dt = 0.001

# Run the simulation with seed 42
np.random.seed(42)
rate_orig = 4.2
tau_ref = 0.005
spike_times_sim = []
curr_t = 0.0
while curr_t < 10.0:
    isi = np.random.exponential(1.0 / rate_orig)
    if isi >= tau_ref:
        curr_t += isi
        if curr_t < 10.0 - 0.02:
            spike_times_sim.append(curr_t)

spike_times_sim = np.array(spike_times_sim)
spike_indices_sim = np.round(spike_times_sim / dt).astype(int)

# Drift
A_sine, f_sine, phi_sine, C_sine = 0.6479, 0.0910, 0.2833, 0.0044
drift_sim = A_sine * np.sin(2 * np.pi * f_sine * t + phi_sine) + C_sine

# Noise generated with seed 42
noise_sim = np.random.normal(loc=0.0, scale=0.2983, size=N)

# Waveform
A_exp, tau_exp_sec = 4.955, 0.00285
wf_pre_pts, wf_post_pts = 1, 14
tmpl_pts = np.arange(-wf_pre_pts, wf_post_pts + 1) * dt
spike_tmpl = np.zeros_like(tmpl_pts)
for idx, tau_val in enumerate(tmpl_pts):
    if tau_val >= 0:
        spike_tmpl[idx] = A_exp * np.exp(-tau_val / tau_exp_sec)

spike_signal_sim = np.zeros(N)
for s_idx in spike_indices_sim:
    for k_idx, val in enumerate(spike_tmpl):
        pos = s_idx - wf_pre_pts + k_idx
        if 0 <= pos < N:
            spike_signal_sim[pos] += val

reconstruction_sim = drift_sim + spike_signal_sim

np.savez('test_cand_seed42.npz',
         t=t,
         drift=drift_sim,
         spike_signal=spike_signal_sim,
         noise=noise_sim,
         reconstruction=reconstruction_sim,
         spike_indices=spike_indices_sim)

print("Evaluating test_cand_seed42 against spike_data.npz:")
res = evaluate('spike_data.npz', 'test_cand_seed42.npz')
print("Final score:", res['final score'])
print("Metrics:", res['numerical evaluation metrics'])
