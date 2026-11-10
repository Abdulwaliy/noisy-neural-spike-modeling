import numpy as np

ref = np.load('spike_data.npz')
raw = ref['raw_data']
t = ref['t']

# Original spike peaks and starts
above = raw > 3.0
starts = np.where(above & ~np.r_[False, above[:-1]])[0]
peaks = [s + np.argmax(raw[s:min(len(raw), s+10)]) for s in starts]
print(f"Reference: {len(starts)} spikes, starts: {starts[:10]}")

# Test 1: np.random.seed(42) with exponential ISI
np.random.seed(42)
times_test1 = []
curr = 0.0
while curr < 10.0:
    isi = np.random.exponential(1.0 / 4.2)
    curr += isi
    if curr < 10.0:
        times_test1.append(int(round(curr * 1000)))
print(f"Test 1 (np.random.seed(42), no ref): {len(times_test1)} spikes, idx: {times_test1[:10]}")

# Test 2: np.random.default_rng(42)
rng = np.random.default_rng(42)
times_test2 = []
curr = 0.0
while curr < 10.0:
    isi = rng.exponential(1.0 / 4.2)
    curr += isi
    if curr < 10.0:
        times_test2.append(int(round(curr * 1000)))
print(f"Test 2 (default_rng(42)): {len(times_test2)} spikes, idx: {times_test2[:10]}")

# Test 3: Poisson event per millisecond with p = 4.2 / 1000
np.random.seed(42)
spikes_bern = np.where(np.random.rand(10000) < (4.2/1000))[0]
print(f"Test 3 (Bernoulli/Poisson rand): {len(spikes_bern)} spikes, idx: {spikes_bern[:10]}")

# Test 4: default_rng(42) Poisson
rng = np.random.default_rng(42)
spikes_bern2 = np.where(rng.random(10000) < (4.2/1000))[0]
print(f"Test 4 (default_rng Bernoulli): {len(spikes_bern2)} spikes, idx: {spikes_bern2[:10]}")
