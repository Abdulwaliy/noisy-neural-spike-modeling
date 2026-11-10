import numpy as np

data = np.load('simulated_spike_data.npz')
print("Keys in simulated_spike_data.npz:", data.files)
for k in data.files:
    v = data[k]
    print(f"  {k:15s}: shape={v.shape}, dtype={v.dtype}, min={v.min():.4f}, max={v.max():.4f}")
