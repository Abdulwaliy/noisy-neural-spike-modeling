import numpy as np

files = [
    'team_Abdulwaliy_et_Eleni_reconstruction.npz',
    'team_Abdulwaliy_et_Eleni_generative_seed42.npz',
    'team_reconstruction.npz'
]

for fn in files:
    data = np.load(fn)
    print(f"=== {fn} ===")
    for k in sorted(data.files):
        v = data[k]
        print(f"  {k:15s}: shape={v.shape}, dtype={v.dtype}, min={v.min():.4f}, max={v.max():.4f}, finite={np.all(np.isfinite(v))}")
