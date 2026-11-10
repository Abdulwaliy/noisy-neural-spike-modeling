import os

files = [
    'Reverse_Engineering_Competition_Solution.ipynb',
    'Reverse_Engineering_Competition_Solution.pdf',
    'team_Abdulwaliy_et_Eleni_reconstruction.npz',
    'team_Abdulwaliy_et_Eleni_generative_seed42.npz',
    'team_reconstruction.npz',
    'evaluate_team.py',
    'README.md',
    'requirements.txt',
    'figures/competition_evaluation_overlay.png'
]

print(f"{'Filename':48s} | {'Size':>10s} | Status")
print("-" * 72)
for f in files:
    if os.path.exists(f):
        sz = os.path.getsize(f)
        print(f"{f:48s} | {sz:10,d} | EXISTS")
    else:
        print(f"{f:48s} | {'MISSING':>10s} | NOT FOUND")
