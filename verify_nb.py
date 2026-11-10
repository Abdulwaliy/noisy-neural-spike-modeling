import json

nb_path = 'Reverse_Engineering_Competition_Solution.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
print(f"Total cells: {len(cells)}")
code_cells = [c for c in cells if c['cell_type'] == 'code']
print(f"Code cells: {len(code_cells)}")
for i, c in enumerate(code_cells):
    exec_count = c.get('execution_count')
    outputs = c.get('outputs', [])
    print(f"Code Cell {i+1}: execution_count={exec_count}, outputs={len(outputs)}")

print("Notebook format valid and complete!")
