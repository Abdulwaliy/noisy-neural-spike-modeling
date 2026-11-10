import re
with open('Reverse_Engineering_Competition_Solution.pdf', 'rb') as f:
    pdf = f.read()
pages = re.findall(rb'/Type\s*/Page\b', pdf)
print(f"Page count: {len(pages)}")
