import pypdf

reader = pypdf.PdfReader('Reverse-Engineering Competition_ Inter-Team Evaluation.pdf')
for i, page in enumerate(reader.pages):
    print(f"=== PAGE {i+1} ===")
    print(page.extract_text())
