def format_extracted(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\r\n') for line in f]
    
    # join single characters or small fragments into words/sentences
    out = ""
    for l in lines:
        if not l:
            if not out.endswith("\n\n"):
                out += "\n"
        elif len(l) == 1 or len(l) <= 3:
            out += l
        else:
            out += " " + l
    return out

with open('clean_competition_text.txt', 'w', encoding='utf-8') as f:
    f.write("=== PAGE 1 ===\n")
    f.write(format_extracted('page_1_text.txt'))
    f.write("\n\n=== PAGE 2 ===\n")
    f.write(format_extracted('page_2_text.txt'))

print("Formatted text saved to clean_competition_text.txt")
