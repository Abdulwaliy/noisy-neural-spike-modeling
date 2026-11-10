import re
import zlib

with open('Reverse-Engineering Competition_ Inter-Team Evaluation.pdf', 'rb') as f:
    pdf = f.read()

font_to_unicode = {
    'F4': '117',
    'F5': '121',
    'F6': '125',
    'F7': '129',
    'F8': '133',
    'F9': '137',
    'F12': '141',
}

cmap_mappings = {}
for fname, cmap_id in font_to_unicode.items():
    obj_pattern = rb'\b' + cmap_id.encode() + rb'\s+0\s+obj.*?stream[\r\n]+(.*?)[\r\n]+endstream'
    cmap_obj = re.search(obj_pattern, pdf, re.DOTALL)
    if cmap_obj:
        stream = zlib.decompress(cmap_obj.group(1)).decode('latin1', errors='ignore')
        mapping = {}
        for bfchar in re.finditer(r'<([0-9a-fA-F]+)>\s+<([0-9a-fA-F]+)>', stream):
            src = int(bfchar.group(1), 16)
            dst_hex = bfchar.group(2)
            dst_chars = "".join([chr(int(dst_hex[i:i+4], 16)) for i in range(0, len(dst_hex), 4)])
            mapping[src] = dst_chars
        for bfrange in re.finditer(r'<([0-9a-fA-F]+)>\s+<([0-9a-fA-F]+)>\s+<([0-9a-fA-F]+)>', stream):
            src_start = int(bfrange.group(1), 16)
            src_end = int(bfrange.group(2), 16)
            dst_start = int(bfrange.group(3), 16)
            for offset in range(src_end - src_start + 1):
                mapping[src_start + offset] = chr(dst_start + offset)
        cmap_mappings[fname] = mapping

# Decode pages
for c_id, p_num in [('10', 1), ('13', 2)]:
    pattern = rb'\b' + c_id.encode() + rb'\s+0\s+obj.*?stream[\r\n]+(.*?)[\r\n]+endstream'
    match = re.search(pattern, pdf, re.DOTALL)
    c_stream = zlib.decompress(match.group(1)).decode('latin1', errors='ignore')
    
    curr_font = 'F5'
    page_text = []
    
    # We parse instructions
    # /F... 12 Tf
    # [ ... ] TJ
    # <...> Tj
    for line in c_stream.split('\n'):
        # font switch
        fmatch = re.search(r'/(F\d+)\s+[\d\.]+\s+Tf', line)
        if fmatch:
            curr_font = fmatch.group(1)
        
        # TJ or Tj
        tj_match = re.findall(r'<([0-9a-fA-F]+)>', line)
        if tj_match and ('Tj' in line or 'TJ' in line):
            line_str = []
            for h in tj_match:
                for i in range(0, len(h), 4):
                    code = int(h[i:i+4], 16)
                    line_str.append(cmap_mappings.get(curr_font, {}).get(code, ' '))
            page_text.append("".join(line_str))
        elif 'T*' in line or "Tj" in line:
            pass

    full_p = "\n".join(page_text)
    with open(f'page_{p_num}_text.txt', 'w', encoding='utf-8') as out:
        out.write(full_p)
    print(f"Wrote page_{p_num}_text.txt successfully ({len(full_p)} chars)")
