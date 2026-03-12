import re

with open(r'd:\Code\CEG-WM\main\watermarking\detect\orchestrator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 精确找到重复行：第一行末尾带有乱码注释，后跟正常的相同行
pattern = r'        "content_evidence_payload": content_evidence_payload,[^\n]+\n        "content_evidence_payload": content_evidence_payload,'
m = re.search(pattern, content)
if m:
    print("Found at:", m.start(), m.end())
    print("Match:", repr(m.group()))
    content = content[:m.start()] + '        "content_evidence_payload": content_evidence_payload,' + content[m.end():]
    with open(r'd:\Code\CEG-WM\main\watermarking\detect\orchestrator.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed successfully")
else:
    print("Pattern not found")
