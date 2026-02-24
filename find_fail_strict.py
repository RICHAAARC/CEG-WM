import json
with open('audit_result_strict.json', 'r', encoding='utf-8-sig') as f:
    content = f.read()
# 清理控制字符
content = ''.join(ch for ch in content if ord(ch) >= 32 or ch in '\n\r\t')
try:
    data = json.loads(content)
    for result in data.get('results', []):
        if result.get('result') == 'FAIL':
            print(f"FAIL audit: {result.get('audit_id')}, severity: {result.get('severity')}")
except:
    # 如果 JSON 解析失败，尝试直接查看摘要
    if 'summary' in content:
        print("Summary found but JSON parsing failed")
