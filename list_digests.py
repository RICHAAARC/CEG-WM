import json

rec = json.load(open('audit_evidence/run_root/records/embed_record.json', encoding='utf-8'))
cv = rec.get('content_evidence', {})

print("所有摘要字段:")
for k in sorted(cv.keys()):
    if 'digest' in k.lower():
        v = cv[k]
        if isinstance(v, str):
            print(f"  {k}: {v[:50]}")
        else:
            print(f"  {k}: {type(v).__name__}")
