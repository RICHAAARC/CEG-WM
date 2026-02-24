import json
with open('audit_result.json', 'r', encoding='utf-8-sig') as f:
    data = json.load(f)
summary = data['summary']
print(f"✓ FreezeSignoffDecision: {summary['FreezeSignoffDecision']}")
print(f"✓ RiskSummary: {summary['RiskSummary']}")
print(f"✓ Counts: PASS={summary['counts']['PASS']}, FAIL={summary['counts']['FAIL']}, N.A.={summary['counts'].get('N.A.', 0)}")
print(f"✓ BlockingReasons: {len(summary['BlockingReasons'])}")
