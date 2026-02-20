import json
from pathlib import Path

embed_path = Path("run_root/records/embed_record.json")
detect_path = Path("run_root_detect/records/detect_record.json")

print("=" * 80)
print("【检查报告】 P1/P2 问题解决验证")
print("=" * 80)

# 加载 embed record
with open(embed_path, 'r', encoding='utf-8') as f:
    embed_rec = json.load(f)

align = embed_rec.get('content_evidence', {}).get('alignment_report', {})
content_ev = embed_rec.get('content_evidence', {})

print("\n【1】Paper Faithfulness 整体对齐状态")
print("-" * 80)
print(f"Overall Status: {align.get('overall_status')}")

print("\n【2】对齐检查结果")
print("-" * 80)
checks = align.get('checks', [])
for c in checks:
    print(f"  • {c.get('check_name'):40} → {c.get('result')}")

print("\n【3】P1 Pipeline Fingerprint 状态")
print("-" * 80)
pfp = content_ev.get('pipeline_fingerprint', {})
print(f"  Status: {pfp.get('status')}")
print(f"  Reason: {pfp.get('reason')}")
print(f"  Digest: {content_ev.get('pipeline_fingerprint_digest', '<absent>')[:24]}...")

# 查找 P1 检查
p1_check = next((c for c in checks if c.get('check_name') == 'pipeline_fingerprint_presence'), None)
if p1_check:
    print(f"  P1 Result: {p1_check.get('result')} ({'✅' if p1_check.get('result') == 'PASS' else '⚠️'})")

print("\n【4】P2 Trajectory Evidence 状态")
print("-" * 80)
traj = content_ev.get('trajectory_evidence', {})
print(f"  Status: {traj.get('status')}")
print(f"  Absent Reason: {traj.get('trajectory_absent_reason')}")
print(f"  Digest: {content_ev.get('trajectory_digest', '<absent>')[:24]}...")

# 查找 P2 检查
p2_check = next((c for c in checks if c.get('check_name') == 'trajectory_digest_reproducibility'), None)
if p2_check:
    print(f"  P2 Result: {p2_check.get('result')} ({'✅' if p2_check.get('result') == 'PASS' else '⚠️'})")

# 加载 detect record
print("\n【5】Detect 记录中的一致性检查")
print("-" * 80)
if detect_path.exists():
    with open(detect_path, 'r', encoding='utf-8') as f:
        detect_rec = json.load(f)
    
    pf_detect = detect_rec.get('paper_faithfulness', {})
    print(f"  paper_faithfulness Status: {pf_detect.get('status')}")
    print(f"  Mismatch Reasons: {pf_detect.get('mismatch_reasons', [])}")
else:
    print(f"  ❌ detect_record.json 未找到")

print("\n【6】最终验收:")
print("-" * 80)
summary_path = Path("alignment_acceptance_summary.json")
if summary_path.exists():
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print(f"  Final Verdict: {summary.get('final_verdict')} ({'✅' if summary.get('final_verdict') == 'ACCEPT' else '❌'})")
    print(f"  Embed Alignment: {summary.get('embed_status', {}).get('alignment_overall_status')}")
    print(f"  Detect PF Status: {summary.get('detect_status', {}).get('paper_faithfulness_status')}")
    print(f"  Pytest Status: {summary.get('pytest_status', {}).get('passed')} passed, {summary.get('pytest_status', {}).get('failed')} failed")
    print(f"  Audit Decision: {summary.get('audit_status', {}).get('freeze_decision')}")

print("\n" + "=" * 80)
