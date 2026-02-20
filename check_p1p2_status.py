"""检查 P1/P2 实际状态"""
import json
from pathlib import Path

rec_path = Path("audit_evidence/run_root/records/embed_record.json")

with open(rec_path, encoding='utf-8') as f:
    rec = json.load(f)

content_ev = rec.get('content_evidence', {})

# 检查管道构建相关的字段
print("=" * 80)
print("Pipeline 状态")
print("=" * 80)
pfp = content_ev.get('pipeline_fingerprint', {})
print(f"pipeline_fingerprint.status: {pfp.get('status')}")
print(f"pipeline_fingerprint.reason: {pfp.get('reason')}")

# 检查推理相关的字段
print("\n" + "=" * 80)
print("Inference 状态")
print("=" * 80)
infer_trace = content_ev.get('infer_trace', {})
print(f"inference_status: {infer_trace.get('inference_status')}")
print(f"inference_error: {infer_trace.get('inference_error')}")
print(f"inference_enabled (from trace): {infer_trace.get('inference_enabled')}")

# 检查 trajectory
print("\n" + "=" * 80)
print("Trajectory 状态")
print("=" * 80)
traj_ev = content_ev.get('trajectory_evidence', {})
print(f"trajectory_status: {traj_ev.get('status')}")
print(f"trajectory_absent_reason: {traj_ev.get('trajectory_absent_reason')}")

# 检查对齐报告中的 P1/P2 结果
print("\n" + "=" * 80)
print("P1/P2 检查结果")
print("=" * 80)
alignment_report = content_ev.get('alignment_report', {})
for check in alignment_report.get('checks', []):
    name = check.get('check_name')
    result = check.get('result')
    if name in ['pipeline_fingerprint_presence', 'trajectory_digest_reproducibility']:
        print(f"{name}: {result}")

# 检查摘要
print("\n" + "=" * 80)
print("摘要字段状态")
print("=" * 80)
print(f"pipeline_fingerprint_digest: {content_ev.get('pipeline_fingerprint_digest', '<absent>')[:16] if content_ev.get('pipeline_fingerprint_digest') else '<absent>'}")
print(f"trajectory_digest: {content_ev.get('trajectory_digest', '<absent>')[:16] if content_ev.get('trajectory_digest') else '<absent>'}")
print(f"injection_site_digest: {content_ev.get('injection_site_digest', '<absent>')[:16] if content_ev.get('injection_site_digest') else '<absent>'}")
