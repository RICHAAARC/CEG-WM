import json
from pathlib import Path

print("=" * 80)
print("【Detect 摘要缺失诊断】")
print("=" * 80)

# 加载记录
embed_path = Path("run_root/records/embed_record.json")
detect_path = Path("run_root_detect/records/detect_record.json")

with open(embed_path, 'r', encoding='utf-8') as f:
    embed_rec = json.load(f)

with open(detect_path, 'r', encoding='utf-8') as f:
    detect_rec = json.load(f)

# Embed 侧数据
embed_cea = embed_rec.get('content_evidence', {})

# Detect 侧数据（两个可能的字段名）
detect_payload = detect_rec.get('content_evidence_payload', {})

print("\n【Embed 侧 content_evidence 摘要字段】")
print("-" * 80)
embed_fields = {
    'pipeline_fingerprint_digest': embed_cea.get('pipeline_fingerprint_digest'),
    'trajectory_digest': embed_cea.get('trajectory_digest'),
    'trajectory_evidence.digest': embed_cea.get('trajectory_evidence', {}).get('digest'),
    'alignment_digest': embed_cea.get('alignment_digest'),
    'injection_site_digest': embed_cea.get('injection_site_digest'),
}

for field, value in embed_fields.items():
    if value:
        print(f"✅ {field:40} {value[:24]}...")
    else:
        print(f"❌ {field:40} <absent>")

print("\n【Detect 侧 content_evidence_payload 字段】")
print("-" * 80)

if not detect_payload:
    print("❌ content_evidence_payload 为空")
    print(f"\n   Detect Record 顶级字段: {list(detect_rec.keys())}")
else:
    detect_fields = {
        'pipeline_fingerprint_digest': detect_payload.get('pipeline_fingerprint_digest'),
        'trajectory_digest': detect_payload.get('trajectory_digest'),
        'alignment_digest': detect_payload.get('alignment_digest'),
        'injection_site_digest': detect_payload.get('injection_site_digest'),
    }
    
    for field, value in detect_fields.items():
        if value:
            print(f"✅ {field:40} {value[:24]}...")
        else:
            print(f"❌ {field:40} <absent>")

print("\n【对齐验证】")
print("-" * 80)

match_count = 0
for field in ['pipeline_fingerprint_digest', 'trajectory_digest', 'alignment_digest', 'injection_site_digest']:
    embed_val = embed_cea.get(field)
    detect_val = detect_payload.get(field)
    
    if embed_val and detect_val and embed_val == detect_val:
        print(f"✅ {field:40} 完全对齐")
        match_count += 1
    elif embed_val and detect_val:
        print(f"❌ {field:40} 不匹配")
        print(f"   Embed:  {embed_val[:20]}...")
        print(f"   Detect: {detect_val[:20]}...")
    elif embed_val and not detect_val:
        print(f"❌ {field:40} Detect 缺失")
        print(f"   Embed:  {embed_val[:20]}...")
        print(f"   Detect: <absent>")
    else:
        print(f"❌ {field:40} 双方都缺失")

print(f"\n[统计] {match_count}/4 摘要完全对齐")

# 根本原因分析
print("\n【根本原因分析】")
print("-" * 80)

if match_count == 0 and detect_payload:
    print("⚠️  content_evidence_payload 存在但为空")
    print("   → Detect 流程未正确复制 Embed 的摘要字段")
    print("   → 检查: main/cli/run_detect.py 是否导入了摘要字段")
elif not detect_payload:
    print("❌ content_evidence_payload 字段本身不存在")
    print("   → Detect Record 生成时未创建此字段")
    print("   → 这是 detect 记录模式的问题")
    
    # 检查是否有其他字段可能包含这些数据
    print("\n   Detect Record 包含的字段:")
    for key in detect_rec.keys():
        if 'evidence' in key.lower() or 'payload' in key.lower():
            print(f"     • {key}")

print("\n" + "=" * 80)
