"""临时清理脚本 - 执行后删除"""
with open('main/watermarking/content_chain/high_freq_embedder.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []

for i, line in enumerate(lines):
    # 删除：老式 HF v1 常量（0-indexed: 21-29）
    # 包括: HIGH_FREQ_EMBEDDER_ID/VERSION/TRACE_VERSION, CONTENT_SCORE_RULE_VERSION, HF_FAILURE_RULE_VERSION
    # 以及: 空行, # 项目内生命名注释, HIGH_FREQ_TEMPLATE_CODEC_V1_ID/VERSION
    if 21 <= i <= 29:
        # 在 i==21 处插入保留的 v2 ID 常量（并保留注释）
        if i == 21:
            new_lines.append('\n')
            new_lines.append('# 项目内生命名（project-internal naming）\n')
            new_lines.append('HIGH_FREQ_TEMPLATE_CODEC_V2_ID = "high_freq_template_codec_v2"\n')
            new_lines.append('HIGH_FREQ_TEMPLATE_CODEC_V2_VERSION = "v2"\n')
        continue

    # 删除：T2SMark 向后兼容别名注释 + 4 个别名（0-indexed: 33-37）
    if 33 <= i <= 37:
        continue

    # 删除：class HighFreqEmbedder（0-indexed: 59-955，行 60-956）
    if 59 <= i <= 955:
        continue

    # 删除：class HighFreqTemplateCodecV1 + HFEmbedderT2SMark 别名（0-indexed: 956-1325，行 957-1326）
    if 956 <= i <= 1325:
        continue

    # 对保留的行做文本替换
    # 1. 修复 V2 类功能注释中的 'T2SMark v2'
    if 'T2SMark v2 \u2014\u2014 keyed Rademacher template' in line:
        line = line.replace(
            'T2SMark v2 \u2014\u2014 keyed Rademacher template',
            'HF \u9ad8\u9891\u6a21\u677f\u7f16\u89e3\u7801 v2 \u2014\u2014 keyed Rademacher template'
        )

    # 2. 修复内部注释/文档中的 T2SMark 引用（模式：T2SMark embed/detect 等）
    if 'T2SMark' in line:
        # 在注释中替换（只替换注释和字符串，保留代码逻辑无关变化）
        line = line.replace('T2SMark embed', 'HF embed v2')
        line = line.replace('T2SMark detect', 'HF detect v2')
        line = line.replace('T2SMark v2', 'HF template codec v2')
        line = line.replace('T2SMark', 'HF template codec')

    new_lines.append(line)

print(f'Original lines: {len(lines)}, New lines: {len(new_lines)}')

with open('main/watermarking/content_chain/high_freq_embedder.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Done! File written.')

with open('main/watermarking/content_chain/high_freq_embedder.py', 'r', encoding='utf-8') as f:
    new_content = f.read()

checks = {
    'HighFreqEmbedder 已删除': 'class HighFreqEmbedder:' not in new_content,
    'HighFreqTemplateCodecV1 已删除': 'class HighFreqTemplateCodecV1:' not in new_content,
    'HFEmbedderT2SMark 已删除': 'HFEmbedderT2SMark = ' not in new_content,
    'HighFreqTemplateCodecV2 保留': 'class HighFreqTemplateCodecV2:' in new_content,
    'HIGH_FREQ_EMBEDDER_ID 已删除': 'HIGH_FREQ_EMBEDDER_ID' not in new_content,
    'HIGH_FREQ_TEMPLATE_CODEC_V1_ID 已删除': 'HIGH_FREQ_TEMPLATE_CODEC_V1_ID' not in new_content,
    'HIGH_FREQ_TEMPLATE_CODEC_V2_ID 保留': 'HIGH_FREQ_TEMPLATE_CODEC_V2_ID' in new_content,
    'HF_EMBEDDER_T2SMARK_ID 已删除': 'HF_EMBEDDER_T2SMARK_ID' not in new_content,
    'T2SMark 外部命名已清除': 'T2SMark' not in new_content,
}

all_ok = True
for desc, result in checks.items():
    status = 'OK' if result else 'FAIL'
    if not result:
        all_ok = False
    print(f'  [{status}] {desc}')

print()
print('All checks passed!' if all_ok else 'SOME CHECKS FAILED!')
