"""临时清理脚本 - 执行后删除"""
with open('main/watermarking/content_chain/low_freq_coder.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []

for i, line in enumerate(lines):
    # 删除块 1：lines[117:875] (行 118-875，1-indexed)
    # 包括：OLD 常量、_normalize_lf_ecc_mode、LowFreqCoder 类、_build_lf_trace_payload
    if 117 <= i <= 874:
        # 在 i==117 处插入 v2 ID 常量
        if i == 117:
            new_lines.append('\n')
            new_lines.append('# 项目内生命名（project-internal naming）\n')
            new_lines.append('LOW_FREQ_TEMPLATE_CODEC_V2_ID = "low_freq_template_codec_v2"\n')
            new_lines.append('LOW_FREQ_TEMPLATE_CODEC_V2_VERSION = "v2"\n')
            new_lines.append('\n')
        continue

    # 删除块 2：lines[1131:1453] (行 1132-1453，1-indexed，class LowFreqTemplateCodecV1)
    if 1131 <= i <= 1452:
        continue

    # 删除块 3：lines[1656:1659] (行 1657-1659，1-indexed，LFCoderPRC 别名及注释)
    if 1656 <= i <= 1658:
        continue

    # 对保留的行做文本替换
    # 1. 修复文件头第 5 行：去掉 PRC-Watermark 引用
    if i == 4:
        line = '- 低频 (LF) 子空间水印编码核心实现。\n'

    # 2. 修复 import 第 24 行：去掉 decode_soft_llr
    if i == 23 and 'decode_soft_llr' in line:
        line = 'from .ldpc_codec import build_ldpc_spec, encode_message_bits\n'

    # 3. 修复 recover_posteriors_erf 文档字符串
    if 'matching PRC-Watermark' in line:
        line = line.replace('matching PRC-Watermark', 'for the pseudogaussian detection mechanism')

    # 4. 修复 V2 类功能注释中的 'PRC v2'
    if 'PRC v2 \u2014\u2014 keyed pseudogaussian template + additive injection \u7f16\u89e3\u7801\u95ed\u73af' in line:
        line = line.replace(
            'PRC v2 \u2014\u2014 keyed pseudogaussian template + additive injection \u7f16\u89e3\u7801\u95ed\u73af',
            'LF \u4f4e\u9891\u6a21\u677f\u7f16\u89e3\u7801 v2 \u2014\u2014 keyed pseudogaussian template + additive injection \u95ed\u73af'
        )

    # 5. 修复 V2 detect_score 功能注释中的 'v2，与 v1 相同算法'
    if '\uff08v2\uff0c\u4e0e v1 \u76f8\u540c\u7b97\u6cd5\uff09' in line:
        line = line.replace('\uff08v2\uff0c\u4e0e v1 \u76f8\u540c\u7b97\u6cd5\uff09', '\uff08v2\uff09')

    new_lines.append(line)

print(f'Original lines: {len(lines)}, New lines: {len(new_lines)}')
print(f'Removed approx {len(lines) - len(new_lines) + 5} lines (net of inserted constants)')

with open('main/watermarking/content_chain/low_freq_coder.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Done! File written.')

# 验证关键检查点
with open('main/watermarking/content_chain/low_freq_coder.py', 'r', encoding='utf-8') as f:
    new_content = f.read()

checks = {
    'LowFreqCoder 已删除': 'class LowFreqCoder:' not in new_content,
    'LowFreqTemplateCodecV1 已删除': 'class LowFreqTemplateCodecV1:' not in new_content,
    'LFCoderPRC 已删除': 'LFCoderPRC = ' not in new_content,
    'LowFreqTemplateCodecV2 保留': 'class LowFreqTemplateCodecV2:' in new_content,
    'LOW_FREQ_CODER_ID 已删除': 'LOW_FREQ_CODER_ID' not in new_content,
    'LOW_FREQ_TEMPLATE_CODEC_V1_ID 已删除': 'LOW_FREQ_TEMPLATE_CODEC_V1_ID' not in new_content,
    'LOW_FREQ_TEMPLATE_CODEC_V2_ID 保留': 'LOW_FREQ_TEMPLATE_CODEC_V2_ID' in new_content,
    'LF_CODER_PRC_ID 已删除': 'LF_CODER_PRC_ID' not in new_content,
    'decode_soft_llr 已删除': 'decode_soft_llr' not in new_content,
    'PRC-Watermark 已删除': 'PRC-Watermark' not in new_content,
    'encode_low_freq_dct 保留': 'def encode_low_freq_dct(' in new_content,
    'compute_lf_attestation_score 保留': 'def compute_lf_attestation_score(' in new_content,
}

all_ok = True
for desc, result in checks.items():
    status = 'OK' if result else 'FAIL'
    if not result:
        all_ok = False
    print(f'  [{status}] {desc}')

print()
print('All checks passed!' if all_ok else 'SOME CHECKS FAILED!')
