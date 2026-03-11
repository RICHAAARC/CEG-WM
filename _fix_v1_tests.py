"""临时脚本：批量将测试文件中的 LowFreqCoder V1 引用替换为 LowFreqTemplateCodecV2。"""
import re
import pathlib

def rewrite_lf_coder_test(path):
    content = pathlib.Path(path).read_text(encoding="utf-8")

    # 1. 更新 import
    content = re.sub(
        r"from main\.watermarking\.content_chain\.low_freq_coder import LowFreqCoder",
        "from main.watermarking.content_chain.low_freq_coder import LowFreqTemplateCodecV2, LOW_FREQ_TEMPLATE_CODEC_V2_ID, LOW_FREQ_TEMPLATE_CODEC_V2_VERSION",
        content,
    )

    # 2. 更新类名
    content = content.replace("LowFreqCoder(", "LowFreqTemplateCodecV2(")
    content = content.replace("LowFreqCoder ", "LowFreqTemplateCodecV2 ")  # 注释或类型注解

    # 3. 更新 impl_id 字符串
    content = content.replace('impl_id="low_freq_coder_v1"', "impl_id=LOW_FREQ_TEMPLATE_CODEC_V2_ID")
    content = content.replace("impl_id='low_freq_coder_v1'", "impl_id=LOW_FREQ_TEMPLATE_CODEC_V2_ID")
    content = content.replace('"low_freq_coder_v1"', '"low_freq_template_codec_v2"')
    content = content.replace("'low_freq_coder_v1'", "'low_freq_template_codec_v2'")

    # 4. 更新 impl_version（只替换明确是 V1 的；V2 version = "v2"）
    # 注意：谨慎匹配，避免误替换配置里其他的 v1
    content = re.sub(
        r'impl_version=["\']v1["\']',
        "impl_version=LOW_FREQ_TEMPLATE_CODEC_V2_VERSION",
        content,
    )

    pathlib.Path(path).write_text(content, encoding="utf-8")
    print(f"Updated: {path}")


files = [
    "tests/test_lf_coder_disabled_returns_absent.py",
    "tests/test_lf_coder_failure_semantics.py",
    "tests/test_lf_coder_plan_digest_binding.py",
    "tests/test_lf_coder_plan_digest_mismatch.py",
    "tests/test_lf_parameter_drift_detection.py",
    "tests/test_lf_score_distribution_sensitivity.py",
    "tests/test_parameter_digest_binding_compliance.py",
]

for f in files:
    rewrite_lf_coder_test(f)

print("Done")
