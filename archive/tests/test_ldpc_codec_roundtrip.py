"""
File purpose: LDPC 编码与软判决译码回归测试。
Module type: General module
"""

from main.watermarking.content_chain.ldpc_codec import (
    build_ldpc_spec,
    decode_soft_llr,
    encode_message_bits,
)


def test_ldpc_codec_roundtrip_outputs_bp_fields() -> None:
    message_bits = [1, -1, 1, -1, 1, -1, 1, -1]
    ldpc_spec = build_ldpc_spec(message_length=len(message_bits), ecc_sparsity=3, seed_key="unit_test_seed")
    encoded_bits = encode_message_bits(message_bits, ldpc_spec)
    llr_values = [float(bit) * 2.5 for bit in encoded_bits]

    decoded = decode_soft_llr(llr_values, ldpc_spec, max_iterations=8)

    assert isinstance(decoded, dict)
    assert len(decoded.get("decoded_bits", [])) == int(ldpc_spec.get("n", 0))
    assert isinstance(decoded.get("bp_converged"), bool)
    assert isinstance(decoded.get("bp_iteration_count"), int)
    assert decoded.get("bp_iteration_count") >= 1
