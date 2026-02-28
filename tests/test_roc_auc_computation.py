"""
File purpose: ROC/AUC 计算回归测试。
Module type: General module
"""

from main.evaluation.metrics import compute_auc, compute_roc_curve


def test_compute_roc_curve_and_auc() -> None:
    records = [
        {"content_evidence_payload": {"score": 0.9}, "label": True},
        {"content_evidence_payload": {"score": 0.8}, "label": True},
        {"content_evidence_payload": {"score": 0.3}, "label": False},
        {"content_evidence_payload": {"score": 0.1}, "label": False},
    ]

    fpr_list, tpr_list, thresholds = compute_roc_curve(records)
    auc_value = compute_auc(fpr_list, tpr_list)

    assert len(fpr_list) == len(tpr_list)
    assert len(thresholds) == len(fpr_list)
    assert 0.0 <= auc_value <= 1.0
    assert auc_value >= 0.9
