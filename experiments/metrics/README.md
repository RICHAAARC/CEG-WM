# Experiment Metrics

`internal.py` 实现只依赖 `experiments/protocol/` 的内部验证指标：

- content-threshold-fit primary null 上的 fixed-FPR threshold，以及冻结阈值下
  registered TPR、primary-null FPR 和独立 wrong-key positive rate；
- matched-budget relative L2/MSE、routing detection gain 与 quality
  non-degradation；
- LF/HF complementarity、combined gain/regression 和 wrong-key behavior；
- rotation/scale/translation error、coverage、residual；
- geometry reliability accept/reject、同 detector/threshold rectification delta；
- raw/rescue/global FPR safety。

FPR 输出同时报告经验比率和单侧 95% Clopper-Pearson 上界；安全判断要求二者都不
超过 target。fixed-FPR threshold 对全部冻结字段重算身份，并在构造与 evaluation
入口校验有限数值、整数计数、经验比例、置信上界及 SHA-256 摘要；calibration case
摘要绑定 split、unit/case/source-cluster、key role 与 score 规范序列。逐 case 结果保留
unit、case、source-cluster 和 split identity，aggregate 只接受单一且由 metric
registry 授权的 split，不把 wrong-key 混入 primary null。

Rescue safety 只接受真实轨迹：raw 阳性不触发 rescue，未触发时不得出现 rectified
阳性，`watermark_decision_positive` 严格等于 `raw_positive or
(rescue_triggered and rectified_positive)`；raw 与 rectified 始终绑定同
detector 和 threshold。

所有 case 拒绝非有限数值、空集合、重复 unit/case/source-cluster role、身份漂移和
`held_out_evaluation`。指标不选择候选、不拟合 LF/HF 权重、不写 records，也不导入
runtime、methods、attacks、runner 或治理代码；CPU 结果不构成科学有效性证明。
