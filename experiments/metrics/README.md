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
超过 target。逐 case 结果保留 unit、case 和 source-cluster identity，聚合不把
wrong-key 混入 primary null。

所有 case 拒绝非有限数值、空集合、重复 unit/case/source-cluster role、身份漂移和
`held_out_evaluation`。指标不选择候选、不拟合 LF/HF 权重、不写 records，也不导入
runtime、methods、attacks、runner 或治理代码；CPU 结果不构成科学有效性证明。
