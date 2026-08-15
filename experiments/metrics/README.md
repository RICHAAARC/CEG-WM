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

`hf_only_reference_metrics.py` 实现 hf_only_reference_protocol 冻结的七个 HF-reference metric identities。
`hf_only_reference_tau_fit` 只接受 content-threshold-fit manifest 的完整 4096 个
`AnalysisUnitIdentity`，以 binary64 `nextafter(max,+inf)` 得到零 fit-FP 的唯一
threshold identity。confirmation 必须消费该 threshold，且 primary null、
registered 与 wrong-key 三类各 4096、逐 cluster 同 detector/config/key/control
身份完整，wrong-key 不与 primary null 混池。

paired quality 的单 pair 入口直接流式消费 HWC RGB8 bytes，计算 normalized MSE
与 relative L2；它返回绑定完整 analysis unit、图像摘要、公式摘要和可重算
result identity 的轻量 case result。独立 aggregate 可消费这些 metric 产出的
4096 个轻量结果，计算 mean、sample SD (`ddof=1`) 与未裁剪的双侧 95% Student-t
区间。正式 confirmation 不信任调用方提交的轻量数值结果，而只接受 4096 个
绝对路径 raw-RGB8 artifact descriptors；它逐 pair 校验 raw artifact SHA-256、
读取精确 HWC byte count 并自行调用单 pair 公式，避免所有原始图像同时驻留内存。
Student-t CDF/quantile 与共享 Clopper-Pearson 原语均为无额外依赖的数值实现并有
reference golden。

正式 confirmation 入口是 `evaluate_hf_only_reference_confirmation_metrics`。它必须同时消费
content-threshold-fit 的精确 4096 条 primary-null score cases，自行重算
`fit_hf_only_reference_tau` 并要求传入 threshold 与重算对象完全相等；随后逐 pair 重放 raw
RGB8 artifacts，交叉验证 score、paired-quality 与 actual-dtype 三表的 exact
manifest unit、clean/marked image digests 和 registered-key identities，并把 fit
case/threshold identity 纳入 `cross_input_digest`；hf_only_threshold_fit_gpu_execution 不得绕过该入口。实现绑定由
`configs/experiments/hf_only_reference_metrics.json` 固定 hf_only_reference_validation spec、完整 hf_only_reference_protocol
authority bundle、七项 split/formula、metric registry、实现 symbols 与源码
SHA-256。该实现不写 records、不执行 promotion decision，也不产生科学结果。

`salient_local_lf_mask_write_validation.py` 对 public uint8 `[1,3,512,512]` pair
计算 signed-int64 `sum((marked-clean)^2)`；唯一 quality pass 为
`S_delta <= 786432`。coverage、IoU、nominal masked-LF causal witness、actual-dtype
budget 与 identity/integrity 独立合取；quality violation 仍是完整 scientific
observation。aggregate 固定 8 分母，quality 要求 8/8、mechanism 要求至少 7/8。
