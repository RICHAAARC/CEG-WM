# Evaluation Design

HF-only 参考验证依次由 `hf_only_reference_protocol`、
`hf_only_reference_metrics` 和 `hf_only_threshold_fit_gpu_execution` 组成；整体科学门
统一称为 `hf_only_reference_validation`，三者完成度不得互相替代。

## Separation Of Purposes

CEG-WM 使用两个不同实验表面：

1. 内部设计验证：回答 LF 职责、LF/HF 组合、Q/K 几何和联合门控是否成立。
2. 外部方法比较：在项目方法冻结后，与登记 baseline 进行公平比较。

内部设计验证不应被强制伪装成外部 baseline 比较；外部比较也不能替代组件机制验证。

## Splits

至少固定以下互不重叠的 source-cluster manifests：

- development：实现调试，不进入正式结论；
- candidate-selection calibration：只选择 LF/HF、routing 与其他预登记候选；其内部预登记 selection 与 untouched confirmation partitions；
- content-threshold-fit calibration：只拟合冻结内容检测器的 `tau`；
- rescue-window-fit calibration：只拟合 `tau_rescue`；
- geometry-reliability-fit calibration：只拟合几何可靠性规则；
- end-to-end-calibration-check：只检查冻结联合检测器，不再拟合或选择；
- evaluation：冻结后只评估，不调参。

聚类单位由 Prompt、seed、生成图像 lineage 和注册 key family 共同定义；同一 source cluster 的所有攻击、回正、多 key 派生样本或近重复内容必须留在同一职责 manifest。候选选择、阈值、rescue、geometry reliability、end-to-end check 和 evaluation 之间不得泄漏。

## Content Validation Matrix

- HF-only；
- LF-only；
- LF/HF routing；
- LF/HF combined；
- route-disabled；
- LF-disabled；
- HF-disabled；
- correct key；
- wrong key；
- unwatermarked negative。

固定权重只能作为明确历史 baseline，不能作为默认项目方法。

## Geometry Validation Matrix

- identity；
- crop；
- scale；
- rotation；
- crop + scale；
- crop + rotation；
- scale + rotation；
- crop + scale + rotation；
- 几何与非几何失真组合。

每类同时报告参数估计误差、可靠率、拒绝率、回正质量、内容检测变化和错误密钥行为。

## Joint Decision Validation

至少比较：

- 原图内容检测；
- 无条件几何恢复；
- 近阈值条件恢复；
- 几何可靠但内容仍未达阈值；
- 几何不可靠；
- oracle transform 诊断上界；
- 回正后同检测器同阈值重判。

需要报告救援成功率、错误救援率、额外 FPR、触发比例和计算成本。

## Metrics

计划指标包括：

- 固定 FPR `0.001` 级别下的 TPR；
- ROC/AUC 作为补充，不替代固定 FPR；
- correct-key 与 wrong-key 分离；
- LF、HF 和组合分数分布；
- rotation、scale、crop/translation 估计误差；
- 几何可靠性校准；
- 图像质量和感知指标；
- 每样本延迟、显存和恢复触发率。

具体指标实现和聚合规则在实验协议阶段冻结。

## Fixed-FPR Target

primary null 定义为无水印生成图像与预登记检测密钥构成的独立检测试验。wrong-key on watermarked image 属于 attribution null，必须单独报告，不能与 primary null 混池。

完整联合检测器的假阳性包括 raw 内容分数越阈值和近阈值样本经几何回正后越过同一阈值。正式 calibration 和 evaluation 必须执行完整联合路径，不能只校准 HF direct score。

每个职责 manifest 必须单独预登记样本量和确定规则：

- candidate selection 由最小相关效应、候选数和预登记选择误差的 power 计算确定；
- content-threshold fit 由目标尾部概率、阈值估计容忍度和 raw 路径误差预算确定；
- rescue-window fit 由预登记增量 TPR/FPR 效应和触发率 power 确定；
- geometry-reliability fit 由变换/key 分层覆盖与可靠性校准精度确定；
- end-to-end check 与 formal evaluation 由联合检测器的单侧置信上界、核心攻击数量和 simultaneous confidence 方案确定。

这些规模必须在访问对应职责数据前冻结。预登记计算可基于 development/pilot 摘要上调样本量；不得根据该职责或 evaluation 结果缩减样本量或改变停止规则。任何单个固定总数都不能同时充当候选选择、阈值、rescue、geometry reliability 或 end-to-end check 的充分性证明。

对 `n` 个独立 primary negatives 中的 `k` 个假阳性，必须报告经验 FPR 和单侧 `95%` Clopper-Pearson 上界。若声明 `FPR <= 0.001`，两者都必须不超过 `0.001`。即使观察到零假阳性，单侧 `95%` 上界达到该要求也至少需要 `2995` 个独立负样本；零误报本身不是充分证据。

多个核心攻击条件同时声称相同 FPR 上界时，必须预登记 family-wise confidence 控制。详细误差预算、样本单位和停止规则见 [research_construction_roadmap.md](research_construction_roadmap.md)。

## Evidence Rules

- 所有阈值和选择规则必须绑定 calibration provenance。
- 失败、排除和资源不足必须显式记录。
- 日志、Notebook 输出、示例图和 harness 报告不能进入正式效果表。
- 当前文件是设计要求，不是实验结果。
