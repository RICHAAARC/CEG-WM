# CEG-WM Evaluation Design

## Purpose Separation

CEG-WM 使用两个互不替代的实验表面：

1. 内部机制验证：回答内容软路由、LF/HF 双载体、Q/K 几何和联合门控是否成立；
2. 外部方法比较：只消费已经冻结的方法，与登记 baseline 做公平比较。

内部验证不能被外部 baseline 结果替代；外部比较不得继续调整项目方法。

## Split Responsibilities

至少冻结以下互不重叠的 source-cluster manifests：

- `development`：实现排错和有限 pilot；
- `candidate_selection_selection`：只做预登记候选选择；
- `candidate_selection_confirmation`：不参与拟合的独立确认；
- `content_threshold_fit`：只读消费已冻结的 soft-LF `W`，拟合 formal branch CDF 与
  冻结 max 内容统计的 `tau`，不得重新拟合 whitening；
- `rescue_window_fit`：只拟合 `tau_rescue`；
- `geometry_reliability_fit`：只拟合可靠性合取门；
- `end_to_end_calibration_check`：检查冻结联合检测器，不再拟合；
- `evaluation`：只评估，不调参。

source cluster 由 Prompt、seed、生成图像 lineage 和注册 key family 共同定义。同一
cluster 的全部攻击、回正和多 key 派生样本必须留在同一职责与 split。任何职责都
不得反向选择路由公式、carrier、组合统计或攻击范围。

## Analysis Unit And Failure Retention

每个预登记单位必须保存：

- exact source-cluster、Prompt/seed/image/key identities；
- method/config/model/code revisions；
- clean、watermarked、attack 和 rectification lineage；
- raw/rectified detector identity、branch scores、route digests 和 threshold identity；
- success、scientific failure、implementation failure、resource failure 或预登记排除；
- 所有预算、质量和非有限量检查。

攻击导致的失败、route 重建失败、无效 Q/K、几何拒绝、回正失败和回正后仍为负都
保留在固定分母。只有运行前定义且与方法效果无关的无效输入条件可以排除。

## Content-Chain Validation Matrix

固定比较臂包括：

- clean；
- HF-only；
- LF-only；
- semantic-texture soft-routed LF/HF；
- route-disabled `m_hf=m_lf=0.5` causal control；
- wrong-key；
- unwatermarked primary null；
- route、budget、quality、identity、integrity、nonfinite 和 runtime failure。

每个单位同时记录：

- `M/T` 和 `m_lf/m_hf` digests、分布、确定性与重建漂移；
- `u_lf/u_hf/u_content` 的 nominal formula witnesses；
- combined actual-dtype `3/250` 总预算，不伪造 branch actual energy；
- `s_lf_soft`、`s_hf_soft`、`z_lf_soft`、`z_hf_soft` 和
  `D_soft_route=max(z_hf_soft,z_lf_soft)`；
- registered/wrong-key/unwatermarked attribution；
- 图像质量和攻击分层结果。

### Routing Causal Question

软路由必须证明：

- `M/T` 由当前普通 RGB8 可确定性重建；
- route-disabled control 不读取 `M/T`；
- 两臂共享 key、Prompt、seed、write position、carrier 和 combined total budget；
- 差异不是由结果后删样、攻击标签、固定权重或不同总能量制造。

### LF/HF Branch Question

LF 与 HF 必须分别证明正确密钥相对 wrong-key 和 primary null 的可校准分离。LF 的
32-clean whitening fit、HF/LF branch standardization 和 max-statistic threshold
使用互斥职责数据；一个分支失败不得被 max statistic 掩盖。

### Combination Question

组合候选只允许固定 `max(z_hf_soft,z_lf_soft)`，不搜索 `a/w/function`，不按攻击
类型切换。确认集必须同时评价：

- 至少一个预登记 HF 易损攻击族的增量 TPR；
- identity 与 HF 已擅长条件下的非退化；
- wrong-key attribution；
- primary-null 与完整联合 FPR；
- combined `3/250` budget 和质量。

## Geometry Validation Matrix

覆盖：

- identity；
- crop；
- scale；
- rotation；
- crop + scale；
- crop + rotation；
- scale + rotation；
- crop + scale + rotation；
- 几何与 JPEG、blur、noise、resize 等非几何失真组合。

每类同时报告 rotation/scale/translation/crop 参数误差、coverage、uniqueness、gap、
key margin、inlier、residual、boundary、可靠率、拒绝率、回正质量、内容分数变化和
wrong-key 行为。oracle transform 只能作为诊断上界。

## Joint-Decision Validation

至少比较：

- raw content only；
- geometry always attempted diagnostic；
- near-threshold conditional recovery；
- geometry reliable but content still negative；
- geometry unreliable；
- oracle transform upper bound；
- rectification followed by the same detector/key/preprocessing/`tau`。

报告 trigger rate、rescue success、false rescue、额外 FPR、计算成本和完整 raw+rescue
FPR。几何分数或可靠性不得直接进入阳性统计。

## Fixed-FPR Target

primary null 是：

```text
unwatermarked generated image + preregistered detection key
```

wrong-key on watermarked image 属于 attribution null，必须单独报告。完整联合检测器的
假阳性包括 raw 越阈值和 rescue 后由同一 `tau` 越阈值，不能只校准 raw 路径。

对 `n` 个独立 primary negatives 中的 `k` 个假阳性，报告经验 FPR 和单侧 `95%`
Clopper-Pearson 上界。若声明 `FPR <= 0.001`，二者都必须不超过 `0.001`。零假阳性
时单侧上界达到该要求至少需要：

```text
n >= ceil(log(0.05) / log(0.999)) = 2995
```

这只是无聚类、单条件、零误报的数学下限。多攻击声明必须预登记 simultaneous
confidence 控制；实际样本量还要按聚类、目标效应、tail tolerance 和 power 确定。

## Metrics

- fixed-FPR TPR、ROC/AUC 补充指标；
- correct-key/wrong-key separation；
- LF、HF、max-statistic 分数分布；
- route determinism、reconstruction drift 和 causal-control effect；
- crop/scale/rotation 参数误差和 geometry reliability calibration；
- raw/rectified/oracle content-score change；
- PSNR、SSIM、LPIPS 或预登记感知指标；
- latency、GPU memory、geometry trigger rate 和 failure rate。

## Evidence Boundary

本文件只规定实验设计。科学结论必须由 governed records 支撑；日志、Notebook
输出、示例图或 harness 报告不能替代固定协议下的科学证据。
