# 内容链论文级规模化验证

目标：在不改冻结面与统计口径的前提下，把内容链推进到“可投稿实验级稳定状态”。

---

## 阶段 0：实验规范锁定

目标：冻结实验协议，避免跑到一半再改规则。

### 0.1 固化实验配置

* 固定：

  * prompts 列表
  * seed 列表
  * watermark key 列表
  * enable_high_freq 组合
  * 分辨率
  * attack_protocol_version
* 生成：

  * experiment_manifest.json
  * seed_bucket 定义（例如 0–9999）

### 0.2 固化评测指标

明确你论文要报告：

* 主指标：TPR@FPR（target_fpr 固定）
* reject_rate
* CI（Wilson 或 bootstrap）
* PSNR / LPIPS（可选）

输出：

* metrics_schema_v1 定义文件（append-only）

---

## 阶段 1：大规模 Null 校准

目标：验证阈值在规模上收敛。

### 1.1 wrong-key null ≥ 10k

* 生成 ≥ 10,000 null 样本
* 只收集 status="ok" 的 content_score
* 输出：

  * thresholds.json
  * threshold_metadata.json

### 1.2 收敛性验证

绘制：

* 阈值 vs 样本数（n 从 1k → 10k）
* FPR 误差 vs 样本数

判断：

* 误差 < 5% 相对偏差
* 阈值收敛平稳

若不平稳：

* 调整嵌入强度归一化（不改冻结口径）

---

## 阶段 2：消融实验矩阵

目标：证明每个组件真实贡献。

### 2.1 三组对照

1. LF-only
2. HF-only
3. LF+HF

每组跑：

* null 校准
* 正样本检测
* 输出 TPR@FPR

### 2.2 消融 mask / planner

* enable_mask=false
* 固定 band_spec
* 输出性能对比

论文必须有这部分。

---

## 阶段 3：攻击鲁棒性扩展

目标：验证在攻击下性能行为。

### 3.1 攻击矩阵

* JPEG Q=30/50/70/90
* Resize 0.5× / 0.75× / 1.25×
* Crop 5% / 10%
* Gaussian noise σ 分层

### 3.2 输出指标

每个攻击：

* TPR@FPR
* reject_rate
* LF-only vs HF-only vs LF+HF 曲线

观察：

* HF 是否显著提高鲁棒性
* FPR 是否仍受控

---

## 阶段 4：分层统计与稳定性分析

目标：论文严谨性。

### 4.1 分层

按：

* seed bucket
* resolution
* prompt 类别

输出：

* 分层 FPR
* 分层 TPR

### 4.2 条件 FPR 分析

验证：

* enable_high_freq 条件下 FPR 不上升
* 各分辨率下 FPR 一致

---

## 阶段 5：置信区间与功效分析

目标：论文说服力。

### 5.1 Wilson interval / bootstrap CI

输出：

* TPR CI
* FPR CI

### 5.2 功效分析

* 给出样本规模 vs 统计功效曲线

---

## 阶段 6：论文复现包封板

目标：工程收口。

### 6.1 生成 reproducibility bundle

包含：

* experiment_manifest.json
* thresholds + metadata
* metrics + report
* 所有 digest 锚点
* 运行脚本

### 6.2 冻结实验版本

* 固定 experiment_version
* 固定 attack_protocol_version
* 生成 signoff_bundle_v2

---

## 你现在立刻可以做的第一步

执行 0：

* 写 manifest
* 固化 attack_protocol
* 定义样本规模目标
* 明确报告指标

不要马上跑 10k 实验。
先把协议写清楚。

---

## 最终状态标准

当满足以下条件时，可以说：

> 内容链论文级阶段完成

1. null ≥ 10k 且阈值收敛
2. LF/HF 消融清晰
3. 攻击曲线稳定
4. 条件 FPR 无异常
5. CI 报告完整
6. 复现包可一键运行