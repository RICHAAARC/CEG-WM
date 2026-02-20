# 一、总体结构

Content 链被划分为四个递进层级：

> L1 运行闭合层 → L2 机制对齐层 → L3 算法实现层 → L4 统计验证层

只有当 L1–L4 全部满足，才能宣称：

> Content 链“论文级完全完成”。

---

# 二、L1 运行闭合层（Execution Closure Layer）

## 目标

保证“真实推理可达 + 工程闭环成立”。

## 必达条件（全部必须为 PASS）

1. 真实 SD3.* Transformer pipeline 加载成功。
2. Embed 阶段真实执行推理（inference_status=ok）。
3. Detect 阶段真实执行推理。
4. records / run_closure 可写盘。
5. strict audits = ALLOW_FREEZE。
6. pytest 全 PASS。

## 阻断条件

* pipeline_obj_is_none
* inference_failed
* records 写盘旁路
* freeze_gate 阻断

## 你当前状态

✅ 已完成

---

# 三、L2 机制对齐层（Paper-Faithful Mechanism Layer）

## 目标

证明“推理实现路径与对齐规范一致”，即真实算法执行路径未漂移。

## 必达条件

### P1 — 实现身份锚点

* pipeline_fingerprint_digest 存在
* pipeline_fingerprint_presence = PASS
* detect 验证 P1 一致

### P2 — 轨迹证据锚点

* trajectory_digest 存在
* trajectory_digest_reproducibility = PASS
* detect 生成 P2'
* P2 与 P2' 各自 status=ok

### 合约对齐

* injection_site_alignment = PASS
* method_specific_parameter_binding = PASS
* alignment_report.overall_status = PASS

## 阻断条件

* P1 或 P2 为 NA/FAIL
* overall 与 checks 不一致
* detect 侧 mismatch

## 你当前状态

✅ 已完成

---

# 四、L3 算法实现层（Algorithm Realization Layer）

## 目标

将“真实水印机制”从 placeholder 变为真实算法实现。

这一层才是 Content 链的“核心技术层”。

### 应包含模块

1. SemanticMaskProvider（真实语义掩码）
2. SubspacePlanner（Shallow Diffuse 风格子空间规划）
3. LowFreqCoder（PRC 风格编码）
4. HighFreqEmbedder（TTS 风格截断）
5. ContentDetector（真实统计判别）

## 必达条件

1. embed_record 不再出现 placeholder 生成逻辑。
2. LF/HF 子空间实际参与 latent 修改。
3. plan_digest 真实依赖 mask_digest。
4. ContentDetector 输出真实统计分数。
5. detect 侧 mismatch/absent 语义完整。
6. 消融开关 enable_content=false 能正确关闭。

## 阻断条件

* 仍使用 placeholder 逻辑
* 未进入 latent 修改路径
* 统计输出为固定值
* digest 未绑定算法参数

## 你当前状态

❌ 尚未完成

---

# 五、L4 统计验证层（Statistical Validation Layer）

## 目标

完成论文级统计验证与校准。

## 必达条件

1. Null 分布分层（wrong-key）。
2. Neyman–Pearson 阈值固定。
3. TPR@FPR 可复算。
4. 多 seed、多攻击协议验证。
5. 报告可独立复算。

## 阻断条件

* evaluate 重估阈值
* FPR 漂移
* threshold_metadata 不可复算

## 你当前状态

❌ 尚未开始

---

# 六、四层完成度总览

| 层级 | 名称   | 当前状态  |
| -- | ---- | ----- |
| L1 | 运行闭合 | ✅ 完成  |
| L2 | 机制对齐 | ✅ 完成  |
| L3 | 算法实现 | ❌ 未完成 |
| L4 | 统计验证 | ❌ 未完成 |

---

# 七、你现在的准确工程定位

> 当前项目已完成 Content 链的运行闭合层与机制对齐层，尚未完成真实算法实现层与统计验证层。

因此：

* ❌ 不能说 Content 链全部完成
* ✅ 可以说 Content 链的论文级机制对齐已完成
* ❌ 不能说 Content 链算法完成
* ❌ 更不能说 Content 链论文级性能完成

---

# 八、下一步明确方向

你现在的任务不是再修机制代码，而是进入：

> L3 算法实现层（真实水印嵌入机制）

即开始：

* SubspacePlanner（Shallow Diffuse 对齐）
* PRC 编码
* TTS 高频鲁棒通道

---

如果你愿意，我可以再给你一个：

**“L3 算法实现层三阶段推进图”**
把真实算法注入拆成可控的最小安全步骤，避免破坏你已经完成的 L1/L2 审计闭环。
