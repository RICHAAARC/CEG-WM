# 一、总体结构

Geometry 链同样分为四层：

> G1 运行闭合层 → G2 机制对齐层 → G3 算法实现层 → G4 统计验证层

注意：
Geometry 链与 Content 链共享 L1/L2 的“运行与对齐基础设施”，但算法层与统计层完全独立。

---

# 二、G1 运行闭合层（Execution Closure Layer）

## 目标

保证几何链不会破坏现有闭环。

## 必达条件

1. enable_geometry=true 时，records 不报 schema 错误。
2. detect 阶段 geometry_evidence 字段存在（允许 absent，但语义必须明确）。
3. strict audits 不 FAIL。
4. 关闭 enable_geometry 时，系统行为可消融且不污染 content_score。

## 阻断条件

* geometry 模块抛异常但被吞掉
* 未注册 impl_id
* 写盘旁路
* 失败语义不区分 absent/mismatch/fail

## 你当前状态

若 geometry 仍为 placeholder，则：

* 运行闭合通常已完成（因为结构已冻结）

---

# 三、G2 机制对齐层（Paper-Faithful Geometry Layer）

## 目标

保证几何链的“对齐证据”可复算、可审计、不可漂移。

## 必达条件

### G2-1 Anchor 指纹

* anchor_digest 存在
* stability_metrics 存在
* anchor_unstable 明确枚举

### G2-2 Sync 机制摘要

* sync_digest 存在
* sync_quality_metrics 可复算

### G2-3 几何对齐结果

* geo_score 存在（或 absent 原因）
* align_trace 结构化

### Detect 一致性

* detect 验证 anchor_digest 一致
* paper_faithfulness.status=ok

## 阻断条件

* 写入绝对坐标到 records（违反冻结策略）
* 几何链失败污染 content_score
* digest 未 canonical 化

---

# 四、G3 算法实现层（Algorithm Realization Layer）

## 目标

实现真正的“几何不变证据链”。

### 应包含模块

1. AttentionAnchorExtractor（基于 Self-Attention 图）
2. LatentSyncTemplate（latent 空间同步模板）
3. GeometryAligner（对齐算法）
4. InvarianceScorer（几何残差统计）

## 必达条件

1. Anchor 提取基于真实 SD3 Transformer attention。
2. Anchor 为“相对关系摘要”，不存绝对像素。
3. 对齐算法真实运行（例如 RANSAC/残差优化）。
4. geo_score 由真实统计生成。
5. failure_reason 枚举完整。

## 阻断条件

* anchor 为固定占位值
* 对齐结果为常数
* 未区分 insufficient_anchors / align_failed

---

# 五、G4 统计验证层（Statistical Validation Layer）

## 目标

证明几何链在攻击下提供增益。

## 必达条件

1. 几何攻击协议版本化。
2. 记录 geo 可用率。
3. 记录融合救回率。
4. 不改变主链 FPR。
5. 条件 FPR 明确（若使用救回策略）。

## 阻断条件

* 几何链导致 FPR 上升
* geo_score 未校准
* 融合规则改变 NP 口径

---

# 六、Geometry 与 Content 的关系

| 层级    | Content | Geometry |
| ----- | ------- | -------- |
| L1/G1 | 运行闭合    | 运行闭合     |
| L2/G2 | 推理对齐    | 锚点/对齐对齐  |
| L3/G3 | 水印算法    | 几何对齐算法   |
| L4/G4 | 统计校准    | 攻击统计验证   |