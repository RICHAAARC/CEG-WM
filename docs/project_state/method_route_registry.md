# CEG-WM Method Route State Registry

## Purpose And Authority

本台账把“方法是什么”与“该方法验证到了哪里”分离。它只记录状态、证据边界和
禁止回流规则；算法本身见 [docs/design](../design/README.md)。

状态类别固定为：

- `adopted_design_unimplemented`：已进入方法设计，但没有对应实现或科学证据；
- `implemented_not_scientifically_validated`：有实现或结构/运行证据，但没有机制晋升；
- `authenticated_development_negative`：由固定 producer、固定分母和可复算 artifact
  支持的开发负结果；
- `historical_non_authoritative_observation`：来自历史偏离项目，只能提示风险；
- `operational_or_resource_failure`：运行环境、资源或交付失败，不能推断机制失败；
- `not_yet_tested`：没有可裁决证据，禁止提前记为成功或失败；
- `superseded_without_scientific_adjudication`：设计被新路线替代，但旧机制没有经过
  足以作科学裁决的验证。

只有 `authenticated_development_negative` 可以进入 CEG-WM 的失败路线清单；其他状态
必须单独保存，不能混写为“方法失败”。

## Adopted Method Route

状态：`implemented_not_scientifically_validated`。

当前五候选实现身份为：

- `routing_semantic_texture_soft`；
- `content_embedding_semantic_texture_soft_lf_hf`；
- `lf_semantic_texture_soft_whitened_matched_score`；
- `hf_semantic_texture_soft_direct_score`；
- `content_combination_semantic_texture_max_standardized`。

当前 reviewed revision：`cd541e5fa7ffeabc1db1f74a3e9f5a925e0112d9`。独立复核绑定：
`independent_semantic_domain_method_runtime_delivery_review:01a00740-db8c-7b81-b370-3b0fd5db8285:cd541e5fa7ffeabc1db1f74a3e9f5a925e0112d9:independent_exact_audit_approve+cumulative_method_runtime_delivery_gate_approve`。
`02fe5dcc2b74482c9eb1e0b192b4a2ce79e0d9eb` 只保留为 semantic-domain 变更前的历史
exact provenance，不是当前 readiness 或方法身份权威。本次 authority rebinding 的
completion profile 状态为 `profile_pending`。
该状态只表示实现/API/CPU-synthetic 语义已审核；soft max 仍是 diagnostic、未晋升，
正式 detector 仍为 HF-only，`full_ceg_wm_eligible=false`。它不提供 runtime
qualification、机制成功、calibration、固定 FPR、GPU 或科学效果证据。

采用的方法路线是：

```text
内容链
├── 语义显著性 M + 纹理复杂度 T 的逐图软路由
├── 独立 keyed LF carrier
├── 独立 keyed HF carrier
├── 同一总预算下的双分支组合嵌入
└── 独立分支分数 + 固定 max 标准化内容统计

几何链
├── Q/K keyed synchronization relation
├── crop / scale / rotation 有界估计
├── 独立可靠性合取门
└── 图像回正

联合判定
├── 原图内容检测
├── 近阈值负样本资格判断
├── 可靠时才调用几何恢复
└── 回正后以同一 detector / key / preprocessing / tau 重判
```

设计采用 InSPyReNet soft probability 作为 `M`，采用确定性灰度 Sobel/P95 映射作为
`T`，不要求逐图 Jacobian、SVD、显式零空间或 Self-Attention 几何锚点。准确公式和
身份见 `docs/design`。这一路线是对开题“内容自适应潜空间水印 + 几何条件恢复”的
最小忠实实现。当前已实现但未实验晋升，也未形成效果证据。

历史 readiness 快照在 revision
`0258ccb2100bfe8b58d1a12079876841192528b3` 绑定 11 个唯一候选身份和 28 个行为节点；
它继续作为旧实现 exact-replay 的历史来源保留，但不再是当前 readiness 权威，也不向
soft-route 五候选传递 CDF、threshold、calibration 或效果证据。

## Authenticated Development Negatives

以下清单截至 2026-08-16 是当前仓库和已验真历史交付中可认证的 CEG-WM 开发负结果。

### `routing_stqr` fixed-half directional diagnosis

- producer revision：`925c2cbc727e3b18e91c0b3981eeed1b470a955a`；
- run：`ceg_wm_content_routing_positive_reference_support_correction_diagnosis`；
- 固定分母：`42/42` terminal；
- 8 个顺序增量指标：`1,1,1,0,0,0,0,0`，即 `3/8=0.375`，未满足 strict `>0.5`；
- clusters `1`,`5`,`6` 的 RGB relative-L2 超限，不能计为成功 cluster；
- 结论：`authenticated_development_negative`。

禁止回流：不得从该 8-probe 结果选择 S/T/R/Q reference、mask、threshold、coverage
或混合参数；不得补样、删 cluster、重跑、增加 attempt、放宽 `3/250`，也不得把
margin-only 子集重签为 winner。

### `content_uniform_combination` directional diagnosis

- producer revision：`7c0d86d6eac5ffcfc4a30f2f5fb22884aaa848da`；
- delivery revision：`6b5e302f4cefb051b34f0da3dc71b29b191b6ba2`；
- run：`ceg_wm_content_uniform_combination_budget_observation_correction_diagnosis`；
- artifact ZIP SHA-256：
  `698dd70f0d6798a86310987f21d54056dd2cc27d4e67743bb1671a9927d31435`；
- 固定分母：`1 operational + 32 reference + 8 probes = 41`，全部 attempt 0
  `COMMITTED`，无 retry、dangling 或 duplicate；
- canonical binary32 budget：`3/250`；
- clusters `1`,`6` 出现 `clean_to_watermarked_rgb_relative_l2` 预算违规；
- aggregate：`mechanism_signal_not_observed`、
  `candidate_not_recommended_for_selection`、
  `allow_request_for_content_combination_candidate_selection=false`；
- 结论：`authenticated_development_negative`。

禁止回流：不得选择旧 `a/w/function`、固定 `0.70/0.30` 或 `0.50/0.50`，不得放宽
质量/预算合取门，不得删 cluster、补样、重跑或用五个 margin-only 候选覆盖真实预算
违规。该结果不否定独立 HF/LF carrier，但关闭依赖该旧组合身份的 downstream 晋升。

## Superseded Or Unadjudicated Designs

### Hard salient-object local-LF family

涉及身份：

- `routing_inspyrenet_salient_local_lf`；
- `content_embedding_global_hf_local_lf`；
- `lf_saliency_masked_null_whitened_matched_score`；
- `content_combination_saliency_max_standardized`。

状态：`superseded_without_scientific_adjudication`。

该路线采用 hard threshold、3x3 erosion、coverage `64..3072`、global-HF/local-LF
写入。它被语义—纹理软路由替代是设计简化决策，不是科学失败。不得把本地 checkpoint
兼容性 smoke、计划中的 F2 或代码存在解释为该机制已经成功或失败；也不得把其 hard
mask、erosion、coverage 门静默带入新软路由。

### LF, Q/K geometry and full joint detector

状态：`not_yet_tested` 或 `implemented_not_scientifically_validated`，具体取决于所指
实现身份。

- LF 的独立盲 key attribution、primary-null FPR 和攻击互补性尚无完整方法晋升证据；
- Q/K 的真实 observation/runtime 可用性不能替代 synchronization write、变换估计、
  reliability 和 rectification 的科学验证；
- 联合判定的 CPU/synthetic 控制流不能替代完整 raw+rescue 固定 FPR；
- 因而完整 CEG-WM 不能由这些窄证据推断为成功或失败。

## Operational And Resource Failures

以下事实保留以防工程问题被遗忘，但不进入科学失败路线：

- InSPyReNet 本地依赖物化曾在满载 `/tmp` tmpfs 上先触发 `EXDEV`，随后
  `copytree` 并以 `Errno 28 No space left on device` 终止；分类为
  `operational_or_resource_failure`。
- 转到新的 native ext4 路径后，单次 checkpoint/API/CUDA compatibility smoke 通过；
  desktop-only GPU-name helper 仍返回 rc1，属于机械 helper 缺陷，不改变该 smoke
  的有限边界。
- 这些事实既不能证明 hard-mask 路线有效，也不能证明它无效。

## Historical Non-Authoritative Observations

以下只用于风险提示，不是 CEG-WM 失败证据：

- `SLM-WM` 与 `SLM-WM-FlowHF` 是历史偏离项目；其中旧 LF-dominant 固定融合、强耦合
  Q/K、reference/private-state 路线不得进入 CEG-WM。
- historical DirectHF 的指定 34-image 固定攻击诊断中，center crop `0.90` 为
  `0/34`，说明未同步的 direct content detection 存在几何失步风险；它不证明当前
  Q/K 条件恢复失败，也不等同于总体 crop 鲁棒性结论。
- historical DirectHF、runtime qualification、CPU/readiness 和代码存在只能支持其
  各自窄边界，不能支撑完整 CEG-WM。

## Re-Entry Rule

已认证开发负结果只能以以下方式重新研究：

1. 建立新的 candidate identity；
2. 明确指出与失败身份的语义差异；
3. 使用全新且互斥的 manifests/splits；
4. 预登记固定分母、预算、质量、wrong-key 和停止规则；
5. 不复用旧结果选择参数；
6. 经过独立审计和用户重新授权。

仅换名称、轻微调权、删失败样本、放宽门禁或从旧结果挑选局部 winner，不构成新路线。

## Update Rule

新增状态记录必须绑定 exact candidate、producer/revision、run、固定分母、artifact
摘要、裁决范围和允许/禁止的下游动作。若缺少其中任一关键身份，只能登记为
`not_yet_tested`、`operational_or_resource_failure` 或
`historical_non_authoritative_observation`，不得登记为已认证科学失败。
