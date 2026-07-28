# CEG-WM Method Mechanism

## Purpose

本文档把 [algorithm_primitives.md](algorithm_primitives.md) 中的算法原语和
[candidate_specifications.md](candidate_specifications.md) 中已关闭的有限候选组织为
可实现、可验证且不越过当前 `method_implemented` 阶段的端到端机制。
当前 13 项职责已按它实现并经 CPU/synthetic readiness 审核；本文档自身、
admission、readiness 和阶段转换都不提供 runtime 或科学证据。

## Method Identity

一个正式 CEG-WM 方法身份必须同时绑定：

- `key_schedule_sha256_counter` 的 root-key encoding、KDF/PRG、domain schema、
  quantile table 和 public digest 规则；
- 内容检测器身份；
- HF 与可选 LF 载体、模板和评分身份；
- 内容路由身份；
- 生成模型、图像编码、预处理、尺寸和 dtype；
- Q/K 观测位置与几何关系身份；
- 变换支持域、估计器、可靠性和回正规则；
- 内容阈值、近阈值区间和 calibration provenance；
- 密钥派生与职责域；
- 实现 revision 和依赖锁。

任一项变化都产生新的方法身份。新身份不得消费旧身份的阈值、分数分布或 formal records。

## Shared Key Mechanism

调用方只把非空 Unicode root key 以严格 UTF-8 语义交给
`main.shared.key_schedule`。共享责任按候选规格生成 HF、LF、geometry 三个秘密
职责流，以及与 secret 无关的 image-only Q/K noise 和 routing sensitivity probe。
wrong-key roster 只从注册 key 的公开 digest 与预登记 index 派生。

内容链与几何链不得重新编码 key、调用设备 RNG 或定义自己的 domain 字段。正式
records 只能保存 root public digest、PRG/domain identity、public-noise identity 和
输出 tensor digest；不得保存 root/derived key material。

## Responsibility Separation

```text
content_chain
├── content_observation_and_routing
├── lf_carrier
├── hf_carrier
├── content_embedder
├── lf_detector
├── hf_detector
└── content_detector

geometry_chain
├── qk_geometry_sync
├── geometric_transform_estimator
├── geometry_reliability
└── image_rectification

joint_decision
├── raw_content_decision
├── near_threshold_gate
├── conditional_geometry_call
└── same_detector_same_threshold_recheck
```

内容链和几何链不互相导入。联合判定只编排公开结果，不重新计算载体、几何目标或阈值。

## Embedding Mechanism

### Inputs

嵌入机制可以消费：

- 生成请求及其公开生成条件；
- 原始水印密钥的内存值；
- 冻结方法配置；
- 受控生成模型内部的合法观测；
- 可复现随机源。

原始密钥不得写入普通请求、records、图像元数据或 artifacts。

### Base Generation

运行时从冻结 prompt、seed、模型 revision、调度器和尺寸构造 clean 与 watermarked 可比链。配对实验必须从同一登记基础随机状态开始，除水印机制外不允许引入不受控差异。

### Content Observation And Routing

在登记的生成时刻提取候选内容观测。`content_router` 只据此输出 `A`、
`mask_lf`、`mask_hf`、route identity/digests 和 disabled uniform control。
路由计算属于内容链；模型执行和张量捕获属于 runtime。router 不决定 `a`，不输出
标量预算，也不拥有 target/realized 写入量。

正式候选必须记录：

- 路由输入来源和公共可用性；
- `A` 与两 mask 的覆盖、范围、partition identity 和退化；
- observation/config digests；
- disabled control 的 `A=mask_lf=mask_hf=1`，以及它不读取 observations。

路由不得读取攻击标签或 evaluation 结果。

### Content Carrier Directions And Embedder Write

HF 写入方向独立使用 CEG-WM HF carrier 职责域密钥。LF 候选写入方向独立使用 LF
职责域密钥。两个 carrier 只拥有模板、支持/滤波结构、router mask 形成的单位方向及
必要 identity，并且必须能够单独启用；它们不产生 delta、不物化写入，也不拥有实际
写入能量。

独立 `content_embedder` 消费两条单位方向和 router masks，独占配置中的冻结 `a`、
`u_content(a)`、`content_relative_l2_nominal=3/250`、
`content_relative_l2_limit=3/250`、HF-only/LF-only/combined
`delta_content_nominal`、nominal total norm/relative L2 与 active/combined
零方向失败。`a` 和
`1-a` 只是 mixing coefficients；combined 方向的 norm 含
`2*a*(1-a)*dot(u_lf,u_hf)` 交叉项。runtime 只在冻结 callback/model/dtype 边界物化
该 delta，并把实际张量及 realized combined total norm/relative L2 返回给 embedder
判定；runtime 不改变预算或方向。carrier、router、runtime 或 detector 不得代行
组合与预算判定。

对 callback 18 actual baseline `z0` 和一个正 binary32 scale `s<=1`，runtime
只执行
`z_s=cast_binary16_RNE(fp32(z0)+f32(s*delta_content_nominal))`，并返回
`delta_actual_s=fp32(z_s)-fp32(z0)`。加法、scale 乘法、row-major norm 累加、
`sqrt`、limit 乘法和比较逐步冻结为 binary32；binary16 RNE、subnormal 与 overflow
语义按候选规格执行。完整性要求 finite、独立 bitwise replay 和最终 actual delta
非零。

`content_embedder` 以
`norm32(delta_actual_s)<=f32((3/250)*norm32(fp32(z0)))` 直接比较，不使用 ratio
边界、`q_budget`、`tau_actual_budget`、tolerance 或实际强度下限。full scale
超限时，它在 binary32 `[0,1]` 上以冻结 midpoint 二分到没有新 representable
midpoint；zero plateau 不作为可行写入，最终返回最大非零可行 scale 或 fail closed。
runtime 不拥有 accept/retry/scale/final-failure 语义。

因此正式候选的组合记录归 `content_embedder`：包括冻结 `a`、方向/支持 identity、
`dot(u_lf,u_hf)`、combined pre-normalization norm、nominal 与 limit、runtime
返回的 realized combined total norm/relative L2、materialization scale、
attempt count、integrity/budget status 和仅诊断的 utilization。若记录 target
component vectors/norms，必须注明它们只是 nominal formula witnesses 且不可相加
为 total；`content_direction`、`active_lf_direction`、`active_hf_direction` 也不
构成 actual branch decomposition。低 utilization 不得在未来实验中被事后筛除。
route record 只保留 observations、`A`、masks、覆盖和 identity/digests。

正式比较至少保持以下因果对照：

- clean；
- HF-only；
- LF-only；
- LF/HF route-disabled；
- LF/HF routed；
- 分支禁用控制。

不得把不同总能量候选的效果差异直接解释为路由或组合增益。

### Geometry Synchronization Write

几何同步使用独立密钥职责域和真实 Q/K 关系。它只为检测端的变换估计提供可恢复结构。
嵌入目标不是生成 conditional forward 的 Q/K：每个 baseline/candidate 必须 replay
剩余生成 suffix 得到普通图像，再按与盲检完全相同的 VAE-mode、公开噪声、
schedule-index-7、三路空文本、无-CFG image-only forward 评分。

几何写入必须满足：

- 不改变内容检测统计的定义；
- 不共享 LF/HF 分数作为优化目标；
- 可以独立禁用并保持内容链可运行；
- 实际 Q/K 关系改善与内容图像质量均可测量；
- 写入失败显式传播，不能退化为无记录的 content-only 成功。

首个候选的几何写入与内容写入均位于 callback index 18，内容先写、几何后写且只做
一次最终 actual-dtype 物化。LF/HF active directions 非正交时按
`U(U^TU)^dagger U^T` 去除完整 content span；每个回溯候选以 actual-dtype geometry
delta、combined delta 和 replay 后 image-only relation score 验收。
geometry/content budget ratio 只在 `qk_relation_similarity` 登记的有限集合中选择。
该候选需验证，不从历史结果继承成功。

### Generated Output

正式方法输出普通图像。检测不依赖输出旁路携带的 embed record、参考图、私有 latent 或 Q/K 缓存。

## Detection Mechanism

### Raw Content Detection

检测从待检图像、检测密钥和公共冻结资产分别构造盲 `s_lf` 与 `s_hf`。
`lf_detector` 和 `hf_detector` 必须是独立可调用责任，三类 `s_lf`、`s_hf`、
`s_combined` 必须独立可观测。`content_detector` 消费两个分支统计并形成
`D_M(I, K)`；在 LF/HF 组合晋升前，`D_M` 等于 CEG-WM HF direct score。组合不得
掩盖任一分支的错误密钥失败。

原图检测必须首先完成，几何链不能预先改变所有输入，也不能成为默认前处理。

### Near-Threshold Gate

原图内容 margin 决定三种互斥路径：

- 达到 `tau`：内容阳性；
- 低于 `tau_rescue`：内容阴性且不触发几何；
- 位于 `[tau_rescue, tau)`：具有几何资格，但仍是内容阴性。

“具有资格”不表示存在水印，也不表示几何可靠。

### Conditional Geometry Estimation

只有近阈值样本进入几何链。几何链重新从当前待检图像提取真实 Q/K，使用检测密钥职责域估计有界 crop、scale、rotation 和必要 translation。

transform estimator 输出原始指标，独立 geometry reliability 组件对这些指标执行
冻结合取门。估计与可靠性输出必须区分：

- 支持域内可靠解；
- 未发现同步；
- 低覆盖；
- 高残差；
- 多候选歧义；
- 超出支持范围；
- 无效回正。

错误密钥下偶然出现高几何分数不能形成阳性。

### Rectification And Recheck

仅可靠解允许回正。回正图像作为一个新的普通图像输入重新执行完整 `D_M`：

```text
s_rectified = D_M(rectified_image, K)
```

禁止：

- 复用原图内容 latent；
- 只重算部分分支；
- 为回正图切换模板、权重或阈值；
- 用几何置信度补足内容 margin；
- 在回正失败时降低阈值。

最终阳性只可能来自 `s_raw >= tau` 或 `s_rectified >= tau`。

## LF/HF Responsibility And Combination

### Stable Division

HF 的稳定职责是当前主密钥归属证据。LF 的候选职责是对 HF 易受损条件提供互补密钥证据。

LF 不是：

- 对所有样本的默认加分项；
- 几何失败时的宽松替代分类器；
- 通用图像低频检测器；
- 继承固定权重的历史分支。

### Selection Flow

```text
HF candidate reproducibility
        ↓
LF-only key attribution
        ↓
LF attack complementarity
        ↓
LF/HF candidate combination on calibration
        ↓
untouched candidate-selection confirmation
        ↓
independent threshold/rescue/reliability fit and end-to-end check
        ├── pass → full content branch eligible
        └── fail → research_question_closed_negative
```

组合选择前，LF 必须先单独通过。组合通过后，检测器身份整体变化，原图、近阈值门和回正重判必须一起更新并重新校准。

失败分支是有效、可报告的研究闭合，但不能与完整 CEG-WM 成功汇合。HF-only 加 geometry 若被继续研究，只能作为重新命名和重新授权的 reduced-scope method；它不得声明已经实现或验证“内容自适应路由 + LF + HF + Q/K + 回正 + 联合判定”的完整方法。

## Geometry Mechanism And Crop Boundary

crop 同时造成内容删除、坐标平移、有效尺度变化和插值重采样。几何链只能估计仍可观测的坐标关系，不能恢复已删除内容。

正式回正规则必须预先固定：

- 输出坐标系和尺寸；
- inverse warp 方向；
- 插值和量化；
- 缺失区域 padding；
- 有效支持；
- 极端 crop 的拒绝边界。

如果 crop 后剩余证据不足，正确行为是可靠性失败或回正后内容仍为负，而不是生成式补全或 oracle 参数救援。

## Configuration And Threshold Identity

内容分数、几何估计和联合判定必须由一个可摘要的配置图连接。至少包括：

```text
method identity
├── key derivation identity
├── content detector identity
│   ├── image encoder and preprocessing
│   ├── HF carrier template and HF direct score
│   ├── optional LF template and score
│   └── optional calibrated combination
├── routing identity
├── geometry identity
│   ├── Q/K observation
│   ├── relation and estimator
│   ├── reliability
│   └── rectification
└── decision identity
    ├── tau
    ├── tau_rescue
    └── calibration provenance
```

阈值摘要必须绑定内容检测器完整身份。几何配置可以影响是否产生回正图，但不能改变 `tau` 的数值解释。完整联合检测器必须额外验证几何救援没有突破预设 FPR 预算。

## Public Interfaces

当前实现按权威 13 项职责提供独立 symbol；接口语义覆盖：

- 构造内容载体候选；
- 组合 LF/HF 写入并保持共同总预算；
- 分别测量 LF、HF 和组合内容统计；
- 提取 Q/K 几何观测；
- 估计变换并由独立组件判断可靠性；
- 回正普通图像；
- 执行联合判定；
- 返回可供实验层持久化的不可变结果。

这些接口不得直接读写 records。records 只能由 `experiments/runners/` 物化。

## Runtime Boundary

runtime 负责：

- 模型和 checkpoint 加载；
- device、dtype 和确定性执行；
- VAE 或等价图像观测编码；
- 生成 callback 或调度器适配；
- 真实 Q/K 捕获；
- 图像张量和普通图像转换。
- 在冻结 callback/model/dtype 边界物化 embedder 的 `delta_content`，并把实际张量
  与 independent replay、integrity、realized combined total norm/relative L2
  返回给 embedder。

runtime 不负责：

- LF/HF 组合规则；
- `a`、nominal/limit、方向选择、hard-budget acceptance、retry、最大非零可行
  scale 或最终 budget failure；
- `tau` 或 `tau_rescue`；
- 几何可靠性决策语义；
- 最终水印阳性；
- records 写入。

## Experiment Boundary

实验协议定义样本、切分、攻击、阈值拟合职责、排除规则和 records schema。实验 method adapter 只连接方法与 runtime，攻击和指标保持正交，runner 是唯一 records 写入层。

内部设计验证和外部 baseline 比较必须分开：

- 内部验证选择项目候选；
- 外部比较只消费已经冻结的项目方法；
- 外部比较不得继续调整 LF/HF、几何或 rescue 参数。

## Failure Propagation

每个输入样本都必须得到成功、失败或按预先规则排除的显式状态。下游汇总必须保留：

- 内容检测失败；
- 几何未触发；
- 几何不可靠；
- 回正失败；
- 回正后仍为负；
- 错误密钥误归属；
- runtime、资源或依赖失败。

只有预先定义且与方法效果无关的输入无效条件可以排除样本。攻击导致的失败属于方法结果，必须保留在分母。

## Mechanism Validation Ladder

方法机制按以下依赖顺序关闭：

1. CEG-WM HF candidate identity、HF direct score 和 raw content detector 可复现；
2. LF-only 具备独立 key attribution；
3. 内容路由在相同预算下提供增益；
4. LF/HF 组合通过晋升门；若未通过，则关闭完整 CEG-WM 成功路径并形成内容分支负结果；
5. Q/K 同步在真实 runtime 可观测；
6. 变换估计与可靠性在 identity、单变换、组合变换和错误密钥下通过；
7. 回正能改善同一内容检测器；
8. 条件恢复相对 raw-only 提供增益且不突破 FPR；
9. 完整方法在冻结 calibration/evaluation 和攻击矩阵下形成 formal records。

前一项未通过时，不得用后一项的复杂度掩盖根因。

root-key/KDF/PRG、Q/K relation/objective、LF write/score、routing observations、
backbone/runtime 已由具名候选关闭，后续 method/runtime 工作必须按候选 ID 工作，不能自行发明
替代方案。候选规格已独立复审批准，CEG-WM 版本身份和 construction admission
已绑定，项目已以不含 `main/` 变更的独立 revision 进入
`method_construction_authorized`。随后独立 revisions 已按候选完成方法实现、
CPU/synthetic 验证与 readiness 审核；独立阶段迁移已经完成。

## Current Status

当前项目登记为
`method_implemented / implemented`：

- 13 项职责、27 个 CPU/synthetic 行为节点和唯一 method readiness 已完成并审计；
- 正式 detector 仍为 HF-only，LF/routing 未实验晋升，
  `full_ceg_wm_eligible=false`；
- 没有冻结 runtime；
- 没有本项目 calibration 阈值；
- 没有正式 GPU 或论文 records；
- 没有 FPR、鲁棒性或比较优势结论。
