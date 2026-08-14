# Content Chain Design

未晋升组合函数的正式身份仅为 `hf_only_standardized_score`、
`weighted_hf_lf_standardized_score` 和
`maximum_hf_lf_standardized_score`；冻结权重保存在独立 `weight` 字段中。

## Frozen Responsibility

内容链是唯一水印证据来源。几何可靠性、配准质量或变换估计不得直接加入内容阳性分数。
LF/HF template、wrong-key roster 和 routing public probe 统一消费
`key_schedule_sha256_counter`；内容组件不得各自定义 key encoding 或设备 RNG。

## HF Carrier And Detector Boundary

CEG-WM 自有 HF carrier 与 HF direct score 当前承担 HF 主检测候选；正式联合判定入口始终称为 content detector。冻结要求包括：

- 给定检测密钥产生可校准分数；
- 正确密钥与错误密钥可独立比较；
- 原图与回正图使用同一检测实现和同一配置身份；
- 具体代码、revision 和来源在迁入前重新登记和验证。

历史实现不因名称相同而自动被接纳。

首个 `hf_sparse_tail` 候选必须保持 historical DirectHF 来源的精确算法顺序：高频剩余经 sparse tail 后直接 L2 normalize，模板构造不中心化；仅 normalized-correlation 评分时中心化观测和重构模板。历史名称和未经本项目验证的成功证据不进入 CEG-WM 身份；任何中心化 sparse tail 的变体都是新的 CEG-WM HF 候选。

## LF Validation Questions

LF 的 `lf_low_pass`、`lf_null_whitened_matched_score`、`routing_stqr` 和
`content_combination_calibrated` 保留为既有候选身份；其中后两者对应的执行路线已
形成 producer-bound development 负证据，不再是 current candidate。继任内容路线
新增四个 `design_candidate_implementation_authorized` 身份，并获得仅限本地独立
revisions 的 implementation admission；四者尚未实现、未绑定 readiness、未获 runtime
qualification、实验执行准入或科学晋升。既有 LF score 只对 32 个独立 clean public RGB-to-VAE observations 的
固定 channel-band diagonal null operator 做只读白化 matched score；它不改变 carrier，
也不把 fit images、参考图或私有 latent 引入检测。实验需要回答：

1. LF 在哪些失真下提供 HF direct score 缺少的互补证据？
2. LF 是否保持密钥归属，而不是只检测通用低频偏移？
3. LF 对图像质量、可见性和错误密钥 FPR 的代价是什么？
4. 显著目标内部局部 LF 是否在固定总预算下提供可审计的因果贡献？
5. 独立 primary-null 标准化后的固定 max statistic 是否保留 HF 证据并增加
   masked-LF attribution？

在这些问题以真实证据关闭前，不得把任一组合定义为正式内容检测器；这不允许实现者偏离已登记候选。

## Routing Requirements

内容自适应路由必须：

- 由冻结 InSPyReNet 对各自普通 RGB8 输入独立得到唯一显著目标 mask；
- 明确嵌入端与检测端重跑同一模型/规则且不得共享私有 mask；
- 不读取攻击后的不可获得私有状态；
- 保存可审计的路由身份与分支覆盖；
- 以 masked-LF causal witness、HF-only、global-HF+local-LF、LF-disabled 和失败
  构成 current 固定分母。

历史首个 `routing_stqr` 只在嵌入端消费已登记 S/T/R/Q 观测；检测使用未 mask 模板并不重建私有 route。其公式、reference 拟合职责和同预算控制见候选规格。

旧 `routing_stqr` 的权威输出只包括生成时观测结果、`A`、`mask_lf`、`mask_hf`、
route identity/digests 和不读取 observations 的 disabled uniform control。它不选择
`a`，不返回标量预算，也不记录 target/realized 写入量。同预算 routed/disabled 比较由
`content_embedder` 的共同总预算和实验配对共同保证。

### Current Salient-Object Local-LF Candidate

当前内容自适应候选为 `routing_inspyrenet_salient_local_lf`。其外部资产只允许：

- Hugging Face `plemeri/InSPyReNet` revision
  `d94c2baaa4d023ab018c6f97be6ef37548e3bd1f` 的 `ckpt_base.pth`，LFS object
  SHA-256 `0a6fe2a73ab0532d6d0b8d82849a9760a226df719e3063d09b4149ece6f80fcd`、
  size `367520613` bytes、MIT；
- source `plemeri/transparent-background` revision
  `f0fa91701a98cfc8e955c554e84522f365ec6da3`、MIT。

Windows `Zone.Identifier`、下载路径和本地文件时间不属于 checkpoint 或方法身份，
不得打包、哈希进候选或传播到 records。checkpoint 必须 strict `state_dict` load。
唯一前向调用是 `InSPyReNet.forward_inspyre(x)`，取返回
`out["saliency"][-1]` 的 raw finest `d0` logit，再调用 `torch.sigmoid` 恰一次；禁止
`Remover.process`、`model.forward`、`forward_inference` 及其逐图 min-max。

嵌入 mask 输入是 callback 18 非 terminal latent 的临时 VAE decode RGB8；检测 mask
输入是普通待检 RGB8。两侧分别执行同一冻结规则：RGB/static `1024 x 1024`、ImageNet
mean/std、float32；probability bilinear resize 到 `64 x 64`，
`align_corners=false`；hard threshold `p>=0.5`；固定 3x3 square erosion 一次并使用
zero padding。不存在连通域选择。eroded mask coverage 必须在 `64..3072` spatial
pixels；无支持或非局部支持 fail closed 并保留固定分母，禁止 global LF fallback。
raw 与 rectified 图像分别重跑模型和 mask；detector 不得读取 embed mask。
development mask-stability 门固定为 IoU `>=0.5`，8-unit pilot 至少 `7/8`。

写入身份 `content_embedding_global_hf_local_lf` 唯一为：

```text
u_hf = normalize(T_hf)
u_lf = normalize(M_embed * T_lf)
u_content = normalize(u_hf + u_lf)
```

最终 actual-dtype total budget 仍为 canonical binary32 `3/250`。masked LF 或 sum
为零/非有限均 fail closed；不存在 `0.70/0.30`、`0.50/0.50`、`a/w` grid。scientific
probe 必须包含 LF-only causal witness：actual LF delta 非零、mask 外逐 bit 为零、
mask 内有能量；combined arm 不得伪分解 actual branch contribution。

检测身份 `lf_saliency_masked_null_whitened_matched_score` 在检测侧重新得到
`M_detect`，并把它同时作用于 public VAE posterior observation 与 key-only template。
它必须从独立的 32 clean null fit 重新拟合自己的 `W`，不得继承旧 unmasked `W`。
`z_hf` 与 `z_lf_masked` 来自相互独立的 primary-null 标准化；
`content_combination_saliency_max_standardized` 唯一统计为
`max(z_hf,z_lf_masked)`。未来 formal threshold 必须直接对该 max statistic 独立拟合。
检测不得读取 reference、Prompt、embed record、private latent、Q/K 或 embed mask。

## Combination Requirements

以下 `a/u_content(a)` 与 calibrated-function 语义只服务 historical exact replay。
组合写入和统计组合仍是两个独立职责。旧 `content_embedder` 只消费 router 与 LF/HF
carrier 方向，按冻结 `a` 构造 `u_content(a)`，保持共同总能量预算并在任一 active
方向或 combined 方向为零时 fail closed；`a` 与 `1-a` 是 mixing coefficients，
combined norm 包含两方向的交叉项。它独占 HF-only/LF-only/combined delta、
nominal/limit、actual-dtype materialization reconciliation 与 realized combined
total norm/relative L2 核验。
runtime 只物化 delta
并返回 actual-dtype 张量及 combined delta 的 total norm/relative L2，不改变预算或
方向，也不提供未定义的分支级实际写入量。embedder 不得计算检测统计。
`lf_detector` 独立从普通待检图像、
检测 key 和公共资产计算盲 `s_lf`；raw normalized-correlation 与新登记的
clean-null-whitened matched score 必须保持不同 candidate identity，后者只能只读
消费冻结 `W` artifact，不得在待检样本上拟合。`content_detector` 消费独立可观测的 `s_lf`
与 `s_hf`，负责标准化、冻结组合和正式 `D_M` 身份，不得隐藏分支失败或错误密钥归属。

共同 nominal relative L2 与 actual-dtype combined content hard limit 均冻结为
`3/250`。runtime 对 embedder 请求的 binary32 scale `s` 只物化
`z_s=cast_actual(fp32(z0)+s*delta_content_nominal)`，用独立 binary16 RNE
bitwise replay 和 row-major binary32 协议返回 actual delta/norm。embedder 以
`A<=f32((3/250)*norm32(fp32(z0)))` 直接比较；full scale 超限时在 binary32
`[0,1]` 二分至没有新 representable midpoint，返回最大非零可行 scale，或在
zero plateau 后仍无非零可行写入时 fail closed。不得用 realized ratio、
`q_budget`、`tau_actual_budget`、经验 tolerance 或实际强度下限代替硬比较。

hard limit 只约束 LF/HF/routing 最终合成的 combined content delta。
`content_direction`、`active_lf_direction`、`active_hf_direction` 与 target
component 只是 nominal formula witnesses，不构成 actual branch decomposition；
geometry delta 与现有 geometry/total budget 独立。`budget_utilization` 仅用于诊断，
低 utilization 不得作为未来实验的事后样本筛选条件。

任何 LF/HF 组合必须：

- 在 calibration split 上冻结；
- 在独立 evaluation split 上报告；
- 保存 LF、HF 和组合分数；
- 包含正确密钥、错误密钥和无水印负样本；
- 证明组合不会通过 LF 分数掩盖 HF 错误密钥失败；
- 不使用针对回正图单独拟合的权重或阈值。

当前正式 content detector 仍为 HF-only。旧 `content_combination_calibrated` 只作
producer-bound 历史复现；当前新设计不重新搜索旧函数族，而只允许上述 max statistic。
新 masked-LF 必须先独立通过 key attribution、32-clean null fit、mask stability 与
causal witness 门，才允许新 max statistic 进入 confirmation。

该映射唯一采用 `content_combination_calibrated` 的有限样本 mid-rank empirical CDF、
`1/(2n)` tail clipping 和与 key schedule 同摘要的 `2^20` midpoint float32 normal
quantile table。candidate-selection 的 provisional
CDF/threshold 在 confirmation 后丢弃；正式 CDF 与 `tau` 只从独立
content-threshold-fit 重新拟合，不能跨五类 calibration 职责复用。

组合只有在 candidate-selection manifest 内预登记、未参与拟合的 confirmation partition 中证明攻击增益、HF-only 非退化和错误密钥分离，随后在独立职责数据上完成阈值、rescue、geometry reliability 与 end-to-end calibration check，才能替换 HF-only content detector。formal evaluation 只报告冻结方法，不再决定选择哪个候选。替换后必须为原图和回正图共同定义一个新的 content detector 身份并重新 calibration；不得沿用 HF-only 阈值。

若 LF 或 routing 未晋升，应登记为内容分支 `research_question_closed_negative`。该负结果可发表、可进入消融和失败分析，但不能通过与完整方法相同的成功门。HF-only 加 geometry 不是“完整 CEG-WM”；若要继续为 reduced-scope 方法，必须重新命名、缩小论文主张并独立获得研究定义与构建授权。

### Producer-Bound Historical Negatives

| route | producer | frozen result | current authority |
| --- | --- | --- | --- |
| `routing_stqr` fixed-half directional diagnosis | producer `925c2cbc727e3b18e91c0b3981eeed1b470a955a`; run `ceg_wm_content_routing_positive_reference_support_correction_diagnosis` | `42/42`; ordered indicator sequence `1,1,1,0,0,0,0,0`; `3/8=0.375` does not satisfy strict `>0.5`; RGB relative-L2 violations at clusters `1`,`5`,`6`, so they are not successful clusters | historical development negative; not current candidate |
| `content_uniform_combination` directional diagnosis | producer `7c0d86d6eac5ffcfc4a30f2f5fb22884aaa848da`; run `ceg_wm_content_uniform_combination_budget_observation_correction_diagnosis` | `1+32+8=41` attempt-0 `COMMITTED`; canonical binary32 `3/250`; budget violations `2` at clusters `1`,`6`; `mechanism_signal_not_observed`; `candidate_not_recommended_for_selection`; request false | historical development negative; not current candidate |

不得从旧 8-probe 结果选择 mask、threshold、erosion、coverage、`a`、`w` 或 function；
不得补样、删 cluster、重跑、增加 attempt、放宽 `3/250`、重写 artifact，或只用
margin-passing 子集形成 winner/promotion。既有 HF 与 LF 各自 32-unit directional
证据继续 producer-bound 有效，但不自动证明新 masked-LF、max statistic 或完整内容
链。旧代码只可用于 producer replay、failure provenance、historical exact-package /
record replay 和与新候选的语义 diff；代码存在不等于 current candidate 或执行授权。
旧 routed content embedder、旧 combined detector 与 conditional-recovery 内容依赖
保持 closed/paused，不得凭现存实现恢复。

## Output Semantics

historical CPU/synthetic 内容检测结果独立携带 LF、HF、combined 分支统计及旧路由、组合、
密钥和失败身份；正式 calibration 阈值与实验 records schema 仍在后续协议阶段登记。
current pending candidate 的 record/control matrix 固定为 clean、HF-only、masked-LF
causal、global-HF+local-LF、LF-disabled 和失败；不含 `A/a/routed/route-disabled`。

## Current Status

HF carrier、HF direct score、旧 LF/路由/组合写入与分支检测已完成 CPU/synthetic
实现和 27 节点内的对应行为验证。正式 detector 仍为 HF-only；新显著目标局部 LF
四候选尚未实现或晋升，`full_ceg_wm_eligible=false`。actual-dtype combined content
写入、完整性和 hard-budget 路径已在冻结 SD3.5 candidate 的真实 GPU
qualification 中通过；这不构成 LF/routing/组合晋升。当前仍没有正式 calibration、
完整联合 FPR 或科学效果证据；实际阶段/status 已由独立 revisions 同步为
`experiment_ready / implemented`。该阶段只登记冻结协议与可追溯执行交付的基础设施
闭环，不提供 `tau`、confirmation 结果、Calibration Locked 或正式 evaluation，也不
晋升 LF/routing/组合/geometry。
