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
`content_combination_calibrated` 候选算法已经关闭，但仍是待证伪内容载体，不预设
已经晋升。新增 LF score 只对 32 个独立 clean public RGB-to-VAE observations 的
固定 channel-band diagonal null operator 做只读白化 matched score；它不改变 carrier，
也不把 fit images、参考图或私有 latent 引入检测。实验需要回答：

1. LF 在哪些失真下提供 HF direct score 缺少的互补证据？
2. LF 是否保持密钥归属，而不是只检测通用低频偏移？
3. LF 对图像质量、可见性和错误密钥 FPR 的代价是什么？
4. LF 是否需要内容路由，路由是嵌入时、检测时还是共享确定性规则？
5. LF 与 HF 的组合是否优于独立选择或条件路由？

在这些问题以真实证据关闭前，不得把任一组合定义为正式内容检测器；这不允许实现者偏离已登记候选。

## Routing Requirements

内容自适应路由必须：

- 由可复现的图像或模型观测决定；
- 明确嵌入端与检测端如何获得一致路由；
- 不读取攻击后的不可获得私有状态；
- 保存可审计的路由身份与分支覆盖；
- 允许 LF-only、HF-only、route-disabled 和组合消融。

首个 `routing_stqr` 只在嵌入端消费已登记 S/T/R/Q 观测；检测使用未 mask 模板并不重建私有 route。其公式、reference 拟合职责和同预算控制见候选规格。

`content_router` 的权威输出只包括生成时观测结果、`A`、`mask_lf`、`mask_hf`、
route identity/digests 和不读取 observations 的 disabled uniform control。它不选择
`a`，不返回标量预算，也不记录 target/realized 写入量。同预算 routed/disabled 比较由
`content_embedder` 的共同总预算和实验配对共同保证。

## Combination Requirements

组合写入和统计组合是两个独立职责。`content_embedder` 只消费 router 与 LF/HF
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

当前正式 content detector 候选仍为 HF-only。候选组合应先把 LF、HF 分支统计分别映射到 calibration null 下方向一致的统计，再比较冻结、单调且攻击无关的组合函数。LF 必须先独立通过 key attribution 和无水印校准门，才允许参与组合选择。

该映射唯一采用 `content_combination_calibrated` 的有限样本 mid-rank empirical CDF、
`1/(2n)` tail clipping 和与 key schedule 同摘要的 `2^20` midpoint float32 normal
quantile table。candidate-selection 的 provisional
CDF/threshold 在 confirmation 后丢弃；正式 CDF 与 `tau` 只从独立
content-threshold-fit 重新拟合，不能跨五类 calibration 职责复用。

组合只有在 candidate-selection manifest 内预登记、未参与拟合的 confirmation partition 中证明攻击增益、HF-only 非退化和错误密钥分离，随后在独立职责数据上完成阈值、rescue、geometry reliability 与 end-to-end calibration check，才能替换 HF-only content detector。formal evaluation 只报告冻结方法，不再决定选择哪个候选。替换后必须为原图和回正图共同定义一个新的 content detector 身份并重新 calibration；不得沿用 HF-only 阈值。

若 LF 或 routing 未晋升，应登记为内容分支 `research_question_closed_negative`。该负结果可发表、可进入消融和失败分析，但不能通过与完整方法相同的成功门。HF-only 加 geometry 不是“完整 CEG-WM”；若要继续为 reduced-scope 方法，必须重新命名、缩小论文主张并独立获得研究定义与构建授权。

## Output Semantics

当前 CPU/synthetic 内容检测结果独立携带 LF、HF、combined 分支统计及路由、组合、
密钥和失败身份；正式 calibration 阈值与实验 records schema 仍在后续协议阶段登记。

## Current Status

HF carrier、HF direct score、LF、路由、组合写入与分支检测已完成 CPU/synthetic
实现和 27 节点内的对应行为验证。正式 detector 仍为 HF-only；LF/routing/组合
尚未通过实验晋升，`full_ceg_wm_eligible=false`。actual-dtype combined content
写入、完整性和 hard-budget 路径已在冻结 SD3.5 candidate 的真实 GPU
qualification 中通过；这不构成 LF/routing/组合晋升。当前仍没有正式 calibration、
完整联合 FPR 或科学效果证据；实际阶段/status 已由独立 revisions 同步为
`experiment_ready / implemented`。该阶段只登记冻结协议与可追溯执行交付的基础设施
闭环，不提供 `tau`、confirmation 结果、Calibration Locked 或正式 evaluation，也不
晋升 LF/routing/组合/geometry。
