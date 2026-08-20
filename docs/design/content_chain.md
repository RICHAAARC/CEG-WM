# Content Chain Design

历史 exact-replay 兼容身份仅限 `hf_only_standardized_score`、
`weighted_hf_lf_standardized_score` 和
`maximum_hf_lf_standardized_score`；历史冻结权重保存在独立 `weight`
字段中。这三个函数只用于历史生产者结果的精确重放，不是语义—纹理
soft-route 方法的可选 detector 身份。本方法的内容统计唯一为
`max(z_hf_soft,z_lf_soft)`。

## Frozen Responsibility

内容链是唯一水印证据来源。几何可靠性、配准质量或变换估计不得直接加入内容阳性分数。
LF/HF template、wrong-key roster 和 routing public probe 统一消费
`key_schedule_sha256_counter`；内容组件不得各自定义 key encoding 或设备 RNG。

## HF Carrier And Detector Boundary

CEG-WM 自有 HF carrier 与 soft-routed HF direct score 承担 HF 分支检测；正式联合
判定入口始终称为 content detector。冻结要求包括：

- 给定检测密钥产生可校准分数；
- 正确密钥与错误密钥可独立比较；
- 原图与回正图使用同一检测实现和同一配置身份；
- 具体代码、revision 和来源在迁入前重新登记和验证。

历史实现不因名称相同而自动被接纳。

首个 `hf_sparse_tail` 候选必须保持 historical DirectHF 来源的精确算法顺序：高频剩余经 sparse tail 后直接 L2 normalize，模板构造不中心化；仅 normalized-correlation 评分时中心化观测和重构模板。历史名称和未经本项目验证的成功证据不进入 CEG-WM 身份；任何中心化 sparse tail 的变体都是新的 CEG-WM HF 候选。

## LF Validation Questions

LF 采用独立职责域 keyed 低通模板，并以普通待检图像上的盲分数回答下列问题：

1. LF 在哪些失真下提供 HF direct score 缺少的互补证据？
2. LF 是否保持密钥归属，而不是只检测通用低频偏移？
3. LF 对图像质量、可见性和错误密钥 FPR 的代价是什么？
4. 语义—纹理软路由后的 LF 是否在固定总预算下提供可审计的因果贡献？
5. 独立 primary-null 标准化后的固定 max statistic 是否保留 HF 证据并增加
   routed-LF attribution？

在这些问题以真实证据关闭前，不得把任一组合定义为正式内容检测器；这不允许实现者偏离已登记候选。

## Routing Requirements

内容自适应路由必须：

- 从普通 RGB8 独立构造语义显著性 `M` 和纹理复杂度 `T`；
- 明确嵌入端、原图检测端和回正检测端分别重跑同一公共规则，不共享私有 route；
- 不读取攻击后的不可获得私有状态；
- 保存可审计的输入、route identity、map digests 和分支方向；
- 以 clean、HF-only、LF-only、soft-routed LF/HF、route-disabled causal control 和
  显式失败构成固定分母。

历史首个 `routing_stqr` 只在嵌入端消费已登记 S/T/R/Q 观测；检测使用未 mask 模板并不重建私有 route。其公式、reference 拟合职责和同预算控制见候选规格。

旧 `routing_stqr` 的权威输出只包括生成时观测结果、`A`、`mask_lf`、`mask_hf`、
route identity/digests 和不读取 observations 的 disabled uniform control。它不选择
`a`，不返回标量预算，也不记录 target/realized 写入量。同预算 routed/disabled 比较由
`content_embedder` 的共同总预算和实验配对共同保证。

### Semantic-Texture Soft Routing

语义图 `M` 使用冻结 InSPyReNet public probability：

- `plemeri/InSPyReNet`、`plemeri/transparent-background`、model/name/revision 与
  `ckpt_base.pth` 只作为 runtime locator/观测元数据，不进入方法或结果强身份。

Notebook 仅从 Drive 固定相对路径复制 regular non-symlink `ckpt_base.pth` 到 fresh
`/content`；不校验 blob SHA、size 或 revision。checkpoint 必须 `weights_only` 且 strict
`state_dict` load。
唯一前向调用是 `InSPyReNet.forward_inspyre(x)`，取返回
`out["saliency"][-1]` 的 raw finest `d0` logit，再调用 `torch.sigmoid` 恰一次；禁止
`Remover.process`、`model.forward`、`forward_inference` 及其逐图 min-max。

嵌入路由输入是 callback 18 非 terminal latent 的临时 VAE decode RGB8；检测路由输入
是普通待检 RGB8。各侧分别执行 RGB/static `1024 x 1024`、ImageNet mean/std、
float32、raw finest `d0`、sigmoid exactly once，并以 bilinear、
`align_corners=false` 映射到 `64 x 64`，得到 `M in [0,1]`。不执行 hard threshold、
erosion、connected-component selection、per-image min-max 或 coverage fallback。

纹理图 `T` 从同一个公共 RGB8 输入确定性构造：按 `(0.299,0.587,0.114)` 转灰度，
replicate-pad 1 pixel，使用标准 3x3 Sobel x/y，计算梯度幅值，以 area downsample
映射到 `64 x 64`。对严格正幅值按 row-major、值升序和 flat-index 平局求 exact
nearest-rank P95 `q95`，再令：

```text
T = clamp(G / q95, 0, 1)
```

若不存在严格正幅值，则 `T=0`；不得用攻击标签、分数、全数据 reference 或
evaluation 结果替代。`M`、`T` 都在检测端从当前普通图像重建。

逐像素软路由唯一为：

```text
m_hf = (1 + M*T) / (2 + M)
m_lf = (1 + M*(1-T)) / (2 + M)
m_hf + m_lf = 1
```

两条 map 始终非零；它们是空间调制，不是实际分支能量或可相加预算。

双分支写入唯一为：

```text
u_hf = normalize(m_hf_embed * T_hf)
u_lf = normalize(m_lf_embed * T_lf)
u_content = normalize(u_hf + u_lf)
```

最终 actual-dtype total budget 为 canonical binary32 `3/250`。任一 active direction
或 sum 为零/非有限均 fail closed；不存在 `0.70/0.30`、`0.50/0.50`、`a/w` grid。
route-disabled causal control 固定 `m_hf=m_lf=0.5`，且不得读取 `M/T`；它与软路由
使用相同 key、Prompt、seed、write position 和总预算。

检测端分别重建 `m_hf_detect` 与 `m_lf_detect`。HF 分数把 `m_hf_detect` 同时作用于
public VAE observation 和 key-only HF template；LF whitened matched score把
`m_lf_detect` 同时作用于 observation 和 key-only LF template。二者各自使用专属
primary null 标准化，内容统计唯一为：

```text
D_soft_route = max(z_hf_soft, z_lf_soft)
```

正式阈值必须直接在该 max statistic 上独立拟合。检测不得读取 reference、Prompt、
embed record、private latent、Q/K 或 embed-side `M/T/route`。

### Adopted Contrastive Attribution And Orthogonal Allocation Design

下一 live design 不重用上述 negative identity。branch-attribution foundation 只有两个
有限候选：独立 five-by-five/nine-by-nine KDF domains 的 multiscale LF，以及独立
five-by-five domain 的 single-scale LF。multiscale detector 的两项 blind correlations
必须 joint covariance-whiten 后沿 equal direction 得分；single-scale 只有 scalar blind
correlation。两者都使用 candidate-specific 32-clean null 和八个 internal decoys，且
external wrong-key 八项另用独立 domain/roster。selection hierarchy 与全部 exact 公式
见 [candidate_specifications.md](candidate_specifications.md)；不得把任一 scale、weight、
null 或 loser 变成结果后菜单。

只有 Stage-A winner confirmation 通过后，`content_adaptive_orthogonal_allocation` 才有
未来 implementation admission。router 对 embed/raw/rectified 各自当前 RGB8 生成
InSPyReNet public `M`、Sobel/P95 edge map `E`，并用 JPEG/blur public feature probes
形成 pointwise `S_lf/S_hf`。它输出 `q_lf/q_hf`、nominal `rho_lf/rho_hf` 与 digests，
不输出 actual branch energy。

embedder 先对 `q_lf*T_lf` 相对 `q_hf*T_hf` 做冻结 binary32 Gram-Schmidt residual，
residual norm 不超过 `2^-10` 即 fail closed，再按 `sqrt(rho)` 合成并走现有 combined
`3/250` hard budget。route-disabled 在入口直接构造 all-one `q`、`rho=0.5`，且不得
读取 `M/E/S/F/probes`；同 carrier、detector、orthogonalization、budget 和 sample
使其成为因果 control。

检测端独立重建 route，并以 fresh branch null 和 combined null 计算：

```text
C = sqrt(rho_hf_detect)*z_hf + sqrt(rho_lf_detect)*z_lf
```

Stage-A identity 可继承算法，但其 assets/CDF/threshold 不得进入该 Stage-B detector。
唯一 monotone fallback 只在 analytic main 的完整 scientific failure 后改变 scalar `rho`
authority；`beta_M=0` 且 inference 不读 attack label。它不改变 spatial `q`、carrier、
orthogonal write、blind detector 或 budget，也不授权第三路线。

上述七个 identity 均为 `adopted_design_unimplemented / not_yet_tested`。当前 formal/
default/joint detector 仍为 HF-only；本节没有 implementation、promotion 或 formal
threshold authority。

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

方法内容统计不搜索旧函数族，只允许上述 max statistic。两条软路由分支必须先分别
通过 key attribution、独立 primary null、路由因果性与总预算门，才允许 max
statistic 进入 confirmation。

soft-route 分支标准化只复用有限样本 mid-rank empirical CDF、`1/(2n)` tail
clipping 和 `2^20` midpoint float32 normal quantile table 这些共享统计原语；共享
原语及其 table digest 必须进入
`content_combination_semantic_texture_max_standardized` 的新 detector identity。
不得继承 `content_combination_calibrated` 的 candidate identity、CDF artifact、split、
threshold、选择结果或效果证据。两条 soft-route 分支必须从本候选专属且互斥的
primary-null 数据重新拟合 provisional CDF；confirmation 后丢弃 provisional CDF
与 threshold。正式 branch CDF 与 `tau` 只能从独立 content-threshold-fit 重新拟合，
不能跨 calibration 职责复用。

组合只有在 candidate-selection manifest 内预登记、未参与拟合的 confirmation partition 中证明攻击增益、HF-only 非退化和错误密钥分离，随后在独立职责数据上完成阈值、rescue、geometry reliability 与 end-to-end calibration check，才能替换 HF-only content detector。formal evaluation 只报告冻结方法，不再决定选择哪个候选。替换后必须为原图和回正图共同定义一个新的 content detector 身份并重新 calibration；不得沿用 HF-only 阈值。

## Output Semantics

内容检测结果必须独立携带 LF、HF、combined 分支统计、route identity、`M/T` digests、
密钥身份、预算和失败语义。
