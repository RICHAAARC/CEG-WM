# Method: 语义显著性驱动的可验证流形水印

## 1. 方法概览（对应当前项目真实实现）

本文方法在工程上实现为一条严格受控的主链：

1. `Prompt -> SD3 推理轨迹采样`（支持每步 latent 回调）。
2. `语义显著性掩码 -> 自适应子空间规划（Subspace Planner）`。
3. `LF/HF 双通道潜空间注入（逐步回调注入）`。
4. `检测端内容证据 + 几何证据并行构建`。
5. `Neyman-Pearson (NP) 融合判决`。

在当前配置 `configs/paper_full_cuda.yaml` 下，系统启用：

- 基础模型：`stabilityai/stable-diffusion-3.5-medium`。
- 子空间与 LF/HF 通道：`watermark.subspace/lf/hf.enabled=true`。
- 显著性来源：`mask.saliency_source=model_v2`（严格模式，非自动 proxy）。
- 几何链：`sync_primary_anchor_secondary=true`，`sync_module_id=geometry_latent_sync_sd3_v2`。
- 融合：`fusion_rule_id=fusion_neyman_pearson_v1`。

## 2. 框架图（Method Framework）

```mermaid
flowchart TD
    A[Prompt / Seed / Config] --> B[SD3 Inference Runtime]
    B --> C[Trajectory Tap<br/>trajectory_evidence]
    B --> D[Step-end Callback]
    A --> E[SemanticMaskProvider<br/>mask_digest + routing_digest]
    C --> F[SubspacePlanner<br/>trajectory+JVP+SVD]
    E --> F
    F --> G[plan_digest + basis_digest<br/>LF/HF basis + region spec]
    G --> D
    D --> H[LatentModifier<br/>LF encode + HF truncation]
    H --> I[Watermarked Image]

    I --> J[Detect Runtime Inference]
    J --> K[final_latents + trajectory_evidence]
    G --> L[Detect-side Plan/Basis Consistency Check]
    K --> M[ContentDetector<br/>LF/HF score]
    L --> M

    J --> N[Geometry Chain<br/>Sync(primary)+Anchor(secondary)+Align]
    M --> O[FusionDecision (NP)]
    N --> O
    O --> P[final_decision<br/>is_watermarked / decision_status]

    Q[Calibration Detect Records] --> R[NP Threshold Artifact]
    R --> O
    R --> S[Evaluate: TPR@FPR, FPR empirical]
```

## 3. 问题定义与符号

给定文本提示词 \(p\)、随机种子 \(s\)、配置 \(c\)，扩散推理得到潜变量序列：

\[
\mathbf{z}_{0:T} = \{\mathbf{z}_t\}_{t=0}^{T}
\]

系统需要在不破坏可审计性的前提下，将水印嵌入到由语义显著性决定的子空间中，并在检测端输出可追责的最终判决 \(y \in \{0,1,\varnothing\}\)（分别对应 watermarked / non-watermarked / abstain）。

## 4. Prompt 驱动推理与轨迹采样

在 `run_sd3_inference(...)` 中，系统从配置读取：

- `inference_prompt`
- `inference_num_steps`
- `inference_guidance_scale`
- 图像分辨率

并通过 `callback_on_step_end` 在每步获取 latent，实现：

1. 注入回调（修改 latent）；
2. 轨迹证据回调（构建 `trajectory_evidence`）；
3. 检测模式下捕获最终 latent（`final_latents`）。

若 embedding 阶段未提供输入图，且 `embed.preview_generation.enabled=true`，系统先做一次 preview 推理生成输入图，再执行主嵌入流程。这使主链在 `main/` 内具备闭环输入能力。

## 5. 语义掩码与路由绑定

### 5.1 语义显著性提取

`SemanticMaskProvider.extract(...)` 输出：

- `status in {ok, absent, failed}`
- `mask_digest`
- `mask_stats`（含路由摘要、显著性来源选择、分辨率绑定）

在当前配置中，`saliency_source=model_v2`，意味着模型不可用时走失败语义而非静默降级。

### 5.2 路由摘要

系统将掩码统计压缩为路由摘要并计算摘要哈希：

\[
d_{\text{mask}} = \mathrm{SHA256}(\mathrm{CanonJSON}(\text{mask\_summary}))
\]

\[
d_{\text{routing}} = \mathrm{SHA256}(\mathrm{CanonJSON}(\text{routing\_summary}))
\]

这两个摘要作为后续子空间规划和审计绑定锚点。

## 6. 自适应子空间构建（Subspace Planner）

`SubspacePlannerImpl.plan(...)` 的核心是“轨迹 + JVP + SVD”：

### 6.1 轨迹矩阵

从推理轨迹得到样本矩阵 \(\mathbf{X}\in \mathbb{R}^{n\times d}\)，中心化后：

\[
\tilde{\mathbf{X}} = \mathbf{X} - \mathbf{1}\mu^\top
\]

### 6.2 Jacobian 向量积（JVP）估计

构造 JVP 样本 \(\mathbf{J}\in \mathbb{R}^{m\times d}\)，来源优先级：

1. runtime operator（可调用 JVP）；
2. real unet JVP；
3. surrogate transition（后备）。

### 6.3 SVD 子空间

拼接分解矩阵：

\[
\mathbf{M} = \begin{bmatrix}\tilde{\mathbf{X}}\\ \mathbf{J}\end{bmatrix}
\]

奇异值分解：

\[
\mathbf{M} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top
\]

取前 \(r\) 维得到主子空间基：

\[
\mathbf{B}_{\text{LF}} = \mathbf{V}_{1:r}^\top
\]

并构建 HF 对应基 \(\mathbf{B}_{\text{HF}}\)（来自尾部或回退策略）。

能量占比：

\[
\eta_r = \frac{\sum_{i=1}^{r}\sigma_i^2}{\sum_{i=1}^{k}\sigma_i^2}
\]

最终输出 `plan_digest`、`basis_digest`、`plan_stats`、`region_index_spec`。  
其中摘要域绑定了轨迹锚点、JVP 源、配置域、mask 路由等信息，形成可复核闭环。

## 7. 双通道潜空间注入（LF/HF）

`LatentModifier.apply_latent_update(...)` 在每步推理回调执行注入。

设当前 latent 向量为 \(\mathbf{z}\)：

### 7.1 LF 通道（编码约束）

投影：

\[
\mathbf{c}_{\text{LF}} = \mathbf{B}_{\text{LF}}^\top \mathbf{z}
\]

编码后系数：

\[
\hat{\mathbf{c}}_{\text{LF}} = \mathcal{E}_{\text{LF}}(\mathbf{c}_{\text{LF}}, k)
\]

重构增量：

\[
\Delta \mathbf{z}_{\text{LF}} = \mathbf{B}_{\text{LF}}\hat{\mathbf{c}}_{\text{LF}} - \mathbf{B}_{\text{LF}}\mathbf{c}_{\text{LF}}
\]

### 7.2 HF 通道（截断约束）

投影：

\[
\mathbf{c}_{\text{HF}} = \mathbf{B}_{\text{HF}}^\top \mathbf{z}
\]

约束（如 top-k / 尾部截断）：

\[
\hat{\mathbf{c}}_{\text{HF}} = \mathcal{T}_{\text{HF}}(\mathbf{c}_{\text{HF}})
\]

重构增量：

\[
\Delta \mathbf{z}_{\text{HF}} = \mathbf{B}_{\text{HF}}\hat{\mathbf{c}}_{\text{HF}} - \mathbf{B}_{\text{HF}}\mathbf{c}_{\text{HF}}
\]

### 7.3 语义区域门控

对增量施加区域掩码 \( \mathbf{m}_{\text{region}} \)：

\[
\Delta \mathbf{z} = (\Delta \mathbf{z}_{\text{LF}} + \Delta \mathbf{z}_{\text{HF}})\odot \mathbf{m}_{\text{region}}
\]

\[
\mathbf{z}' = \mathbf{z} + \Delta \mathbf{z}
\]

逐步注入证据会写入 `injection_evidence`，最终形成 `injection_trace_digest` 与 `subspace_binding_digest`。

## 8. 检测端内容证据链

## 8.1 输入构造

`_build_content_inputs_for_detect(...)` 优先级为：

1. 显式 `image` / `latent`
2. `input_record.inputs`
3. `image_path` 解析
4. 从输入记录注入 `expected_plan_digest`、`observed_plan_digest`、LF/HF 证据

## 8.2 一致性校验与状态语义

检测端对以下锚点做一致性检查：

- `plan_digest`
- `basis_digest`
- trajectory digests
- 注入 trace digests

若关键锚点 mismatch，走 guard 决策（`decision_status=error`）；若证据 absent，走 `abstain`。

## 8.3 内容评分

`ContentDetector.extract(...)` 对 LF/HF 合成规则：

- HF absent：\(s_c = s_{\text{LF}}\)
- HF available：

\[
s_c = 0.7\, s_{\text{LF}} + 0.3\, s_{\text{HF}}
\]

其中任一关键失败语义（mismatch/failed）会阻断为 `score=None`，而不是输出伪稳定分数。

## 9. 几何证据链：Sync 主链 + Anchor 辅助

当前方法采用 `sync_primary_anchor_secondary`：

1. **Primary**：`GeometryLatentSyncSD3V2` 计算 sync 质量并绑定 `relation_digest`；
2. **Secondary**：attention anchor 与对齐器做一致性补充；
3. 不满足可信条件时输出失败/不匹配语义。

sync 模块显式不确定度门控：

\[
u = 1 - q_{\text{sync}}
\]

若 \(u > 0.5\) 则直接 `status="mismatch"`（`sync_uncertainty_too_high`），防止几何伪证据通过。

对齐阶段使用鲁棒拟合（IRLS + inlier/residual/variance 约束）构建 `geo_score`，并可给出 `geo_available` 供策略层使用。

## 10. Real/Fallback 运行模式判定

检测端 `detect_runtime_mode` 默认 `fallback_identity_v0`，仅在以下条件同时满足时切换 `real`：

\[
\text{detect\_lf\_status} = \text{ok}
\land \text{subspace\_consistency\_status} \neq \text{inconsistent}
\land \text{runtime\_built} = \text{true}
\land \text{synthetic\_pipeline} = \text{false}
\]

该门控避免“有路径但无可信数据流”的伪 real 判定。

## 11. 融合判决（NP 框架）

融合器 `fusion_neyman_pearson_v1` 的 `decision_status` 为：

- `decided`
- `abstain`
- `error`

NP 阈值 \( \tau \) 默认必须来自只读阈值工件 `__thresholds_artifact__`。  
基础判决：

\[
\hat{y}_{\text{NP}} = \mathbb{I}[s_c \ge \tau]
\]

系统支持单侧 rescue band（仅可将 false -> true，不能反向）：

\[
\hat{y} = 
\begin{cases}
1, & s_c \in [\tau-\delta, \tau),\ \text{geo\_gate}=1 \\
\hat{y}_{\text{NP}}, & \text{otherwise}
\end{cases}
\]

最终输出 `final_decision`（只读投影自 `fusion_result`）：

- `decision_status`
- `is_watermarked`
- `routing_decisions`
- `threshold_source`

## 12. 校准与评估：可复算统计闭环

### 12.1 NP 校准

用负类（null）分数集合 \(\{s_i\}_{i=1}^n\) 计算目标 FPR 的“higher”分位阈值。  
本质上是次序统计量：

\[
\tau = Q_{1-\alpha}^{\text{higher}}(\{s_i\})
\]

其中 \(\alpha=\text{target\_fpr}\)。

输出阈值工件包含：

- `threshold_id`
- `target_fpr`
- `threshold_value`
- `threshold_key_used`
- `order_statistics`

### 12.2 评估

评估阶段只读阈值工件，计算：

\[
\mathrm{TPR}@\mathrm{FPR} = \frac{\mathrm{TP}}{\mathrm{P}},\quad
\mathrm{FPR}_{emp} = \frac{\mathrm{FP}}{\mathrm{N}}
\]

并可按攻击条件与几何可用性分组输出条件指标。

## 13. 当前实现的创新点（相对常见固定强度/固定子空间方案）

### 13.1 语义驱动的动态子空间注入

不是固定低维子空间，而是通过显著性掩码 + 轨迹/JVP/SVD 动态构建样本级子空间，并将 `mask_digest -> plan_digest -> injection_digest` 串联绑定。

### 13.2 LF/HF 分治的内容证据机制

LF 通道偏稳定编码，HF 通道偏敏感约束，检测端保留显式失败语义（mismatch/failed/absent），避免把低可信连续分数伪装成可判决证据。

### 13.3 Sync 主几何、Anchor 辅几何的层级化证据

几何链采用“sync 主导 + anchor 辅助”的结构，并在 uncertainty、inlier、residual、fit-stability 上多级门控，符合“可拒绝而非伪鲁棒”的设计。

### 13.4 决策层与统计层的可审计冻结

NP 决策只读阈值工件、判决状态枚举固定、final_decision 只读投影，减少运行时语义漂移。  
这使方法不仅“能检测”，还“可复核、可追责”。

### 13.5 生成事件级证明（可选扩展）

代码中提供 cryptographic generation attestation（事件绑定而非仅“是否含水印”），可将 LF/HF/GEO 多通道得分融合为事件归属判定，提升来源可证明性。

## 14. 训练/部署视角总结

本项目当前实现不是“单一打分器”，而是一个**带证据绑定与失败语义的端到端可验证水印系统**：

1. Prompt 驱动推理产生可绑定轨迹；
2. 语义显著性决定子空间与注入区域；
3. LF/HF/Geometry 分链提证；
4. NP 融合输出唯一最终判决；
5. 校准/评估在只读阈值工件下可复算。

该结构使“方法创新（语义自适应流形注入）”与“工程可审计性（digest/冻结语义/统计闭环）”在同一主链中闭合。

