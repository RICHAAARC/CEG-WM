# Experiment Configurations

此目录保存 CEG-WM 内部设计验证和外部 `ComparisonProtocol` 的冻结配置。部署到
Colab 或 GPU 服务器时，应携带已经填充并通过 preflight 的项目配置，而不是在运行后
补写条件。

`internal_scientific_validation_protocol.json` 固定八个职责 split、source-cluster identity、
当前执行访问范围、四种 execution status、资源/执行/科学 failure class、每个 unit 的最多三次 record attempts、
逐样本与 run/case collection schema、实际 protocol/manifest 摘要绑定、非初始
attempt 的 parent 要求与可重试 parent outcomes、13 职责身份及 promotion
prerequisites。
协议加载会逐字段核对全部 canonical 语义（包括集合顺序、promotion failure 与
scientific claim boundary）；正式 collection validator 只接受精确冻结 dataclass
作为 protocol/manifest trust anchors，不接受 duck object 或 subclass。
它显式禁止当前执行访问 `held_out_evaluation`（最终 held-out 职责）。逐样本 schema
现为 `ceg_wm_internal_sample_record_v4`，补充 candidate/input-manifest/execution/
resource provenance，并逐 record 保存几何操作身份和 experiments 层可靠性配置摘要。该配置与 CPU schema tests
不构成 calibration、正式运行或科学有效性证据。

`hf_only_reference_validation.json` 是 hf_only_reference_protocol 的 HF-only reference 预注册入口；它绑定
`hf_sparse_tail` 的完整 write/embed/detect/formal-content/runtime source bundle、
已通过的 runtime qualification 身份、稳定 method-adapter 子配置摘要、actual dtype/
hard budget、独立 primary-null 与 wrong-key 统计门、paired attribution 和 final
RGB8 paired-quality 公式。它把 threshold fit 与 untouched confirmation 固定为两个
不可混跑阶段：前者只能读取 `content_threshold_fit`，后者必须等待独立审计的
`hf_only_tau_frozen` artifact 且禁止 refit。两个 compact manifest 从提交的 pinned
PartiPrompts TSV snapshot 和带明文 prompt 的 1632-row roster 离线物化，各含 4096
source clusters，prompt 完全互斥、并集覆盖全部 roster，且每个
`(category, challenge)` stratum 计数差不超过 1。当前文件只冻结 hf_only_reference_protocol 协议和预算；
不授权 hf_only_reference_metrics、hf_only_threshold_fit_gpu_execution、GPU、held-out、baseline 或科学结果。

`internal_execution_components.json` 冻结 internal_execution_components 内部执行组件 registry：

- 项目方法薄适配器及 13 个职责、公开 callable、结果身份字段的 canonical
  精确三元组；wrong-key 操作显式记录
  `derive_wrong_key_material -> derive_wrong_key_stream` 两步 provenance；
- identity/crop/scale/rotation/组合攻击、参数边界、插值、padding、量化与尺寸策略；
- fixed-FPR、wrong-key、质量、routing、LF/HF、几何、可靠性、回正与 rescue 指标集合；
- 指标的 `AnalysisUnitIdentity`/case/source-cluster 分析单位、每个 metric
  的合法 split 集合和 `held_out_evaluation` 禁止访问。

三个 loader 都拒绝未登记配置键，并分别重算 adapter、attack 和 metric registry
摘要。该配置不包含固定 LF/HF 权重，不授权 runner、Notebook、GPU、baseline 或
held-out 执行。

## Contrastive LF branch-attribution preregistration

`contrastive_lf_branch_attribution.json` 与同名前缀 roster/manifest 冻结
`contrastive_lf_branch_attribution` 的协议面，尚不提供方法、runner、GPU 或
Stage-B 执行实现。它从登记的 PartiPrompts snapshot 中按 raw row 升序选取经既有
配置使用排除后的首 96 个唯一 prompt（rows 132--227），固定分成 32 null-fit、
32 candidate-selection 与 32 untouched-confirmation；三个 split 的 prompt、source
row、seed、lineage 与 source-cluster identity 全部互斥。

Null-fit 预分配 32 clean observations 与 96 个 HF/multiscale/single raw null
statistics。Selection 预分配 128 base generations、512 attacked observations、
3840 detector records、96 budget records 与 384 paired-quality records；只有经认证
selection artifact 指定的 winner 可进入 confirmation，其分母分别为 96、384、
2560、64 与 256。每个 slot 只允许一次 attempt；首个失败保留 bounded reason，
其后只允许完整 unstarted tail，不得截断、补样、替换或 resume。

两项 LF candidate 的 internal decoy 与 external wrong-key roster 身份分离；阈值、
population standardization、identity attribution、逐 condition null/wrong、blur
complement、quality 与 binary32 budget 规则均由配置和 protocol loader 精确认证。
该工作只建立 diagnostic/provisional 预登记：不创建 formal tau/FPR，不 promotion，
不改变正式 HF-only detector，也不授权 Stage-B、模型运行、GPU 或 Colab。旧
soft-route negative、W/CDF/tau/roster/result 均不得继承。

`contrastive_lf_branch_attribution_execution.json` 与
`contrastive_lf_branch_attribution_execution_components.json` 只绑定 Stage-A
null-fit + candidate-selection 的真实 public adapter/runtime/method/runner 执行面。
它们不授权 untouched confirmation、Stage-B、formal detector 或 promotion。
