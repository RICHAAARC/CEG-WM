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
现为 `ceg_wm_internal_sample_record_v3`，补充 candidate/input-manifest/execution/
resource provenance。该配置与 CPU schema tests
不构成 calibration、正式运行或科学有效性证据。

`internal_execution_components.json` 冻结 A-2 内部执行组件 registry：

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
