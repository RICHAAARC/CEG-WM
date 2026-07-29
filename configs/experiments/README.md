# Experiment Configurations

此目录保存 CEG-WM 内部设计验证和外部 `ComparisonProtocol` 的冻结配置。部署到
Colab 或 GPU 服务器时，应携带已经填充并通过 preflight 的项目配置，而不是在运行后
补写条件。

`internal_scientific_validation_protocol.json` 固定八个职责 split、source-cluster identity、
当前执行访问范围、四种 execution status、每个 unit 的最多三次 record attempts、
逐样本与 run/case collection schema、实际 protocol/manifest 摘要绑定、非初始
attempt 的 parent 要求与可重试 parent outcomes、13 职责身份及 promotion
prerequisites。
它显式禁止当前执行访问 `held_out_evaluation`（最终 held-out 职责）。该配置与 CPU schema tests
不构成 calibration、正式运行或科学有效性证据。
