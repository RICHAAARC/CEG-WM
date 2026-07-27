# Operational Guides

此目录只索引当前已经提供的操作指引，不重复定义 policy。

| guide | purpose |
| --- | --- |
| [colab_usage.md](colab_usage.md) | 保持 Colab Notebook 为薄编排入口。 |
| [runtime_gpu_qualification_workflow.md](runtime_gpu_qualification_workflow.md) | 从 `method_implemented` 本地 CPU 构建、固定 Colab 薄入口、Google Drive 结果回收到 `runtime_verified` 的完整操作门序。 |
| [artifact_rebuild.md](artifact_rebuild.md) | 从冻结 records 与 manifest 重建论文产物。 |
| [cpu_validation_environment.md](cpu_validation_environment.md) | 按 `governance`、`method`、`full` 档位运行 Conda CPU 验证，并限定 `.venv` 的定向轻量用途。 |
| [project_advancement.md](project_advancement.md) | 从 `research_defined` 开始的阶段准入、执行/审计分离和停止规则索引。 |
| [historical_code_migration.md](historical_code_migration.md) | 历史来源、复用权、候选/职责映射及未来迁移门禁。 |
| [method_readiness_workflow.md](method_readiness_workflow.md) | 当前唯一 readiness YAML 的创建点、绑定和独立复核生命周期。 |
| [paper_evidence_production.svg](diagrams/paper_evidence_production.svg) ([Drawio](diagrams/paper_evidence_production.drawio)) | 从冻结方法/协议到 records、artifacts、claims 和 release packages 的派生操作图。 |

拆包内容以 `docs/reference/extraction_profiles.md` 为准；外层执行工具不属于研究文档或交付内容。当前已有 runtime/GPU qualification 操作指南，但尚无单独的数据准备或正式实验执行指南。

指南和 guide 图不构成构建授权、runtime 验证、实验事实或论文证据。
