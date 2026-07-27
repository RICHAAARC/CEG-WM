# Project Reference

此目录保存当前确实存在的登记表和边界参考：

| document | responsibility |
| --- | --- |
| [field_registry.md](field_registry.md) | 持久化研究字段的等级、用途与说明。 |
| [test_inventory.md](test_inventory.md) | 当前测试入口、等级和实时收集命令。 |
| [baseline_registry.md](baseline_registry.md) | 外部 baseline 来源、revision、适配和偏差登记。 |
| [migration/](migration/README.md) | 历史 source registry、历史 13 组件 migration plan，以及未来逐组件 migration manifest 与当前 readiness 的分离边界。 |
| [extraction_profiles.md](extraction_profiles.md) | 三种可执行拆包 profile 的精确内容。 |
| [artifact_evidence.md](artifact_evidence.md) | Records、manifests、artifacts 与 claims 的研究证据语义。 |

迁移 reference 中：source registry 只保存来源准入事实；component migration plan
只保存历史 `not_started` 计划；实际迁移后的 provenance 必须进入 future
per-component manifest；实现完成门禁只进入当前唯一 method readiness。四者不得互相
替代。

本目录不保存运行生成的审计报告、实验 records 或具体论文结果。
