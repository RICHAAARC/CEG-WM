# Artifact And Evidence Semantics

## 证据链

1. `ExperimentRecord` 是实验结果事实来源。
2. Tables 从 records 或带 provenance 的 tables 重建。
3. Figures 从 records 或 tables 重建。
4. Reports 从 records、tables、figures 和 manifests 生成。
5. Manifests 记录输入、输出、`code_version`、`config_digest` 和 `rebuild_command`。
6. Supported claims 只能引用上述链路产生的 artifacts。

测试、audit、日志、Notebook 临时输出、目录存在和代码数量不能替代实验 records。

通用 record 契约位于 `experiments/protocol/records.py`，artifact manifest 与 digest helper 位于 `paper_artifacts/`。具体 table、figure 和 report builders 也应位于 `paper_artifacts/`；该层只能消费协议和冻结证据，不重新运行方法、模型或攻击。

具体流程见 `docs/guides/artifact_rebuild.md`。

从冻结方法与协议到 release packages 的派生操作图见
[paper evidence production](../guides/diagrams/paper_evidence_production.svg)；该图
不能替代本文件、governed records 或 frozen manifests。
