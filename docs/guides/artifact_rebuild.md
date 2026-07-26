# Artifact Rebuild Guide

## 前提

- 已冻结并通过 `ExperimentRecord` schema 校验的 governed records；
- 声明输入、输出、`config_digest`、`code_version` 和 `rebuild_command` 的 `ArtifactManifest`；
- 具体项目在 `paper_artifacts/` 实现的可独立运行 builder。

当前 CEG-WM 只提供 digest 与 manifest helper，没有具体 table、figure 或 report builder，因此不能单独完成论文产物重建。

## 具体项目流程

1. 校验 records、manifest schema 和输入 digest。
2. 确认 manifest 的 `code_version` 与当前 rebuild 环境一致。
3. 使用 manifest 声明的命令调用 `paper_artifacts/` builder。
4. 将 tables、figures 和 reports 写入未提交输出目录。
5. 校验输出路径与 digest，并生成或更新 manifest。
6. 只从通过校验的 governed artifacts 建立 claim evidence path。

Notebook 可以检查和展示结果，但不得成为 rebuild 实现或正式证据的唯一载体。证据语义见 `docs/reference/artifact_evidence.md`。

完整的 planned evidence flow 见
[paper evidence production](diagrams/paper_evidence_production.svg)
（[可编辑 Drawio](diagrams/paper_evidence_production.drawio)）。该图是操作指南，不是
实验事实或结果。
