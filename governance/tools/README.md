# Governance Tools

此目录保存可拆卸外层工具。`extract_release_package.py` 读取研究项目并生成交付候选，但不会进入任何交付 profile 或拆除后的研究项目。

真正属于研究运行或 artifact rebuild 的脚本应按职责放在 `scripts/experiment_execution/` 或 `scripts/artifact_rebuild/`，并由对应 profile 显式纳入。
