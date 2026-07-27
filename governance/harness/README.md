# Governance Harness

此目录把合同和机器 policy 转化为结构化审计。`run_all_audits.py` 是完整入口；`inspect_repository.py` 提供仓库检查辅助能力；`audits/` 和 `lib/` 分别保存独立审计与共享读取逻辑。

运行：

```bash
.venv/bin/python governance/harness/run_all_audits.py
```

审计包括 CEG-WM 研究定义、语义阶段边界和组件化方法就绪门禁。审计通过不构成研究方法或实验效果证据。
