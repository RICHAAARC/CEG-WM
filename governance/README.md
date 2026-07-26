# Governance Control Plane

此目录是随具体研究项目复制的外层控制平面。研究与运行代码不得导入它。

| path | responsibility |
| --- | --- |
| `contracts/` | 人可读的稳定架构契约。 |
| `policies/` | JSON-compatible YAML 机器规则。 |
| `harness/` | 读取 policy 并输出结构化审计结果。 |
| `tests/` | 控制平面自身的轻量测试。 |
| `docs/` | 只解释外层护栏，不进入研究文档交付。 |
| `templates/` | 审计专用元数据模板。 |
| `tools/` | 可拆卸的项目复制与交付候选拆包工具。 |

整个目录可从研究交付中移除；研究代码、项目测试、脚本和 Notebook 可执行代码不得导入它。治理通过不能替代方法实现、真实运行或实验记录。当前阶段与允许工作仍以 `.codex/project_contract.md` 为最高项目约束。
