# Documentation Index

本目录按文档职责组织，不按开发时间或临时阶段平铺文件。

## 研究文档边界

本目录只保存研究设计、运行指引、决策、实验协议参考与证据语义。外层构建护栏不在这里维护，因此移除外层目录不会留下研究文档对其的运行依赖。

## 文档分类

| path | responsibility | current_contents |
| --- | --- | --- |
| [design/](design/README.md) | 只保存 CEG-WM 方法定义、算法原语、双链机制、评估设计、验证门和纯方法图。 | 十份登记设计文档；不保存实施或证据状态。 |
| [project_state/](project_state/README.md) | 保存阶段、采用路线、已认证负结果、待验证、资源阻断和禁止回流规则。 | 方法路线状态台账；不得修改算法公式。 |
| [guides/](guides/README.md) | 项目推进、历史迁移、readiness、Colab 和 artifact rebuild 操作。 | 薄操作指引及一张论文证据生产 guide 图。 |
| [reference/](reference/README.md) | 字段、测试、baseline、迁移来源/计划、拆包 profile 和 artifact evidence 参考。 | 稳定登记与边界参考。 |
| [decisions/](decisions/README.md) | 改变架构权威的语义化决策记录。 | 当前没有独立 decision record。 |

设计文档和 Notebook 不能单独支持 research claim；论文证据必须追溯到 records、manifests 和可重建 artifacts。
