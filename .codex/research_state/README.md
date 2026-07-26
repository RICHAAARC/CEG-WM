# Research Construction State

此目录只保存构建期阶段门禁读取的外层元数据，不属于研究配置或运行输入。

- `research_definition.yaml` 连接当前十份权威设计和九个冻结方法不变量。
- `research_defined` 只能关闭候选规格；候选规格独立审计、用户实施授权和可审计 repository revision 齐备后，须先以不含方法实现的独立变更进入 `method_construction_authorized`。
- 获得授权时，按 `governance/templates/method_construction_admission.yaml` 登记授权引用和候选规格 revision；该文件当前不存在是正确状态，不能预先填写或虚构。
- 实质方法工作只能在后续变更中开始；进入 `method_implemented` 前，按
  `governance/templates/method_readiness.yaml` 创建 `method_readiness.yaml`，逐项连接
  唯一的 13 个正式职责组件、固定架构路径、候选 ID、责任、唯一实现 symbol、
  方法特异性非同构行为测试和 revision-bound 独立语义复核。10 个候选 ID 与
  13 项职责不是同一计数。

删除 `.codex/` 后，研究项目必须仍可运行；这里的元数据不能成为方法输入。
