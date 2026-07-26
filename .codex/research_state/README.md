# Research Construction State

此目录只保存构建期阶段门禁读取的外层元数据，不属于研究配置或运行输入。

- `research_definition.yaml` 连接当前十份权威设计和九个冻结方法不变量。
- `method_construction_admission.yaml` 已按模板登记候选规格关闭、独立审计批准、
  用户授权引用和完整 authorization base revision；它只证明合法进入
  `method_construction_authorized`，不证明实现完成。
- 当前为 `method_construction_authorized / not_implemented`；阶段转换本身不含
  `main/` 变更，13 项职责均尚未实现。用户要求本轮停在转换后，实质方法工作只能
  在本 revision 独立审计通过后的后续单独授权变更中开始。
- 进入 `method_implemented` 前，按
  `governance/templates/method_readiness.yaml` 创建 `method_readiness.yaml`，逐项连接
  唯一的 13 个正式职责组件、固定架构路径、候选 ID、责任、唯一实现 symbol、
  方法特异性非同构行为测试和 revision-bound 独立语义复核。10 个候选 ID 与
  13 项职责不是同一计数。
- 当前 `method_readiness.yaml` 不存在是正确状态。

删除 `.codex/` 后，研究项目必须仍可运行；这里的元数据不能成为方法输入。
