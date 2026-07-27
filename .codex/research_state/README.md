# Research Construction State

此目录只保存构建期阶段门禁读取的外层元数据，不属于研究配置或运行输入。

- `research_definition.yaml` 连接当前十份权威设计和九个冻结方法不变量。
- `method_construction_admission.yaml` 已按模板登记候选规格关闭、独立审计批准、
  用户授权引用和完整 authorization base revision；它只证明合法进入
  `method_construction_authorized`，不证明实现完成。
- 当前为 `method_construction_authorized / not_implemented`；阶段转换本身不含
  `main/` 变更；随后独立 revisions 已完成 13 项职责和 27 个 CPU/synthetic
  方法行为节点，但阶段/status 尚未迁移。
- 唯一 `method_readiness.yaml` 已从模板实例化，逐项连接固定路径、候选 ID、
  责任、唯一实现 symbol、27 个非同构行为节点、候选摘要和 revision-bound
  独立语义复核。10 个候选 ID 与 13 项职责不是同一计数。
- readiness 只记录方法构建闭合，不是阶段文件，也不是 runtime、GPU、正式 FPR
  或科学证据。当前正式 detector 仍为 HF-only，LF/routing 未实验晋升，
  `full_ceg_wm_eligible=false`；等待独立阶段迁移。

删除 `.codex/` 后，研究项目必须仍可运行；这里的元数据不能成为方法输入。
