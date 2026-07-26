# Governance Templates

此目录只保存外层检查使用的元数据模板，不进入研究运行配置或交付包。

- `research_definition.yaml` 用于登记 CEG-WM 权威设计路径和冻结方法不变量。
- `method_construction_admission.yaml` 只在候选规格独立审计通过且用户授权后使用，绑定用户授权引用和授权前的规范 revision；research-definition audit 会从该 revision 检查必须存在一个不含 `main/` 变更的独立阶段转换。
- `method_readiness.yaml` 用于连接 13 个必需职责组件的固定架构路径、候选 ID、唯一
  实现 symbol、方法特异性默认 pytest 节点，以及覆盖同一候选/实现/test revision
  的独立语义复核。content embedder、LF/HF detector、content detector、transform
  estimator 和 geometry reliability 均是分离责任。AST 审计只证明必要的结构接线，
  不能单独证明非代理实现。

模板值只是构建期入口，不能进入研究运行依赖或替代真实设计与实现。
