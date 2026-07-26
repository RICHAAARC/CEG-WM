# Machine-Readable Policies

此目录保存供 harness 读取的 JSON-compatible YAML 规则：

- `project_roots.yaml`：顶级目录登记与审计范围；
- `dependency_rules.yaml`：跨层依赖与 records 写入权；
- `project_skills.yaml`：项目级 skills 登记；
- `notebook_rules.yaml`：Notebook 路径、大小和输出规则；
- `method_readiness_rules.yaml`：研究定义阶段、13 项双链职责组件、固定实现路径、
  现有 10 个候选 ID 的组件绑定、方法特异性验收节点和独立语义复核要求。组件数
  与候选 registry 计数不得混淆。

policy 只定义可执行约束，不保存项目参数或实验结果。

当前依赖审计已经消费 `layers`、`forbidden_dependency` 和 `delivery_code_roots`。`record_writer_layers` 声明首次出现真实 runner 时必须落实的写入职责；在 runner 尚不存在时，它不表示 records 写入路径已经实现或经过验证。首次加入 runner 时必须同步增加写入职责检查和负向测试。
