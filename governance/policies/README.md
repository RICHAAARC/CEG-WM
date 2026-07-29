# Machine-Readable Policies

此目录保存供 harness 读取的 JSON-compatible YAML 规则：

- `project_roots.yaml`：顶级目录登记与审计范围；
- `dependency_rules.yaml`：跨层依赖与 records 唯一写入权；
- `project_skills.yaml`：项目级 skills 登记；
- `notebook_rules.yaml`：Notebook 路径、大小和输出规则；
- `method_readiness_rules.yaml`：研究定义阶段、13 项双链职责组件、固定实现路径、
  现有 10 个候选 ID 的组件绑定、方法特异性验收节点和独立语义复核要求。组件数
  与候选 registry 计数不得混淆。

policy 只定义可执行约束，不保存项目参数或实验结果。

当前依赖审计消费 `layers`、`forbidden_dependency`、`delivery_code_roots` 和
`record_writer_layers`。实验 protocol、methods、attacks 与 metrics 的显式文件写入
会被拒绝；内部正式 records 只允许由 `experiments.runners` 写入，并已有负向测试。
