# Machine-Readable Policies

此目录保存供 harness 读取的 JSON-compatible YAML 规则：

- `project_roots.yaml`：顶级目录登记与审计范围；
- `dependency_rules.yaml`：跨层依赖与 records 唯一写入权；
- `project_skills.yaml`：项目级 skills 登记；
- `notebook_rules.yaml`：Notebook 路径、大小和输出规则；
- `method_readiness_rules.yaml`：研究定义阶段、13 项双链职责组件、固定实现路径、
  readiness-bound 当前12个唯一候选身份、17个方法特异性验收节点和独立语义复核要求。
  设计 registry 的20个 ID（19个具名候选加1个 mandatory control）、readiness
  绑定的12个候选身份与13项职责是三种不同计数。语义—纹理软路由五候选已由
  producer `02fe5dcc2b74482c9eb1e0b192b4a2ce79e0d9eb` 实现并独立复核，但只处于
  `implemented_not_scientifically_validated`；旧的11候选/28节点 readiness 快照在
  `0258ccb2100bfe8b58d1a12079876841192528b3` 保留为历史事实。

current 内容候选固定 max statistic，不存在 `a/w/function` 或组合参数 selection；
calibration 只承担分支 primary-null/CDF 标准化、max `tau` 与其他独立职责。
historical `a/w/function` 只按原 producer replay，不能由 policy 说明恢复为 current。

policy 只定义可执行约束，不保存项目参数或实验结果。
`models/` 以 `generated_asset / audited=false` 登记，只是本地非权威模型
资产/缓存根；其 checkpoint 和下载附属元数据由 `.gitignore` 排除。

当前依赖审计消费 `layers`、`forbidden_dependency`、`delivery_code_roots` 和
`record_writer_layers`。实验 protocol、methods、attacks 与 metrics 的显式文件写入
会被拒绝；内部正式 records 只允许由 `experiments.runners` 写入，并已有负向测试。
