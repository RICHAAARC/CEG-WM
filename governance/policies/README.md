# Machine-Readable Policies

此目录保存供 harness 读取的 JSON-compatible YAML 规则：

- `project_roots.yaml`：顶级目录登记与审计范围；
- `dependency_rules.yaml`：跨层依赖与 records 唯一写入权；
- `project_skills.yaml`：项目级 skills 登记；
- `notebook_rules.yaml`：Notebook 路径、大小和输出规则；
- `method_readiness_rules.yaml`：研究定义阶段、13 项双链职责组件、固定实现路径、
  readiness-bound 旧实现的11个候选身份、方法特异性验收节点和独立语义复核要求。
  设计 registry 的15个 ID（14个具名候选加1个 mandatory control）、readiness
  绑定的11个旧实现候选与13项职责是三种不同计数；新增显著目标局部 LF 四候选为
  `design_candidate_implementation_authorized` / `implementation_admission=YES`，只授权
  本地独立 revisions 实施；四者尚未实现、未获 runtime qualification、实验执行准入或
  科学晋升，policy YAML 和 readiness 本轮均未重绑。

current 内容候选固定 max statistic，不存在 `a/w/function` 或组合参数 selection；
calibration 只承担分支 primary-null/CDF 标准化、max `tau` 与其他独立职责。
historical `a/w/function` 只按原 producer replay，不能由 policy 说明恢复为 current。

policy 只定义可执行约束，不保存项目参数或实验结果。

当前依赖审计消费 `layers`、`forbidden_dependency`、`delivery_code_roots` 和
`record_writer_layers`。实验 protocol、methods、attacks 与 metrics 的显式文件写入
会被拒绝；内部正式 records 只允许由 `experiments.runners` 写入，并已有负向测试。
