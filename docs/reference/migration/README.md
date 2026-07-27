# Historical Migration Reference

本目录只保存历史来源的稳定、只读登记，不保存迁移实现、进度表、源代码副本或完成
声明。

- [historical_source_registry.yaml](historical_source_registry.yaml)：历史来源准入事实，
  包括四个非权威来源的路径、版本/摘要身份、复用权状态和允许/禁止用途。
- [component_migration_plan.yaml](component_migration_plan.yaml)：历史迁移计划快照；
  所有条目仍为 `not_started`，不表示当前实现来自历史源码或已经迁移。
- 迁移操作边界见
  [historical_code_migration.md](../../guides/historical_code_migration.md)。
- 候选和 13 项目标职责的唯一权威映射见
  [candidate_specifications.md](../../design/candidate_specifications.md)。

四个表面严格分离：

1. source registry 是历史来源准入事实；
2. component migration plan 是构建前 13 项计划；
3. future per-component migration manifest 是实际迁移后才创建的 provenance；
4. 当前唯一 method readiness 是实现完成门禁，不是 migration provenance。

四者不得互相替代。registry 中的 `provisional`、`read_only` 或 `unresolved` 不是
migration evidence；plan 也不是构建授权、实际 migration manifest 或 readiness。
未来逐组件 manifest 必须绑定来源 digest、候选、目标职责、目标路径、复用模式、
测试、CEG-WM revision 和独立语义复核；本轮不创建空 manifest 或模板。
