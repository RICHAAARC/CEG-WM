# Naming Governance Rules

命名审计覆盖受检查路径中的文件、目录、Python 标识符、注释、docstring、配置键和 Notebook code cell。正式名称使用语义明确的 `snake_case`，表达真实职责、机制、数据含义或版本角色。

不得用 `v1`、`v1v2`、`p1`、数字阶段名、`proxy`、`new`、`old`、`best` 或 `final` 承担正式语义。`schema_version`、`model_revision` 与 `upstream_commit` 等明确角色可以使用；外部资产通过 adapter 或登记表保留原名。

字段实例及等级以 `docs/reference/field_registry.md` 为准。
