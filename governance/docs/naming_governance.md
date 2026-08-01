# Naming Governance Rules

命名审计覆盖每个受检查文件/目录 basename、Python 标识符、注释、docstring、正式
Python 字符串身份、JSON/YAML/TOML 键与正式 identity/name/label/path 值、Notebook
code/Markdown、Markdown、SVG、Drawio 和登记字段。正式名称使用语义明确的
`snake_case`，表达真实职责、机制、数据含义或版本角色；任一表面通过 snake_case
并不免除 weak 与 ordinal identity 检查。

不得用 `v1`、`v1v2`、`p1`、数字阶段名、`proxy`、`new`、`old`、`best` 或 `final`
承担正式语义。不得以 `A1`、`A-2`、`A3a`、`A3b`、`C0`、`C1-P/M/E`、`c1_*`、
`R1/R2`、`S1/S2`、`P1/P2`、`P_1`、`P-2`、`a3b_metric`、`Runtime Batch N`、
`Batch N`、`stage_N` 及其下划线、连字符、紧凑变体、前置或后续语义 token
代替语义身份。下划线可以作为 identity token 的前置边界，不是全局前缀屏障；
Batch/Stage 分支仍要求完整 token 边界，不把带明确 fixture 后缀的维度名称截断为
工作包身份。唯一上下文例外是物理位于
任一名为 `tests` 的目录下、AST 节点本身为 FunctionDef/AsyncFunctionDef 且以 `test_` 开头的测试
函数名；同文件 basename、函数体标识符、字符串和配置仍受审计。正式字符串值同时接受 weak 与 ordinal 两条独立审计，
并产生可区分 reason；不得把任一规则的通过当成另一规则的替代。

窄例外仅包括 `relative_l2`、`F32`、`RGB8`、`P95`、`x86_64`、`L4`、
`SHA-256`、`SHA256`，以及立即定义或用反引号标出的局部数学记号 `C_0`、
`C_1(w)`、`S_0`。`schema_version`、`model_revision` 与 `upstream_commit` 等明确角色
可以使用；外部资产通过 adapter 或登记表保留原名。不得扩张为通用 allowlist，
也不得保留两个可充当权威的 alias。

字段实例及等级以 `docs/reference/field_registry.md` 为准。
