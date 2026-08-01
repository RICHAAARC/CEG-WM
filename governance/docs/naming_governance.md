# Naming Governance Rules

命名审计覆盖每个受检查文件/目录 basename、Python 标识符、注释、docstring、正式
Python 字符串身份、JSON/YAML/TOML 键与正式 identity/name/label/path 值、Notebook
code/Markdown、Markdown、SVG、Drawio 和登记字段。正式名称使用语义明确的
`snake_case`，表达真实职责、机制、数据含义或版本角色；任一表面通过 snake_case
并不免除 weak 与 ordinal identity 检查。

不得用单字母加数字的内部工作身份代替职责名称；该规则不是固定字母清单，覆盖
`B1`、`D-2`、`M_3`、`candidate_x1_gate` 等紧凑、下划线、连字符及语义前后缀
形式。职责词加数字也属于序号身份：`phase`、`step`、`stage`、`batch`、`tier`、
`level`、`group`、`track`、`route`、`gate`、`case`、`option`、`variant`、
`module`、`component`、`method`、`model`、`baseline`、`run`、`experiment`、
`trial` 与数字形成的紧凑或分隔变体都必须拒绝。真实 `test_*` 节点与其他 Python
函数同样受审计，不存在 tests 目录通用豁免。仅非正式的测试 fixture 局部变量可以
描述 fixture 维度或 synthetic 对象；它们不得成为 persisted/public/registered identity、
测试节点或兼容 alias。正式绑定、路径、函数/class 节点、注释和 docstring 不共享该例外。

正式身份、持久化/公开身份、项目业务路径、artifact 与 evidence identity 不得使用
`tmp`、`temp`、`misc`、`other`、`todo`、`tbd`、`dummy`、`fake`、`mock`、
`proxy`、`new`、`old`、`latest`、`best`、`final`、`backup`、`copy`、`foo`、
`bar` 作为临时或不明语义。测试中明确非正式的 synthetic fixture 可以使用
`fake_gpu`、`mock_backend`，但这些值一旦绑定正式 identity 字段仍必须失败；不得以
blanket tests exemption 放行。所有无明确版本职责的机械数字后缀按类别禁止，而非
只检查固定名词；`detector2`、`metric_3`、`config_2`、`result4`、`router2`、
`artifact_3`、`candidate4`、`protocol_7`、`method_v2` 都必须失败。明确 dtype
literal 属于科学职责，不是身份编号。业务生产路径中的普通变量、attribute、参数与
keyword 也执行该检查，不因尚未成为函数/class 或手写 formal token 而豁免。业务
代码注释与 docstring 中以代码 token 形式出现的机械身份同样失败；普通数值、tensor
shape 和明确统计/版本职责不因此进入身份审计。

`docs/reference/field_registry.md` 中 category 为 `method_identity` 或
`runtime_identity` 的全部字段值都执行 weak 与 ordinal 两条检查；未登记但显然承担
identity/name/label/path 职责的键继续使用语义 token fallback。字段登记表不得为了
迁就审计而删改类别。外层治理保存当前 identity 字段类别的非降级契约；登记字段缺失、
类别降级或 identity 字段重复登记必须在 naming 与 field 两个独立审计中失败。登记表
缺失、为空或相关行不可可靠解析时，naming audit 不得退回手写 regex 后继续通过。
JSON、YAML 或 TOML 解析失败必须产生明确 `config_unreadable`。这些约束只属于可拆卸
governance，不进入项目 schema 或运行对象。两条身份检查产生可区分 reason，不得互相替代。

窄例外仅包括 `relative_l2`、`F32`、`RGB8`、`P95`、`x86_64`、`L4`、
`SHA-256`、`SHA256`、实际使用的外部模型 literal `SD3.5`，以及
`realized_total_l2` 这类明确表达 L2 norm 职责的语义标识符和立即定义或用反引号标出的局部数学记号 `C_0`、
`C_1(w)`、`S_0`。`schema_version`、`model_revision` 与 `upstream_commit` 等明确角色
可以使用；外部资产通过 adapter 或登记表保留原名。不得扩张为通用 allowlist，
也不得保留两个可充当权威的 alias。

字段实例及等级以 `docs/reference/field_registry.md` 为准。
