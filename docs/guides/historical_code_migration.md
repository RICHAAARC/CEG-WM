# Historical Code Migration Guide

## Current Boundary

本指南只规定未来另行获授权后的迁移流程。当前处于
`method_implemented / implemented`；本次独立阶段迁移不包含历史源码迁移，
仍只允许保留来源事实和候选映射计划，禁止复制、改写、迁入或执行历史源码。

四个历史项目均为非权威只读来源。来源事实见
[historical source registry](../reference/migration/historical_source_registry.yaml)；
13 项构建前映射见
[component migration plan](../reference/migration/component_migration_plan.yaml)；
CEG-WM 算法身份仍只来自
[candidate specifications](../design/candidate_specifications.md)。

## Required Flow

```text
历史来源登记
  → revision/archive digest 与复用权核验
  → CEG-WM candidate ID
  → 13 项目标职责
  → policy-fixed 目标路径
  → 直接复用 / 独立重写判定
  → 历史偏离修正
  → 方法特异性测试
  → 独立语义复核
  → readiness 登记
```

任何一步失败都停止后续步骤。没有许可证文件、许可证不覆盖拟议用途或用户复用权未
关闭时，直接复制 fail closed。用户可以另行授权依据权威公式进行不复制历史源码的
独立重写，但该授权不能从本指南推断。

## Step Requirements

### 1. Source Identity And Rights

- Git 来源绑定真实 immutable revision；工作树脏状态不能混入该 revision 的身份。
- 无 Git 来源在准入前生成只读 archive/file manifest，按排序相对路径和文件 bytes
  计算 SHA-256，并记录包含/排除规则；没有 digest 时不得用于迁移。
- 登记发现的 LICENSE/COPYING/NOTICE、适用范围和复用授权引用。
- `license_status` 或 `reuse_authorization_status` 为 unresolved 时禁止直接复用。

### 2. Candidate And Responsibility Mapping

每个拟迁移文件必须先绑定一个既有 candidate ID，再绑定
[13 项职责表](../design/candidate_specifications.md#authority-and-status) 中唯一的
component、responsibility 和 planned path。不得新增候选别名、折叠职责或让一个
历史集中模块绕过计划路径。

组件级迁移记录至少需要：

- `source_id`、不可变 revision 或 archive digest、source relative path 和 file SHA-256；
- license/复用权状态及授权引用；
- candidate ID、目标 component、responsibility 和 policy-fixed target path；
- `reuse_mode`（仅可为经批准的 direct reuse 或 independent rewrite）；
- 历史偏离及修正说明；
- 方法特异性测试节点、CEG-WM implementation revision 和独立语义复核引用；
- 明确的 migration status。

这些是未来逐组件 migration manifest 的最低信息，不是当前 schema 或空模板。本轮
只创建 component migration plan，不创建 13 份空 manifest。

### 3. Reuse Decision

- direct reuse 需要来源身份、逐文件 digest、许可证/授权和候选等价性全部闭合。
- independent rewrite 只能依据 CEG-WM 权威规格编写，不能通过改变量名掩盖源码复制。
- 历史参数只能作为已登记的候选值；必须在 CEG-WM 中重新测试，不能继承成功状态。
- 任一算法偏离必须先修订候选规格并重新审计，不能在迁移实现中自行选择。

### 4. Tests, Review And Readiness

- 测试必须调用 policy-fixed 真实组件 symbol，并验证候选特异性、数据依赖和失败语义。
- 通用算术、重复同构测试、集中代理和 AST pass 不能证明迁移完成。
- 独立审计必须绑定候选摘要、实现 revision、目标路径和测试节点。
- 只有实际迁移后，才为对应组件创建逐组件 migration manifest；只有审计后受保护
  路径未变化，才可按
  [method readiness workflow](method_readiness_workflow.md) 登记 readiness。

## Forbidden Imports And Claims

禁止迁入或继承：

- 固定 `0.70/0.30`、`0.50/0.50` 或其他历史 LF/HF 融合；
- reference-based 几何、原始参考图注册或 oracle attack 参数；
- 私有 embed state、latent 统计或嵌入端 Q/K cache 的检测依赖；
- inversion/oracle 检测主线；
- payload、attestation 或事件证明主线；
- 历史配置、阈值、结果、完成状态、固定 FPR 结论或论文证据；
- 历史项目关于候选成功、失败或优越性的结论。

历史代码、配置和结果永远不能自行成为 CEG-WM 实现或证据。任何未来迁移仍须保持
内容证据唯一阳性、几何只做条件恢复，以及回正前后同 detector、key semantics、
preprocess、config identity 和 threshold。

## Current Stop

CEG-WM 已建立可审计 revision，并已依据冻结候选独立完成当前方法实现；这不表示
历史源码已迁移。四个来源的复用权仍未关闭，两个 archive 来源也没有冻结 snapshot
digest，因此所有直接历史源码迁移继续 fail closed。plan 不能替代 future
per-component migration manifest、当前 readiness 或独立迁移授权。
