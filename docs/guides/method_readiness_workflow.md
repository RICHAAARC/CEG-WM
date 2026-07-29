# Method Readiness Workflow

## Single Source Of Truth

唯一的项目方法就绪记录是：

```text
.codex/research_state/method_readiness.yaml
```

当前该文件已在 13 项职责、27 个 CPU/synthetic 方法行为节点和 revision-bound
独立语义复核完成后从模板唯一实例化。实际 stage/status 已由独立 revision 同步为
`runtime_verified / implemented`；readiness 本身未自动修改任何阶段。外层权威
`governance/templates/method_readiness.yaml` 和
`governance/policies/method_readiness_rules.yaml` 仍分别定义结构与规则，不是第二份
完成记录。

不得维护第二份 13 模块状态表、手工进度表或人类签字清单。人类报告只能从该 YAML
和审计结果生成，不能成为另一事实源。

## Creation Point

只有在以下条件全部满足后，才允许从模板实例化 readiness YAML：

1. 项目已在更早的独立 revision 合法进入 `method_construction_authorized`；
2. 13 项职责的真实实现和方法特异性测试已经完成；
3. 候选规格摘要、实现路径和测试节点已经冻结；
4. 执行者准备申请进入 `method_implemented`；
5. 有可审计 CEG-WM revision 可供独立语义复核绑定。

该创建点已满足并登记；创建本身不代表阶段已经通过。

## Required Bindings

当前记录对 policy 中每项唯一职责绑定：

- 唯一 component 和精确 responsibility；
- policy-fixed implementation path；
- 既有 candidate ID 集合；
- 唯一、真实、非 alias 的 implementation symbol；
- 方法特异性、数据依赖且非同构的 test node；
- 候选规格文件摘要；
- CEG-WM repository revision；
- revision-bound 独立语义复核结论。

同一 symbol 不得代理多个职责；carrier、embedder、detector、estimator、
reliability 和 joint decision 不得折叠。27 个行为节点的精确清单与组件绑定只以
`governance/policies/method_readiness_rules.yaml` 为准，本指南
不复制第二份清单。

## Review Lifecycle

1. 从 template 生成候选记录，填入真实路径、symbol、测试和摘要。
2. 运行默认测试、governance self-tests、readiness audit 和全部 harness。
3. 独立审计者核对每个绑定并执行方法特异性测试。
4. 独立审计绑定同一 repository revision 与 candidate digest。
5. 审计后候选、实现或登记测试路径发生变化时，批准失效并重新复核。
6. 全部门禁通过后，才可单独申请 `method_implemented` 阶段转换。

当前已完成步骤 1–6，并通过三任务独立复核及独立阶段迁移。
readiness 只证明“登记候选已由真实项目组件实现并通过规定门禁”。它不证明 runtime
真实可用、固定 FPR、攻击鲁棒性、论文有效性或完整 CEG-WM 科学成功。当前另有独立
真实 GPU qualification 支持 runtime 边界，但不改变 readiness 的职责。正式
detector 仍为 HF-only；LF/routing 尚未实验晋升，`full_ceg_wm_eligible=false`。

## Fail-Closed Conditions

- 13/27 实现或独立语义复核完成前提前创建 readiness YAML；
- 任一职责缺失、路径错置、symbol 复用、alias-only 或集中代理；
- 候选 ID、摘要或 implementation revision 缺失；
- 测试只验证 API 形状、常量、通用算术或重复同构行为；
- 独立语义复核缺失、拒绝或与当前 revision 不一致；
- 把 AST/audit pass、文档、目录或模板当作实现证据。

任一条件成立时不得申请 `method_implemented`，也不得以人类报告覆盖机器记录。
