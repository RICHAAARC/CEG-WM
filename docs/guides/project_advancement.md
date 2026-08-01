# Project Advancement Guide

## Scope And Authority

本指南是从当前 `experiment_ready` 状态向后推进的薄操作索引，不定义新阶段，也不替代
外层权威 `.codex/project_contract.md`、
[构建路线图](../design/research_construction_roadmap.md) 或
`governance/policies/method_readiness_rules.yaml`。发生冲突时，以
上述权威文件和登记设计为准。完成本指南、图或审计不构成构建授权、方法实现、
runtime 验证或论文证据。

当前状态固定为：

- `project_stage: experiment_ready`；
- `implementation_status: implemented`；
- authorization base revision 为
  `e325c5efa3f35d0881e4d1d1743ab9d1ce87dbb9`；
- method transition base revision 为
  `15e138ecf99e75084c6862b1f291b1747986123b`；
- construction admission、13 项职责、27 个 CPU/synthetic 行为节点、唯一 readiness
  和 revision-bound 独立语义复核均已登记；
- 正式 detector 仍为 HF-only，LF/routing 未实验晋升，
  `full_ceg_wm_eligible=false`；
- 真实 SD3.5 runtime qualification 与独立阶段迁移已经完成，证据绑定 candidate
  `8b2344756c4c247906ff0d4eab68e46a773e13f5` 和 run `20260729T110628Z`；
- 实验协议与可追溯执行交付基础设施已经闭环，但尚无 `tau`、confirmation 结果、
  Calibration Locked、正式 evaluation 或正式实验 records；
- 未完成 LF/routing/组合/geometry 晋升、完整联合 FPR、攻击鲁棒性或科学效果。

## Advancement Map

| current → requested stage | allowed work | required inputs | required output | admission and stop rule |
| --- | --- | --- | --- | --- |
| `research_defined → method_construction_authorized` | 关闭候选规格审计；核验历史来源与复用权；准备版本身份和独立阶段变更。 | 十份登记设计；候选规格独立 `approve`；用户明确授权；用户授权建立的 CEG-WM revision。 | 从模板创建的 construction admission；不含 `main/` 实现的独立阶段 revision。 | 任一输入缺失即停止；不得把本文档授权解释为版本、阶段或实现授权。 |
| `method_construction_authorized → method_implemented` | 当前实现/readiness 已闭合；只准备独立阶段迁移。 | 已批准 admission；固定候选摘要；受保护实现/测试路径；已审 readiness。 | 全部真实组件、27 个非同构行为测试、唯一 readiness YAML 和 revision-bound 独立语义复核。 | 阶段迁移前再次核验受保护路径；缺组件、候选漂移或复核失效即停止在在建阶段。 |
| `method_implemented → runtime_verified` | 接入冻结真实模型、callback、VAE、Q/K、device/dtype 边界并验证。 | 已批准方法 revision；冻结 runtime candidate 与依赖。 | 真实 runtime identity、determinism、actual-dtype combined delta、Q/K observation 和失败记录。 | CPU fixture、mock 或 dry run 不能替代真实 runtime；资源或身份失败保持 fail closed。 |
| `runtime_verified → experiment_ready` | 实现并冻结内部设计验证、外部 comparison、互斥 calibration、runner 和 records 协议。 | runtime 证据；预登记样本/split/攻击/指标；baseline 来源与许可。 | preflight 通过的冻结协议、唯一 governed runner、可追溯配置和空白 evaluation admission。 | 数据泄漏、不公平预算、baseline 权限未闭合或 rescue/FPR 口径不完整即停止。 |
| `experiment_ready → formal_evidence_available` | 运行正式矩阵并冻结全部成功、失败、排除 records 与 manifests。 | 未被 evaluation 使用过的冻结方法、协议、数据和 revision。 | governed records、provenance、frozen manifests；可重建 artifacts 的事实输入。 | 运行中改方法/协议、失败丢分母或 FPR 样本量不足时不得宣称正式证据。 |

论文 artifacts 和 release package 的闭合发生在已有正式证据之后，详细门序见
[构建路线图](../design/research_construction_roadmap.md)；它们不新增语义阶段。

## Execution And Independent Audit

每个阶段转换必须分离：

1. 执行者准备该阶段所需的实现、记录或 admission，并提供精确 revision 和验证结果。
2. 独立审计者只读核验权威输入、失败分母、候选/配置摘要和受保护路径。
3. 审计结论只能是有证据的 `approve` 或 `request_changes`。
4. 审计后受保护文件变化会使批准失效；必须重新审计。
5. 用户授权负责版本身份、阶段转换、迁移/实现和高成本运行等不同权限，不得合并推断。

同一轮不得同时完成 `method_construction_authorized` 阶段转换和实质 `main/` 实现。

## Failure Semantics

- 缺少 revision、许可证/复用权、候选映射或独立批准：准入失败，不得绕过。
- 方法组件或测试失败：保留真实失败，不能以文档、metadata 或 AST pass 替代。
- runtime/GPU/资源失败：记为 runtime 或 resource failure，不得解释成科学通过。
- LF、路由或完整方法候选未晋升：可形成诚实负结果，但不得包装成完整 CEG-WM 成功。
- 几何失败、不可靠或远负样本：保持原内容负判定；几何不得直接产生阳性。
- calibration/evaluation 泄漏或 formal run 漂移：冻结结果失效，使用新 revision 和全新 evaluation。

## Operator Checklist

每轮推进前：

1. 读取外层权威 `.codex/project_contract.md` 和
   `.codex/research_state/research_definition.yaml`。
2. 核对阶段名与所需 evidence 已登记于
   `governance/policies/method_readiness_rules.yaml`。
3. 核对候选、13 项职责和目标路径见
   [candidate specifications](../design/candidate_specifications.md)。
4. 历史来源工作遵循
   [historical code migration guide](historical_code_migration.md)。
5. readiness 工作遵循
   [method readiness workflow](method_readiness_workflow.md)。
6. 论文 evidence 链遵循
   [artifact evidence semantics](../reference/artifact_evidence.md)。

阶段推进和阶段迁移属于 `full` 档位：

```bash
conda run -n CEG-WM python governance/tools/run_validation_profile.py full
```

该命令只验证当前可执行门禁，不会自动授权阶段推进。`.venv` 缺少 PyTorch 时
只能运行轻量治理检查，不得用于会收集默认方法测试的根 pytest 命令。
