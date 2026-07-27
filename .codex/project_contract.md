# CEG-WM Project Contract

## Project Position

CEG-WM 是双链生成式图像水印研究项目。内容链负责水印证据，几何链负责条件同步、变换估计和图像回正，联合判定负责在严格门控下调用几何恢复。`.agents/`、`.codex/` 和 `governance/` 是可拆卸的构建期护栏，不属于方法或实验运行时。

## Current Stage

- `project_stage`: `method_implemented`
- `target_construction_phase`: `method_implemented`
- `method_transition_authorization_base_revision`:
  `15e138ecf99e75084c6862b1f291b1747986123b`
- 项目已依据关闭的候选规格、独立审计批准、用户明确授权和可审计基线 revision
  合法进入构建准入阶段。随后独立 revisions 已完成固定 13 项真实职责、27 个
  方法特异性 CPU/synthetic 行为节点和唯一 `method_readiness.yaml`，并通过
  revision-bound 独立语义审计。
- 用户已基于上述完整 revision 明确授权本次独立阶段迁移。本 revision 只把实际
  阶段和 research-definition 状态同步为 `method_implemented / implemented`，
  不修改方法实现、登记测试、候选规格或 readiness 语义。
- 当前不得接入真实模型，不得运行 GPU、正式攻击矩阵或论文实验。
- root-key/KDF/PRG、Q/K relation/objective、LF write/score、routing observations、
  backbone/runtime、搜索与回正已经在
  `docs/design/candidate_specifications.md` 中关闭为有限、可实施、可证伪的候选；
  registry 是 9 个具名候选加 1 个强制 routing 禁用对照。CPU/synthetic 实现不
  等于实验晋升；仍开放的是 LF/routing/组合晋升结果和 calibration 数值。
- 当前正式 detector 仍为 HF-only，LF/routing 尚未实验晋升，
  `full_ceg_wm_eligible=false`。readiness、文档、阶段转换或 admission 都不是
  runtime、GPU、固定 FPR、鲁棒性或科学效果证据。

## Method Authority

### Content Chain

1. `main.shared` 的 `key_schedule_sha256_counter` 是 LF、HF、Q/K、wrong-key 与
   public-noise 的唯一共享密钥/随机协议候选；原始 key 不持久化。
2. 内容链包含 `content_router`、`lf_carrier`、`hf_carrier`、独立
   `content_embedder`、`lf_detector`、`hf_detector` 和正式联合入口
   `content_detector`；组合写入、LF 盲分数和统计组合不得互相代行。
3. 当前冻结的 HF 主检测候选使用 CEG-WM 自有身份；其稀疏尾部算法具有 historical DirectHF 来源的 provisional provenance。当前实现按 CEG-WM 候选规格独立完成，不声明历史源码已迁移、复用权已关闭或继承历史效果证据。
4. LF、路由和 LF/HF 组合已有明确有限候选；`content_embedder` 独占共同总预算
   与冻结 `a` 的组合写入，`lf_detector` 独占 `s_lf`。它们仍属于本项目设计验证
   问题，不是已晋升方法事实。
5. `content_router` 只输出生成时 observations、`A`、`mask_lf`、`mask_hf`、
   route identity/digests 和 disabled uniform control；它不决定 `a` 或输出能量
   预算。carrier 只输出模板和 masked unit direction。runtime 只物化 embedder 的
   delta 并返回 actual-dtype 张量与 realized combined total norm/relative L2，
   预算合格与否仍由 embedder 判定；不得把 mixing coefficients 解释为可加分支
   能量，也不得声称 runtime 可观测 actual branch energy。
6. 不得从历史项目继承固定 `0.7/0.3`、`0.5/0.5` 或其他未经 calibration/evaluation 验证的组合规则。
7. 错误密钥、分支消融和组件分数必须可独立观测，组合分数不得掩盖密钥归属失败。

### Geometry Chain

1. 几何链目标是基于 Q/K 观测进行同步，估计 crop、scale 和 rotation，并完成图像回正。
2. 正式检测只能消费待检图像、检测密钥和冻结的公共方法/模型资产。
3. 不得消费原始参考图、嵌入端 record、嵌入 latent 统计或其他私有嵌入状态。
4. `geometric_transform_estimator` 输出原始估计指标，独立
   `geometry_reliability` 对错误 key、覆盖、唯一性、gap、残差、边界和非有限量
   执行冻结合取门；几何证据不得进入内容分数或直接产生水印阳性。

### Joint Decision

1. 当前冻结门控以 CEG-WM HF direct score 形成的 content detector 原图分数和校准阈值为入口。
2. 原图达到阈值时由内容证据判阳性；远离阈值的负样本不得启动几何恢复。
3. 只有近阈值负样本且几何可靠时，才允许执行回正。
4. 回正后必须使用同一检测器身份、同一密钥语义和同一阈值重判。
5. 不得增加独立的宽松 rescue classifier，也不得让几何可靠性直接补足内容分数。
6. 若后续经治理设计验证将 LF/HF 组合升级为正式内容检测器，则原图与回正图必须调用同一组合检测器和同一阈值。

## Historical Source Boundary

`SLM-WM`、`SLM-WM-FlowHF`、`CEG-WM-OLD-main` 和 `CEG-O-master` 是非权威历史来源。`candidate_specifications.md` 已只读登记前两个项目的 provisional revision、文件摘要、可迁移候选与历史偏离；这不表示代码已迁入。任何历史代码进入本项目之前仍必须关闭复用权、建立 CEG-WM revision、按候选 ID 迁移并重新测试；历史固定融合、reference-based 几何、私有嵌入状态依赖、payload 或 attestation 主线不得隐式迁入。

## Governance Boundary

1. `.agents/skills/` 保存项目工作流。
2. `governance/contracts/` 保存人可读架构契约。
3. `governance/policies/` 保存机器可读根目录、依赖和阶段策略。
4. `governance/harness/` 保存外层治理审计。
5. `governance/tests/` 保存控制平面自测。
6. 项目研究代码、交付代码和 Notebook 可执行代码不得导入 `governance/`。
7. 删除 `.agents/`、`.codex/` 与 `governance/` 后，研究代码、项目测试、实验协议和 artifact rebuild 必须仍可运行。
8. 审计专用元数据不得进入方法 API、runtime 对象、研究配置或实验 records。

## Project Directory Rules

1. `main/` 是核心方法边界；计划内部分为共享类型、内容链、几何链和联合判定，目录存在不代表实现完成。
2. 内容链与几何链不得互相导入；联合判定可以消费两链公开结果，但不得重写两链算法。
3. `runtime/` 只通过 `main/` 公开接口提供模型执行和 Q/K 观测，不拥有方法判定语义。
4. `experiments/protocol/` 保存内部设计验证、外部比较和 records 契约。
5. `experiments/methods/`、`attacks/` 和 `metrics/` 保持正交。
6. `experiments/runners/` 是唯一 governed records 写入层。
7. `paper_artifacts/` 只从冻结 records 和 manifests 重建产物。
8. Notebook 只能作为薄编排入口。
9. 外部 baseline 必须登记不可变来源、许可证、配置和偏差，并在高成本运行前通过 comparison preflight。

## Stage Governance

1. 阶段名称必须登记在 `governance/policies/method_readiness_rules.yaml`。
2. `research_defined` 及以后阶段必须提供 `.codex/research_state/research_definition.yaml`，连接具体设计文档和冻结方法不变量。
3. `method_construction_authorized` 是唯一允许开始实质 `main/` 实施的在建阶段。进入该阶段必须同时满足：候选规格已关闭并独立审计通过、用户明确授权、已有用户授权建立的可审计 repository revision、按模板登记 construction admission、阶段变更本身不包含 `main/` 实现；research-definition audit 必须从 admission 绑定的 base revision 验证该独立转换。
4. `method_construction_authorized` 本身只表示允许实施，不自动表示组件完成；当前组件完成事实由后续独立 revisions、唯一 readiness 和独立语义复核另行记录。
5. `method_implemented` 及以后阶段必须提供 `.codex/research_state/method_readiness.yaml`，逐项连接唯一的 13 个正式职责组件、固定架构路径、候选 ID、具体实现 symbol、责任和方法特异性行为测试。候选 registry 的 10 个 ID 与 13 项职责不得混淆。
6. 只有全部必需组件、候选特异性非同构行为测试以及绑定同一候选摘要和受保护 revision 的独立语义复核通过后，才允许从 `method_construction_authorized` 进入 `method_implemented`。
7. readiness AST 审计只检查必要的结构和接线；空目录、placeholder、单一通用函数、集中式代理模块、重复同构测试、机械 audit pass 或 readiness 元数据不能替代项目实现与独立语义复核。
8. `runtime_verified` 必须有真实 runtime 边界证据；`experiment_ready` 必须有冻结协议与可追溯执行入口；`formal_evidence_available` 必须有真实 records。
9. 阶段表示证据可用性，不自动表示完整 CEG-WM 科学成功。LF、内容自适应路由或其他完整方法必需机制得到负结果时，可以形成诚实负证据，但不得声明完整 CEG-WM 已验证。

## Repository Revision Admission

1. CEG-WM 已由用户授权建立可审计版本身份；本次 construction admission 精确绑定
   根基线 `e325c5efa3f35d0881e4d1d1743ab9d1ce87dbb9`。不得虚构、缩写或替换该授权
   基线。
2. 在 CEG-WM Git 建立前，允许完成并独立审计候选规格与历史源 provisional
   provenance；历史仓库真实 revision/文件摘要可登记，但不得声称已经迁移。
3. 候选规格审计、版本身份和 construction authorization 已经分别完成；此前进入
   `method_construction_authorized` 的阶段转换不授权在同一 revision 中实施、
   迁移历史源码或继承历史证据。
4. 在迁移历史代码或进入 `method_construction_authorized` 前，必须由用户明确授权建立 CEG-WM 版本身份；历史源 revision 或许可证缺口不得阻止提出该授权请求，但实际复制代码在缺口关闭前 fail closed。
5. 版本身份只提供 provenance，不替代候选规格、实现测试、runtime 验证或科学证据。

## Governance Freeze And Extension Rule

1. 本轮阻断关闭后，治理平面进入冻结状态。
2. 本次独立 revision 只同步 `method_implemented` 阶段和 `implemented` 状态。
   下一步 runtime qualification 和进入 `runtime_verified` 必须另行获得明确授权；
   本次迁移不授权 runtime、GPU 或实验。
3. 除非真实方法、runtime、实验或证据工作暴露一个具体且可复现的缺口，不得新增通用 policy、skill、schema、harness 或治理目录。
4. 发现具体缺口时，优先最小修改现有规则与测试；不得以治理文件数量、机械审计通过或元数据规模替代研究推进。

## Artifact And Claim Governance

1. records 是实验结果事实来源。
2. tables、figures 和 reports 必须可由 records 与 manifests 重建。
3. supported claims 必须绑定 governed artifacts。
4. Placeholder、中间状态、日志、Notebook 输出和 harness 报告不得支撑 supported claims。
5. 正式输出不得写入已提交的 `outputs/`。
6. records 必须显式保存成功、失败或排除状态及必要 provenance。

## Naming, Fields And Tests

1. 正式名称使用语义明确的 `snake_case`。
2. Placeholder 字段以 `_placeholder` 结尾；random trace 字段以 `_random` 或 `_digest_random` 结尾。
3. 跨边界或持久化字段必须登记到 `docs/reference/field_registry.md`。
4. 默认项目测试只运行轻量 unit、constraint 和 quick 测试。
5. integration、smoke、slow、formal 和真实模型/GPU 测试默认排除。
6. `governance/tests/` 使用独立 `governance/pytest.ini`。

## Task Scope Completion

1. 治理任务必须修改并验证治理层，不得借机写入研究实现。
2. 方法、runtime 或实验任务在阶段允许时必须产生目标业务层实质实现；只改 docs、policy、skills 或 tests 不构成方法完成。
3. 当前阶段不允许的工作必须停在权威设计和阶段门禁，不得以 placeholder 实现绕过。

## Required Completion Commands

完整三门使用登记的 `CEG-WM` CPU Conda 环境：

```bash
conda run -n CEG-WM python -m pytest -q -s
conda run -n CEG-WM python -m pytest -q -s -c governance/pytest.ini
conda run -n CEG-WM python governance/harness/run_all_audits.py
```

`.venv` 只用于不依赖 PyTorch 的轻量治理检查；缺少 `torch` 时不得用它运行会收集
默认方法测试的根 pytest 命令。
