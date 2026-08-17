# CEG-WM Project Contract

## Project Position

CEG-WM 是双链生成式图像水印研究项目。内容链负责水印证据，几何链负责条件同步、变换估计和图像回正，联合判定负责在严格门控下调用几何恢复。`.agents/`、`.codex/` 和 `governance/` 是可拆卸的构建期护栏，不属于方法或实验运行时。

## Current Stage

- `project_stage`: `experiment_ready`
- `target_construction_phase`: `experiment_ready`
- `method_transition_authorization_base_revision`:
  `15e138ecf99e75084c6862b1f291b1747986123b`
- `runtime_transition_authorization_base_revision`:
  `82e1fefe2ccae2dc4be8205cc39a62b2052137ea`
- `runtime_transition_authorization_reference`:
  `source-thread=019fa382-130d-7a81-aef8-41c692a87676;authorization-text=批准创建 runtime_verified 阶段迁移 revision;authorized-after=qualification-passed-20260729T110628Z`
- `experiment_transition_authorization_base_revision`:
  `bc44dfae57f2471524d2b2aabbfb51228b04bc31`
- `experiment_transition_authorization_reference`:
  `source-thread=019fa387-9113-76d1-bff9-8a09b41746b5;authorization-text=批准创建 experiment_ready 阶段迁移 revision;authorization-base=bc44dfae57f2471524d2b2aabbfb51228b04bc31`
- 项目已依据关闭的候选规格、独立审计批准、用户明确授权和可审计基线 revision
  合法进入构建准入阶段。当前 readiness 在固定 13 项真实职责上绑定 12 个唯一
  候选身份、17 个方法特异性 CPU/synthetic 行为节点和唯一
  `method_readiness.yaml`，并通过 revision-bound 独立语义审计。旧的 11 候选/
  28 行为节点 readiness 快照继续由 revision
  `0258ccb2100bfe8b58d1a12079876841192528b3` 保留为历史事实，不是当前权威。
- 用户已基于 experiment_ready_infrastructure_closure 全部完成、冻结实验协议和可追溯执行入口明确授权本次独立
  阶段迁移。本 revision 只把实际阶段同步为 `experiment_ready / implemented`，
  不修改方法、runtime、Notebook、登记测试、候选规格或 readiness 语义。
- 已独立核验的 runtime 边界精确绑定 candidate
  `8b2344756c4c247906ff0d4eab68e46a773e13f5`、execution package SHA-256
  `8290abeed79931eb7208ac9ca280f1ea401f4725abfead35f12617a0ef54dd38`、
  `qualification / passed` run `20260729T110628Z` 及 result ZIP SHA-256
  `d9b7d91d41cc963098c077268445ad80e9994c809227ca2f68615a37ac93ac37`。
  该结果证明冻结 SD3.5 runtime 的 callback 18、actual dtype/integrity/budget、
  VAE、两层真实 Q/K、registered-key 重复确定性和 negative-key identity control
  可执行；Optional Replay 在没有具体缺口时不强制。
- 当前 `experiment_ready` 只表示 experiment_ready_infrastructure_closure 已完成且冻结协议与可追溯执行入口可用；
  不授权 calibration、hf_only_reference_validation 晋升、GPU 高成本运行、正式攻击矩阵或论文实验，也不构成
  科学效果结论。
- root-key/KDF/PRG、Q/K relation/objective、LF/HF write/score、routing observations、
  backbone/runtime、搜索与回正已经在
  `docs/design/candidate_specifications.md` 中关闭为有限、可实施、可证伪的候选；
  registry 现为 19 个具名候选加 1 个强制 routing 禁用对照，共 20 个 ID。
  语义—纹理软路由五候选采用 InSPyReNet soft `M`、Sobel/P95 `T`、soft-routed
  LF/HF write、两条盲分支分数和固定 max statistic；五候选实现 producer 为
  `02fe5dcc2b74482c9eb1e0b192b4a2ce79e0d9eb`，状态为
  `implemented_not_scientifically_validated`，并由登记的 independent exact audit
  批准其实现语义。hard salient-object local-LF 四候选状态为
  `superseded_without_scientific_adjudication`。状态权威见
  `docs/project_state/method_route_registry.md`。
- 当前正式 detector 仍为 HF-only；旧 routing/combination 已形成 producer-bound
  development negative；语义—纹理软路由五候选虽已实现，但 soft max 仍仅为
  diagnostic、未实验晋升且没有 formal threshold，
  `full_ceg_wm_eligible=false`。`negative_identity` 只证明 runtime/key identity
  control 与 registered identity 分离，不是 wrong-key FPR、attribution 效果或科学
  证据。readiness、runtime qualification 和阶段转换也不证明固定 FPR、鲁棒性、
  几何恢复效果或完整 CEG-WM 科学成功。

## Method Authority

### Content Chain

1. `main.shared` 的 `key_schedule_sha256_counter` 是 LF、HF、Q/K、wrong-key 与
   public-noise 的唯一共享密钥/随机协议候选；原始 key 不持久化。
2. 内容链包含 `content_router`、`lf_carrier`、`hf_carrier`、独立
   `content_embedder`、`lf_detector`、`hf_detector` 和正式联合入口
   `content_detector`；组合写入、LF 盲分数和统计组合不得互相代行。
3. 当前冻结的 HF 主检测候选使用 CEG-WM 自有身份；其稀疏尾部算法具有 historical DirectHF 来源的 provisional provenance。当前实现按 CEG-WM 候选规格独立完成，不声明历史源码已迁移、复用权已关闭或继承历史效果证据。
4. `routing_stqr` 与 `content_combination_calibrated` 的既有执行路线已经形成
   producer-bound development 负证据，保留作历史复现而不再是当前内容候选。
   方法设计使用 InSPyReNet soft semantic probability `M` 与 deterministic Sobel/P95
   texture `T` 构造逐图正软路由；它不增加第 14 项职责，仍由既有
   `content_router`、carrier、`content_embedder`、分支 detector 与
   `content_detector` 分工。
5. `content_router` 只输出 `M/T`、
   `m_hf=(1+M*T)/(2+M)`、`m_lf=(1+M*(1-T))/(2+M)` 及 identity/digests；
   它不决定标量混合权重或输出能量预算。carrier 只输出模板和 routed unit
   direction。runtime 只物化 embedder 的
   delta 并返回 actual-dtype 张量与 realized combined total norm/relative L2，
   预算合格与否仍由 embedder 判定；不得把 mixing coefficients 解释为可加分支
   能量，也不得声称 runtime 可观测 actual branch energy。
6. 当前内容候选把 `content_relative_l2_nominal` 与
   `content_relative_l2_limit` 都冻结为 `3/250`。对 callback 18 的 actual-dtype
   baseline `z0`，runtime 仅按 embedder 请求的 binary32 scale `s` 物化
   `z_s=cast_actual(fp32(z0)+s*delta_content_nominal)` 并返回
   `delta_actual_s=fp32(z_s)-fp32(z0)`、完整性与 realized 测量；embedder 独占
   hard-budget 直接比较、重试、最大非零可行 scale 选择和最终 fail-closed。
   realized ratio/utilization 只作诊断，不是 gate，不得设置 `tau_actual_budget`、
   经验 tolerance 或 actual 强度下限。
7. 上述 actual hard limit 只约束 LF/HF/routing 最终合成并物化后的 combined
   content delta；nominal directions/components 只用于公式重放，不构成 actual
   branch decomposition。geometry delta 与现有 geometry/total budget 独立。
8. 不得从历史项目继承固定 `0.7/0.3`、`0.5/0.5` 或其他未经 calibration/evaluation 验证的组合规则；语义—纹理候选唯一写入方向为
   `normalize(normalize(m_hf*T_hf)+normalize(m_lf*T_lf))`，不是权重网格；其
   `max(z_hf_soft,z_lf_soft)` 仅为已实现、未科学验证且未晋升的 diagnostic。
   当前正式/default/joint `D_M` 仍为 HF-only 和既有阈值。未来晋升必须完成独立分支
   calibration、max threshold fit、固定 FPR/科学确认与显式 promotion，并让原图与
   回正图共同使用一个新的 detector/config identity 和新阈值；不得继承旧 W/CDF、
   `tau` 或 HF-only threshold。
9. 错误密钥、分支消融和组件分数必须可独立观测，组合分数不得掩盖密钥归属失败。

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
10. `models/` 是本地、非权威、不审计的模型资产/缓存根；checkpoint 和
    Windows `Zone.Identifier` 等下载附属元数据必须保持 Git 忽略，不得支撑
    方法、readiness 或科学结论。
11. model/repository/name/revision、checkpoint blob SHA/size 和环境版本/设备只能作为
    runtime 选择 locator 或观测元数据；不得进入 KDF、方法配置/result 强身份或相等性门。
    行为改变的 pipeline/scheduler、callback/VAE/QK、dtype/尺寸/steps、依赖/API/CUDA/
    资源 capability 与 InSPyReNet public API/preprocess/output/strict state_dict 仍 fail closed。

## Stage Governance

1. 阶段名称必须登记在 `governance/policies/method_readiness_rules.yaml`。
2. `research_defined` 及以后阶段必须提供 `.codex/research_state/research_definition.yaml`，连接具体设计文档和冻结方法不变量。
3. `method_construction_authorized` 是唯一允许开始实质 `main/` 实施的在建阶段。进入该阶段必须同时满足：候选规格已关闭并独立审计通过、用户明确授权、已有用户授权建立的可审计 repository revision、按模板登记 construction admission、阶段变更本身不包含 `main/` 实现；research-definition audit 必须从 admission 绑定的 base revision 验证该独立转换。
4. `method_construction_authorized` 本身只表示允许实施，不自动表示组件完成；当前组件完成事实由后续独立 revisions、唯一 readiness 和独立语义复核另行记录。
5. `method_implemented` 及以后阶段必须提供 `.codex/research_state/method_readiness.yaml`，逐项连接唯一的 13 个正式职责组件、固定架构路径、候选 ID、具体实现 symbol、责任和方法特异性行为测试。候选 registry 的 20 个 ID 与 13 项职责不得混淆；当前 readiness 的 12 个唯一候选身份和 17 个行为节点只绑定已审核实现，不得被解释为 runtime qualification、机制晋升、calibration、固定 FPR、GPU 或科学效果证据。
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
2. 本次独立 revision 只同步 `experiment_ready` 阶段，保留 `implemented` 状态；
   experiment_ready_infrastructure_closure 已完成，冻结协议与可追溯执行入口可用。本次迁移不授权 calibration、
   hf_only_reference_validation 晋升、GPU 高成本运行、正式攻击矩阵或论文实验。
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

## Required Completion Profiles

所有任务先运行最小受影响测试，再且只再选择一个完成档位：

| profile | scope | completion gates |
| --- | --- | --- |
| `governance` | 仅修改外层治理实现/测试/skill，或不改变研究语义的普通指南、索引与说明。 | governance pytest + harness |
| `method` | 仅修改研究代码、研究测试或非治理运行配置，不改变阶段、候选规格或治理规则。 | project pytest + harness |
| `full` | 同时跨研究层与治理层，或修改阶段、`.codex/research_state/`、登记设计、候选/readiness 规则、`AGENTS.md`、本合同、`pyproject.toml`、`governance/pytest.ini` 或验证档位本身。 | project pytest + governance pytest + harness |

权威设计虽然是文档，仍属于 `full`；普通文档也不得仅凭扩展名自动归为
`governance`。范围含混时使用 `full`。统一入口：

```bash
conda run -n CEG-WM python governance/tools/run_validation_profile.py governance
conda run -n CEG-WM python governance/tools/run_validation_profile.py method
conda run -n CEG-WM python governance/tools/run_validation_profile.py full
```

`governance` 档位不收集项目方法测试；`method` 档位不运行治理 pytest；三个档位都运行
完整 harness。治理合同测试会验证研究代码在拆包与移除治理层后的可运行性，因此可能
导入 `main` 并需要 PyTorch；三个正式档位统一使用登记 Conda 环境。缺少 `torch`
的 `.venv` 只能运行明确不导入研究代码的定向轻量检查，不能完成任何 profile。
