# CEG-WM Research Construction Roadmap

`experiment_ready_infrastructure_closure` 的七项职责为：研究范围与阶段治理、方法架构
与证据边界、内容/几何/联合判定设计、算法与候选冻结、真实方法实现、runtime
qualification、实验协议与可追溯交付。后续参考验证统一称为
`hf_only_reference_validation`。

## Roadmap Authority

本文档定义从当前 `experiment_ready` 到论文全部数据、可重建产物和
受支持结论的构建路线。它是研究与工程准入顺序，不是完成状态报告。

文中的语义化名称是证据门，不是新的 `project_stage`。正式阶段仍以 `governance/policies/method_readiness_rules.yaml` 登记的顺序为准：

```text
research_defined
    ↓
method_construction_authorized
    ↓
method_implemented
    ↓
runtime_verified
    ↓
experiment_ready
    ↓
formal_evidence_available
```

任何证据门未通过时，必须停在当前阶段、保存失败事实并修订对应设计，不能跳过、放宽门禁或用后续实验掩盖失败。`method_construction_authorized` 是实施准入/在建阶段，不是完成结论：候选规格关闭并独立审计通过后，用户先授权建立可审计版本身份与阶段变更；阶段变更本身不得包含 `main/` 实现，方法实施只能在之后的独立变更中开始。

当前检查点：13 项职责、27 个 CPU/synthetic 方法行为节点、唯一 readiness 和真实
SD3.5 runtime qualification 已分别完成并审计；实际 stage/status 已由独立
revisions 同步为 `experiment_ready / implemented`。runtime 证据精确绑定 candidate
`8b2344756c4c247906ff0d4eab68e46a773e13f5` 和 qualification run
`20260729T110628Z`。正式 detector 仍为 HF-only，LF/routing 未实验晋升，
`full_ceg_wm_eligible=false`。实验准备基础设施已冻结协议与可追溯执行交付，但没有
`tau`、confirmation 结果、Calibration Locked、正式 evaluation、完整联合 FPR、
正式 records 或科学效果证据，也没有 LF/routing/组合/geometry 晋升。

## Paper Research Target

最终目标是形成一套可独立重建的 CEG-WM 论文证据，至少支持以下问题：

- CEG-WM HF direct score 是否能在本项目中保持正确密钥归属；
- LF 是否在 HF 易受损条件下提供独立且互补的密钥证据；
- 内容自适应路由是否在相同总能量下优于无路由；
- LF/HF 组合是否优于 HF-only，且不掩盖错误密钥失败；
- Q/K 几何链是否能盲估计 crop、scale、rotation 并可靠拒绝不可恢复样本；
- 条件几何恢复是否提高同一内容检测器的 TPR，同时保持同一阈值和 FPR；
- 完整方法在预登记攻击、baseline、图像质量和资源成本下是否具有论文价值；
- 完整联合检测器是否达到固定 FPR `0.001` 级别。

“达到 FPR=0.001 级别”必须同时满足：

- operating point 在 calibration 中预先冻结；
- evaluation 使用独立主负样本；
- 报告经验 FPR 和精确置信区间；
- 若声称 `FPR <= 0.001`，单侧置信上界也必须不超过 `0.001`；
- wrong-key null 与 unwatermarked-image null 分开报告；
- raw 与 rectified 路径共同计入完整联合检测器 FPR。

## Evidence Gate: Research Specification Closed

### Work

- 冻结研究目标、访问模型、攻击者能力和非目标；
- 冻结内容证据唯一阳性权；
- 冻结 HF direct score 形成的 content detector 原始入口；
- 冻结几何盲检测和 conditional recovery；
- 定义算法原语、方法机制、评估设计和本路线图；
- 明确 key schedule、LF、路由、组合和 Q/K 的有限候选与验证门；阈值数值仍由后续互斥 calibration 职责拟合。

### Validation

- research-definition audit 通过；
- 所有登记设计路径存在、内容实质且互不矛盾；
- 该门关闭时 `implementation_status` 为 `not_implemented`；
- 该门关闭 revision 的 `main/` 不包含实质方法实现；
- 默认测试、治理自测和全量 harness 通过。

### Pass Result

通过后只允许关闭方法候选规格，不自动获得实施授权，也不自动进入 `method_construction_authorized` 或 `method_implemented`。

## Evidence Gate: Method Candidate Specification Closed

### Work

- 登记候选生成 backbone、checkpoint、revision、scheduler、dtype 和图像尺寸；
- 冻结 `key_schedule_sha256_counter` 的 root-key/UTF-8、stable JSON、KDF/PRG、
  counter、normal table、职责域、wrong-key/public-noise 与 golden vectors；
- 在不要求 CEG-WM 已有 Git 的前提下，只读登记历史源真实 revision、逐文件摘要、许可证缺口、候选/偏离边界和逐参数映射，形成 provisional provenance；
- 把 CEG-WM `hf_sparse_tail` 候选冻结为 `sparse tail → direct L2 normalize → score-time centering`，并只在 provenance 中注明 historical DirectHF 来源；任何 template-time centering 变体使用新的 CEG-WM HF 候选身份且不继承旧证据；
- 冻结候选 HF 模板/单位方向、content embedder 的 mixing coefficients、
  combined pre-normalization geometry、`3/250` nominal 与 actual hard limit、
  nominal delta、runtime 物化位置/realized combined total norm/relative L2
  返回边界、binary32 最大非零可行 scale 选择和 HF direct score；
- 预登记 LF 模板/单位方向、embedder 使用、runtime 物化边界和 score 候选集合；
- 预登记内容路由的 observations、`A`、两 mask、identity/digests 和不读取观测的
  disabled uniform control；相同预算由 embedder/实验配对约束；
- 预登记 LF/HF empirical-CDF/normal-quantile 组合的有限候选族、candidate-selection
  provisional operating point、正式 content-threshold-fit 边界和晋升门；
- 预登记 Q/K 层、头、image-only empty-condition 条件、四通道关系、非正交 content
  subspace 投影、actual-dtype budget、变换支持域、W/V 采样、搜索平局和可靠性候选；
- 定义小型 synthetic/CPU 行为检查与真实 runtime 检查的边界。

### Validation

- 每个候选都有明确输入、输出、身份、失败语义和可证伪指标；
- registry 计数固定为 11 个 ID：10 个具名候选和 1 个 mandatory
  `routing_uniform_control`；不得把对照计入方法候选数；
- 不存在固定历史 LF/HF 权重；
- 不存在 reference image、embed record 或私有嵌入状态检测依赖；
- 组合候选不按攻击类型切换；
- 回正前后 detector 和 threshold 身份保持一致；
- 候选选择只使用 development 或 calibration 职责数据。
- 在 CEG-WM baseline revision 建立前，允许独立审计候选规格和历史源 provisional
  provenance；不得声称已迁移、已有 CEG-WM method revision 或完整 migration
  provenance。
- key/KDF/PRG、Q/K relation/objective、LF write/score、routing observations、
  backbone/runtime 均已有明确、可证伪且无隐式选择的候选规格。

### Pass Result

本门在无 CEG-WM Git 时即可提交独立审计。该流程已完成候选审计、基线版本身份和
construction admission；以下是此前进入 `method_construction_authorized` 所遵循的
两项分离授权：

1. 建立 CEG-WM 可审计版本身份，并把当前已审候选规格保存为 authorization base revision；
2. 按 `method_construction_admission.yaml` 绑定审计/用户授权引用，以一个不含任何 `main/` 变更的独立 revision 进入 `method_construction_authorized`，随后才在后续 revision 开始迁移/实现。

历史源 revision 或许可证无法确认时记录 fail-closed migration gap；它不阻止提出
CEG-WM 版本身份授权，但缺口关闭前不得实际复制历史代码。construction admission 的
research-definition
audit 必须从 authorization base revision
`e325c5efa3f35d0881e4d1d1743ab9d1ce87dbb9` 验证阶段转换。这一步只开放后续
实施，不表示方法已实现；实际实现和 readiness 已在更后的独立 revisions 完成。

## Evidence Gate: Method Implementation Ready

### Work

该门要求项目先在更早的独立变更中进入 `method_construction_authorized`，再于后续
变更实现：

- `main/shared/key_schedule.py` 的独占 root-key/KDF/PRG 责任和不可变结果类型；
- content router、LF/HF carrier 与独立 content embedder；
- 独立 LF detector、HF detector/direct score 与 content detector；
- Q/K 几何同步；
- transform estimator；
- 独立 geometry reliability gate；
- image rectifier；
- conditional recovery decision。

以上按唯一的 13 项正式职责和固定路径实施；当时 readiness 审核绑定的 10 个候选 ID
不是组件计数。此后设计新增的第 11 项 `lf_null_whitened_matched_score` 尚未实现，
不得在对应实现与独立语义审核完成前写入 readiness。同时建立的
`.codex/research_state/method_readiness.yaml` 逐组件连接已实现候选 ID、
架构规定路径、声明责任、具体且唯一的实现 symbol、方法特异性验收测试和实现完成后
的独立语义审计 revision。

### Validation

- 13 个必需组件都有非 placeholder 实现；
- unit tests 检查 stable serialization、counter/quantile golden、wrong/public
  derivation、数学性质、密钥域分离、确定性、dtype 和失败边界；
- functional tests 直接导入真实实现并使用非恒定断言；
- 相同通用函数不得冒充多个组件，集中式代理模块、常量/输入无关返回和重复同构测试必须被 readiness audit 拒绝；
- 验收节点必须覆盖 key root/domain、counter/quantile golden、wrong/public derivation、
  HF sparse support/模板归一顺序/单位 L2、HF score-time centering、LF carrier 与
  独立盲 score、routing mask partition/range 与 disabled uniform control、
  router masks 经 carrier directions 进入 content embedder、embedder 的冻结
  mixing coefficients/非正交交叉项/共同 nominal/actual hard limit/
  HF-only-LF-only-combined nominal delta/零方向、LF/HF/combined 独立可观测
  与冻结组合、真实 Q/K relation、synthetic transform、独立 reliability、
  rectification 和 same-detector joint decision；actual-dtype 的冻结算术、
  integrity、单调预算谓词、终止、最大可行 scale、plateau/subnormal/轻微超限与
  无非零可行写入先由 CPU property tests 覆盖，真实 SD3.5 物化仍必须留到真实
  runtime gate；不得由 CPU 节点或可加分支伪字段冒充 GPU evidence；
- AST 审计只构成结构/接线必要门；独立语义审计必须审阅实现 revision、候选规格摘要和真实测试，且审阅后受保护代码不得变化；
- `main.content_chain` 与 `main.geometry_chain` 无互相依赖；
- 方法代码不导入 runtime、experiments 或 governance；
- 默认测试和全量 audits 通过。

### Pass Result

只有全部组件、方法特异性验收节点和独立语义审计通过，才可以进入
`method_implemented`。缺少任一 13 项职责，或把 content embedder、LF detector、
geometry reliability 折回其他组件，都不得推进；机械 readiness pass
单独不证明非代理实现。

当前该实现/readiness 门已由 13 项职责、27 个非同构 CPU/synthetic 节点、候选摘要、
受保护 revision 和三任务独立语义复核闭合；独立阶段迁移已经完成。

## Evidence Gate: HF Candidate Identity Reproduced

### Work

- 建立本项目 model/runtime adapter；
- 从同一基础随机状态运行 clean 与 watermarked 配对生成；
- 对迁移候选逐参数核对模板、名义写入、actual-dtype hard-budget
  scale/replay/integrity/status、realized combined total norm/relative L2、
  最终图像重编码和评分；
- 使用预登记正确密钥与错误密钥 roster；
- 保存普通图像、最小必要数值记录和 provenance；
- 禁止把 inversion 设为主检测路径。

### Validation

- CPU golden vectors 证明 PRG、模板和 score 的确定性；
- 真实模型 smoke 证明 callback、调度器和 dtype 身份；
- 小型 development roster 只用于排错；
- 独立 holdout roster 复验正确密钥排序、margin、clean negative 和图像质量；
- 迁移偏差全部显式记录；
- 失败 Prompt 保留在分母。

### Pass Result

通过后冻结 CEG-WM HF direct score 作为 HF-only content detector baseline。未通过时停止 LF、几何和正式攻击扩展，先修复或重新定义 HF 候选。

## Evidence Gate: Content Branch Decision Closed

### LF-Only Work

- 比较多个预登记 LF 候选；
- 使用 LF-only 正确密钥、错误密钥和无水印负样本；
- 测量 identity、JPEG、blur、resize 和其他 HF 易受损条件；
- 报告图像质量、可见性、实际能量和计算成本。

### Routing Work

- 在相同 Prompt、seed、key 和总能量下比较 route-disabled 与 routed；
- 分别运行 HF-only、LF-only 和组合载体；
- 检查攻击前后路由可解释性和覆盖退化；
- 拒绝检测端需要私有嵌入路由的候选。

### Combination Work

- 只让已经通过 LF-only 门的候选进入组合；
- 在 candidate-selection manifest 的 selection partition 中冻结分支标准化和组合参数；
- 在该 manifest 预登记且未参与拟合的 confirmation partition 中评价组合 TPR、wrong-key attribution 和 HF-only 退化；
- formal evaluation 对已经冻结的方法同时保留 LF、HF 和组合分数，但不再承担候选选择；
- 不为不同攻击或回正图拟合不同组合。

### Validation

- LF 提供至少一个预登记攻击族的稳定增量；
- LF 的 key attribution 和 unwatermarked FPR 独立成立；
- routing 增益不是总能量差异造成；
- 组合在 candidate-selection confirmation partition 中满足晋升门；
- 组合完整检测器重新校准，不沿用 HF-only 阈值。

### Pass Result

- 组合通过：正式 `D_M` 晋升为冻结 LF/HF 组合检测器；
- 组合未通过：登记 `content_branch_research_question_closed_negative`，LF 与 routing 作为可报告的负结果，完整 CEG-WM 成功路径在此关闭；
- 若要继续 HF-only + geometry，必须建立重新命名、缩小主张且单独授权的 `reduced_scope_method_candidate`，不得沿用完整 CEG-WM 的方法成功或完成门；
- 两种结果都必须形成诚实研究结论，不允许无限调参，但只有组合通过才可继续申请完整 CEG-WM 方法成功。

## Evidence Gate: Geometry Chain Verified

### Synthetic Geometry Work

- 使用合成 Q/K 关系验证坐标约定和 estimator；
- 覆盖 identity、单一 rotation、scale、translation/crop 和组合变换；
- 注入多候选歧义、低覆盖、高残差和越界条件；
- 验证可靠性 fail closed。

### Real Q/K Work

- 在冻结模型中捕获真实 Q/K；
- 验证层、头、token 网格和关系摘要稳定；
- 比较无几何同步、正确密钥同步和错误密钥同步；
- 评价 Q/K 同步写入对内容质量和 HF direct score 的干扰。

### Rectification Work

- 对真实图像攻击执行盲估计；
- 按冻结逆变换回正；
- 报告 rotation error、scale error、translation/crop error、coverage、reliability 和 rectification quality；
- 使用同一个内容检测器测量回正前后变化；
- 使用 oracle transform 仅计算诊断上界。

### Validation

- identity 条件接近恒等；
- 支持域内变换误差满足预登记容忍度；
- 错误密钥和不可辨识条件可靠拒绝；
- 几何可靠性不直接产生阳性；
- 回正增益来自同一内容 detector；
- extreme crop 等不可恢复条件保持失败，不使用生成式补全。

### Pass Result

通过后冻结 Q/K observation、estimator、support domain、reliability 和 rectifier。未通过时不得通过扩大 rescue 区间或降低内容阈值补偿。

## Evidence Gate: Joint Detector Verified

### Work

- 冻结 `D_M`；
- 在 calibration 中冻结 `tau`；
- 在独立 calibration 职责数据中冻结 `tau_rescue` 和几何可靠性；
- 对 raw-only、geometry-always、conditional recovery 和 oracle upper bound 做消融；
- 完整运行 unwatermarked negatives、wrong keys、watermarked positives 和攻击样本；
- 保存每个样本的 raw 分数、触发、估计、可靠性、回正状态、rectified 分数和最终判定。

### Validation

- raw 与 rectified detector identity、key semantics 和 `tau` 完全一致；
- 远离阈值样本不触发几何；
- 几何失败保留原负判定；
- 可靠几何但 rectified 内容未达阈值时仍为负；
- conditional recovery 相对 raw-only 提供预登记增益；
- geometry-always 不被默认解释为更强方法；
- 完整联合检测器满足预设 FPR 预算。

### Pass Result

通过后方法机制冻结。此后任何内容 detector、组合、几何或阈值语义变化都需要新方法身份、重新 calibration 和新的 formal run。

## Evidence Gate: Runtime Verified

### Work

- 固定模型与依赖来源；
- 锁定 Python、框架、CUDA、dtype 和关键算子；
- 建立 CPU property tests、真实模型 smoke 和 GPU runtime qualification；
- 验证 deterministic seed policy、OOM/资源失败和 restart 语义；
- 验证普通图像输入输出与密钥不落盘；
- 建立最小 Colab 或服务器薄入口，但不在 Notebook 中定义方法。

### Validation

- 真实模型加载、生成、图像编码和 Q/K 捕获完成；
- 运行身份和代码 revision 可追溯；
- CPU 通过不冒充 GPU 通过；
- 失败运行产生显式失败记录；
- runtime 不拥有判定语义；
- 同一冻结输入能够按协议复现。

### Pass Result

满足正式 runtime 边界后进入 `runtime_verified`，随后才能冻结实验协议和高成本执行入口。

## Evidence Gate: Experiment Protocol Frozen

### Work

- 分离 development、calibration 和 evaluation；
- 冻结 Prompt、seed、key、图像、攻击和 baseline manifests；
- 冻结 calibration 子职责；
- 冻结完整 records schema、失败和排除策略；
- 冻结攻击矩阵、指标集合、compute budget 和 tuning budget；
- 登记外部 baseline 的不可变来源、许可证、配置和偏差；
- 为每个正式结果定义 artifact rebuild 路径。

### Validation

- calibration/evaluation 无样本、Prompt、seed、key 或近重复内容泄漏；
- 同一源样本派生攻击不跨 split；
- evaluation 不再调参；
- internal design validation 与 external comparison 分开；
- 项目方法和 baseline 共享公平的样本、攻击和指标条件；
- protocol preflight、schema tests、field registry、dependency audits 和失败 fixtures 通过。

### Pass Result

通过后进入 `experiment_ready`。只有这个阶段才允许执行正式 calibration、攻击矩阵和论文实验。

## Fixed-FPR Statistical Design

### Primary Null

固定 FPR 的主要单位是一个独立的：

```text
unwatermarked generated image + preregistered detection key
```

每个主负样本必须有独立样本身份。若同一图像配多个 key 或同一 key 配多个图像，这些扩展试验必须按 image/key 聚类处理，不能简单当作完全独立样本扩大 `n`。

wrong-key on watermarked image 是 attribution null，单独报告，不能与 primary null 混池。

### End-To-End Error

完整联合检测器的假阳性包括：

- raw 内容分数直接越过 `tau`；
- raw 未越阈值但进入 rescue，并在回正后越过同一 `tau`。

因此只校准 raw HF direct score FPR 不足以证明完整方法 FPR。formal calibration check 和 evaluation 必须运行完整联合路径。

### Error Budget

正式目标：

```text
alpha_total = 0.001
```

在查看 calibration/evaluation 结果前预登记：

```text
alpha_raw + alpha_rescue <= alpha_total
```

`alpha_raw` 与 `alpha_rescue` 的分配必须由预登记的 tail、触发率和统计 power 假设说明，不预设无证据的固定对半数字。最终裁决只看完整联合检测器是否满足 `alpha_total`，不允许用预算拆分掩盖总 FPR 超标。

若 LF/HF 组合替换 HF-only content detector，全部预算和阈值必须重新拟合。

### Calibration Separation

正式 calibration 必须分成互不重叠的 source-cluster manifests：

- LF/HF and routing selection；
- content-threshold fit；
- rescue-window fit；
- geometry-reliability fit；
- end-to-end calibration check。

候选选择只可读 candidate-selection manifest，不能读取阈值、rescue、geometry reliability 或 end-to-end check 数据；其余职责也不能反向选择候选。默认不跨职责 cross-fit；若未来确需 cross-fitting，必须在查看数据前登记 fold 级角色、聚类约束与独立 end-to-end check，且 evaluation 样本始终只能用于最终评估。

source cluster 由同一 Prompt、seed、生成图像 lineage 与注册 key family 定义。同一 source cluster 的全部攻击、回正、多 key 派生样本必须落入同一职责和 split；不能通过把派生行当独立样本扩大 `n`。

### Negative Sample Planning

每个职责单独登记样本量或规模确定规则：

- candidate selection：由最小相关增益、候选数、错误选择率与预登记 power 确定；
- content-threshold fit：由目标尾部概率、阈值估计容忍度与 raw 路径误差预算确定；
- rescue-window fit：由触发率、预登记增量 TPR/FPR 效应与 power 确定；
- geometry-reliability fit：由变换支持域、key/null 分层覆盖与可靠性校准精度确定；
- end-to-end calibration check 与 formal evaluation：由完整联合检测器的单侧置信上界、核心攻击数量与 simultaneous confidence 方案确定；
- 任何核心攻击条件若单独声称 `FPR <= 0.001`，必须有满足该条件置信要求的独立负样本量。

规模必须在访问对应职责数据前冻结。development/pilot 摘要可触发新的预登记并上调规模；不能看到该职责或 evaluation 结果后缩减规模、改变停止规则或把多个职责合并成一个虚假的总样本量。

### Confidence Requirement

对 `n` 个独立 primary negatives 观察到 `k` 个假阳性时，报告：

- empirical FPR `k / n`；
- 单侧 `95%` Clopper-Pearson 上界；
- 分母、失败、排除和聚类信息。

若 `k = 0`，要使单侧 `95%` 上界不超过 `0.001`，至少需要：

```text
n >= ceil(log(0.05) / log(0.999)) = 2995
```

这只是零假阳性时的数学最低值，不足以稳定刻画尾部，也不自动满足攻击分层、多重比较或聚类要求。正式规模由上述预登记计算决定。

若同时对 `A` 个核心攻击条件声称相同上界，必须使用预登记 simultaneous confidence 控制。零假阳性时的保守下限可按：

```text
n_per_condition >= ceil(log(0.05 / A) / log(0.999))
```

或使用预登记的等价 family-wise 方法。未达到样本量或上界要求时，只能报告观察值，不能声称达到 `FPR <= 0.001`。

## Evidence Gate: Calibration Locked

### Work

- 在 content-threshold fit 上冻结 `tau`；
- 在 candidate-selection manifest 上冻结通过晋升门的 LF/HF 与 routing；若未通过则关闭完整 CEG-WM 成功路径，不进入其 calibration；
- 在 rescue-window-fit 上冻结 `tau_rescue`；
- 在 geometry-reliability-fit 上冻结 geometry reliability；
- 在 calibration check 上运行完整联合检测器；
- 生成不可变 split、threshold 和 method manifests；
- 独立审核所有摘要、样本计数和失败状态。

### Validation

- 没有 evaluation 数据访问；
- 完整 detector 的 calibration FPR 满足预算；
- raw/rectified 使用同一 `tau`；
- 任何 detector identity 变化都会使 calibration 失效；
- candidate selection、threshold、rescue、geometry reliability 和 end-to-end check 有互斥 source-cluster manifests 与独立 provenance；
- calibration 失败时不允许进入 evaluation。

### Pass Result

通过后冻结 formal method package 与 calibration package，只允许按同一身份运行 evaluation。

## Evidence Gate: Formal Evaluation Complete

### Core Evaluation

- identity 下的 watermarked positives 和 primary negatives；
- correct-key 与 wrong-key attribution；
- fixed-FPR TPR；
- LF、HF 和 combined score distributions；
- image quality；
- latency、GPU memory、geometry trigger rate 和 failure rate。

### Attack Evaluation

至少覆盖：

- JPEG 与其他压缩；
- Gaussian blur；
- resize/down-up；
- noise；
- color、brightness 和 contrast；
- crop；
- scale；
- rotation；
- crop + scale；
- crop + rotation；
- scale + rotation；
- crop + scale + rotation；
- 几何与非几何组合；
- 预登记生成式攻击；
- 独立自适应攻击协议。

每个攻击样本都保留成功、方法失败、几何拒绝、runtime 失败或按预先规则排除状态。

### Generalization Evaluation

- 不同 Prompt 类别和复杂度；
- 多 seed；
- 多注册 key；
- 错误 key roster；
- 若论文主张包含，则覆盖不同尺寸、scheduler 或模型；否则明确限制；
- 不能把同一 Prompt/seed 的派生样本跨 calibration/evaluation。

### Baseline Comparison

- 登记并复现外部生成式图像水印 baseline；
- 使用相同 sample manifest、攻击、质量和计算预算；
- 分别报告本项目完整方法、HF-only、LF-only、geometry-disabled 和 baseline；
- baseline 失败必须区分源码、环境、资源和科学结果。

### Validation

- evaluation 全程不改方法、阈值或攻击；
- formal records 通过 schema 和 provenance 验证；
- primary FPR 与 simultaneous confidence 满足目标；
- 所有攻击和失败保留在分母；
- 统计分析脚本只读 frozen records；
- 独立复算关键表格和置信区间。

### Pass Result

通过后进入 `formal_evidence_available`。未通过的指标必须如实形成限制或负结果，不能回写方法参数后继续沿用同一 evaluation。

## Distinct Research Outcomes

以下终态不能共用完成门或论文措辞：

- `full_ceg_wm_evidence_available`：内容自适应路由、LF、HF、Q/K、回正和联合判定全部存在、通过各自机制门，并由完整联合检测器证据支持；
- `content_branch_research_question_closed_negative`：LF、routing 或组合未晋升，形成可发表的负结果，但完整 CEG-WM 方法成功未成立；
- `reduced_scope_method_evidence_available`：仅在重新命名、重新定义贡献范围、重新授权并独立校准/评估后，才可用于 HF-only + geometry 等缩减方法。

`formal_evidence_available` 只说明某个已登记研究身份拥有冻结记录；论文和 artifacts 必须同时标明上述 outcome identity。任何缩减方法不得被包装成完整 CEG-WM。

## Paper Result Inventory

若 outcome 为完整 CEG-WM，最终论文至少需要以下可重建结果；若为负结果或 reduced-scope 方法，清单必须按真实身份缩减并明确缺失的完整方法结论：

### Main Results

- 固定 FPR `0.001` 下 identity 与核心攻击 TPR；
- empirical FPR、单侧置信上界和负样本数；
- 与外部 baseline 的公平比较；
- correct-key 与 wrong-key attribution。

### Content Mechanism

- HF-only、LF-only、route-disabled、routed 和 combined；
- LF/HF 分数分布和组合贡献；
- 内容路由覆盖、mixing coefficients、方向内积/组合归一因子，以及
  `content_embedder` 核验的 nominal/limit、materialization scale、attempt/integrity/
  budget status、diagnostic utilization 与 realized combined total norm/relative
  L2；
- LF 失败或未晋升时的完整负结果。

### Geometry Mechanism

- rotation、scale、translation/crop 参数误差；
- reliability、coverage、inlier、residual 和拒绝率；
- raw、rectified 和 oracle upper bound；
- wrong-key geometry 与 unreliable controls。

### Joint Decision

- raw-only、geometry-always 和 conditional recovery；
- rescue trigger rate、rescue success、false rescue 和增量 FPR；
- 同检测器同阈值身份核对；
- 几何可靠但内容仍失败的样本统计。

### Quality And Cost

- PSNR、SSIM、LPIPS 或预登记感知指标；
- 生成和检测延迟；
- GPU memory；
- 几何触发带来的平均和尾部成本；
- runtime 与资源失败率。

### Robustness And Generalization

- 非几何攻击；
- 几何与组合攻击；
- 生成式攻击；
- 自适应攻击；
- Prompt、seed 和 key 分层结果；
- 预登记范围外的限制。

### Failure Analysis

- false positives；
- false negatives；
- key attribution failures；
- geometry ambiguity；
- extreme crop；
- rectification degradation；
- runtime/resource failures。

## Evidence Gate: Paper Evidence Closed

### Work

- 从 frozen records 和 manifests 重建所有 tables、figures 和 reports；
- 每个 supported claim 绑定 artifact 和 source records；
- 独立复算 FPR、TPR、置信区间和主要消融；
- 生成 artifact inventory、checksums 和 rebuild commands；
- 准备 minimal method、experiment execution 和 paper artifact rebuild packages；
- 对论文文字执行 claim audit。

### Validation

- 删除临时输出后仍能从 frozen inputs 重建；
- Notebook 不保存唯一方法或统计逻辑；
- artifact builder 不读取开发日志或 harness 报告作为科学数据；
- 表格、图和正文数字一致；
- unsupported、insufficient-evidence 和 failed 结论明确区分；
- release package 不包含原始密钥、私有路径、缓存或历史 outputs；
- 独立环境按 README 完成重建。

### Pass Result

只有此门通过后，项目才完成“论文所有数据结果”的目标。`formal_evidence_available` 表示已有真实证据，不自动表示论文主张、artifact 和 release 已全部闭合。

## Governance Freeze And Extension Rule

方法 readiness、真实 runtime qualification、实验协议与可追溯执行交付以及独立
`experiment_ready` 阶段迁移完成后，通用治理平面继续冻结。下一步真实运行、
candidate-selection、confirmation、calibration、Calibration Locked 和正式
evaluation 必须按冻结协议及后续授权执行；本段不授权这些工作。后续主线仍须按
既定门序进入 experiment 和 evidence 工作。
除非这些真实工作暴露可复现的具体缺口，否则不得新增通用 policy、
skill、schema 或 harness；存在缺口时优先对现有规则做最小、可测试的增量修订。
治理文件数量、机械 audit 数量或文档篇幅都不能替代研究实现和证据推进。

## Stop And Return Rules

以下情况必须停止并回到对应证据门：

- CEG-WM HF 候选无法在本项目复现；
- LF 不具备独立 key attribution；
- 组合掩盖错误密钥失败或降低 HF-only；
- Q/K 不可稳定观测；
- geometry reliability 不能 fail closed；
- 回正需要私有嵌入状态；
- raw/rectified detector 或阈值身份不同；
- end-to-end FPR 超过预算；
- calibration/evaluation 泄漏；
- formal run 修改了方法或协议；
- 样本量不足以支持 `0.001` 级别结论；
- artifacts 无法从 frozen records 重建。

返回意味着形成新的设计或协议 revision、重新 calibration，并使用全新的 evaluation；不得覆盖旧失败记录。
