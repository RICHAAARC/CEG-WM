# Governed Experiment Runners

此目录是方法、攻击、指标和协议的唯一组合层，也是 governed records 的唯一写入层。

- `internal.py` 只消费精确的冻结内部 protocol、split manifest、case input manifest、
  `CegWmExperimentAdapter`、attack registry、metric registry 和公开方法对象；它调用
  真实 attack 与 joint-decision API，不复制方法、攻击或指标算法。artifact 当前
  像素与 attack specification 当前参数由 attacks 公共入口在 affine/grid 前重校验。
  正式 content/geometry callable 必须显式实现
  `formal_runner_semantic_declaration()`，返回稳定 canonical JSON 配置；runner 将
  role、callable 类型和该声明摘要绑定到冻结 expectation，并在 attack 前和正式
  write 前重新计算。任一原位语义漂移直接失败，不转写为 execution/resource record。
  当前执行仍拒绝
  `held_out_evaluation`。
- `formal_operations.py` 提供 package-contained runner 使用的正式公共图像 callable：
  content 路径调用真实 `CegWmExperimentAdapter.detect_hf` 与
  `detect_content`，geometry 路径调用真实 runtime Q/K observation 与
  一个无 runtime binding 的 method adapter 的 `synchronize_qk_observation`、
  `geometric_transform_estimator`。runner 通过普通公共方法调用这些能力，并校验
  方法/runtime 配置、输入与结果语义及执行范围；不锚定 Python descriptor、bound
  method 或 module alias，也不复制 detector、Q/K 或 estimator 算法。CPU synthetic
  wiring 的 raw-positive case 不执行未初始化的模型 backend。
- `record_writer.py` 是内部正式 records 的唯一 writer。构造时必须同时接收冻结
  case input manifest；每次 load 或物化前，它都通过
  `validate_run_case_record_collection` 复验完整 collection，逐 record 核对
  确定性 ID/sequence/attempt/parent、逐 unit provenance、routing、key/control、
  detector/config/preprocess、threshold/tau、content/geometry callable 配置、
  几何操作和 experiments 层可靠性配置摘要，以及
  code/protocol/candidate/config/input/resource provenance，并检查所有序列化字段已在
  包内 `protocol/internal_record_registry.py` 的可执行 registry 中允许进入 records；
  运行时不读取 `docs/`。writer 在构造时深拷贝 protocol、split manifest 与 bindings
  以及 case input manifest 的 canonical primitive 快照，load/append/write 均复验锚点。写入使用 case 文件锁、临时文件 `fsync`、
  `os.replace` 和目录 `fsync`；既有 canonical records 必须先通过同样重放校验。
- resume 不重复已完成 unit；资源失败按冻结 attempt 上限形成显式 parent lineage，
  `scientific_failure` 只来自显式 `ConditionalRecoveryResult` 科学失败语义，意外
  Python/method 错误记录为终止 `execution_failure`。replay 从真实 record 文件重算 schema、判定与
  metric case 一致性，日志不进入 records。

`FrozenRecordBindings` 是 runner 构造前已经验证的外部输入。本层只保证 execution
context、writer 和逐 record provenance 在内部全生命周期一致；正式 code/package/
bootstrap revision 权威由后续 experiment_execution_delivery 外部信任链绑定，不由 governed_internal_runner 自我声明。

包含外部 baseline 的高成本运行仍必须先接收
`experiments.protocol.comparison.PreflightApproval`，重新计算当前
`ComparisonProtocol` digest 并确认一致。本次内部 runner 不实现 baseline、
Notebook、GPU 调度、calibration、held-out evaluation 或 stage 推进。

CPU/synthetic runner tests 只证明编排、身份绑定、原子写入、resume/replay 和失败
语义，不构成候选晋升、runtime/GPU 或科学效果证据。

`contrastive_lf_branch_attribution.py` 为 Stage-A 预分配 exact 128-item null-fit 与
4960-item selection collections；每次最多一次 attempt，首个 failed slot 后完整保留
unstarted tail。Production factory 明确装配 `Sd35PipelineBackend ->
Sd35RuntimeAdapter -> CegWmExperimentAdapter -> Stage-A operations`；synthetic fake
只存在于 tests，不是 package 默认路径。
