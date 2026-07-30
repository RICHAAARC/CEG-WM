# Governed Experiment Runners

此目录是方法、攻击、指标和协议的唯一组合层，也是 governed records 的唯一写入层。

- `internal.py` 只消费精确的冻结内部 protocol、split manifest、case input manifest、
  `CegWmExperimentAdapter`、attack registry、metric registry 和公开方法对象；它调用
  真实 attack 与 joint-decision API，不复制方法、攻击或指标算法。artifact 当前
  像素与 attack specification 当前参数由 attacks 公共入口在 affine/grid 前重校验。
  当前执行仍拒绝
  `held_out_evaluation`。
- `record_writer.py` 是内部正式 records 的唯一 writer。构造时必须同时接收冻结
  case input manifest；每次 load 或物化前，它都通过
  `validate_run_case_record_collection` 复验完整 collection，逐 record 核对
  确定性 ID/sequence/attempt/parent、逐 unit provenance、routing、key/control、
  detector/config/preprocess、threshold/tau、几何操作和 experiments 层可靠性
  配置摘要，以及
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
bootstrap revision 权威由后续 A3b 外部信任链绑定，不由 A3a 自我声明。

包含外部 baseline 的高成本运行仍必须先接收
`experiments.protocol.comparison.PreflightApproval`，重新计算当前
`ComparisonProtocol` digest 并确认一致。本次内部 runner 不实现 baseline、
Notebook、GPU 调度、calibration、held-out evaluation 或 stage 推进。

CPU/synthetic runner tests 只证明编排、身份绑定、原子写入、resume/replay 和失败
语义，不构成候选晋升、runtime/GPU 或科学效果证据。
