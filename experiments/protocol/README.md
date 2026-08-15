# Experiment Protocol

此目录保存与具体方法、攻击和运行后端解耦的共享实验契约，不依赖 runtime、methods、attacks、metrics、runners 或治理代码。

## 当前实现

- `hf_only_reference_protocol.py`：加载 hf_only_reference_protocol 的 HF-only reference spec、离线 PartiPrompts
  snapshot/明文 roster 和两个 compact manifests，并确定性物化既有
  `FrozenSplitManifest`。bundle loader 同时核对文件 SHA、权威 method/runtime
  source bundle、稳定的 method-adapter 子配置摘要、key family、每 split 4096
  source clusters、prompt 互斥并集和 `(category, challenge)` 分层平衡。该模块只
  冻结公式身份、两阶段运行顺序与工作量，不实现 hf_only_reference_validation metrics、不运行模型、不产生结果。
- `internal_splits.py`：定义 unit/case/source-cluster identity、八个互斥职责 split、
  显式 frozen manifest 和当前执行访问门；授权对象必须匹配当前 access identity
  与精确允许集合，伪造或扩展 grant 失败，当前访问 `held_out_evaluation` 必须
  fail closed。
- `internal_matrix.py`：逐项连接 13 项方法职责与科学问题、split、metrics、
  negative controls、promotion gates 和 record field groups，并定义停止/晋升前置门。
- `internal_records.py`：定义逐样本正式记录，保留 raw/rectified detector 身份、
  LF/HF/combined、routing、geometry、threshold、key/control、decision 和 provenance，
  并区分 `success`、`failed`、`excluded`、`retry`，对失败进一步区分
  `resource_failure`、`execution_failure` 与 `scientific_failure`；provenance 还绑定 candidate config、
  独立 input manifest、execution config 和 resource identity 摘要。其中只保留由正式入口调用的
  私有 collection 结构校验 helper，验证所有非初始 outcome 的可重试 parent、
  连续 attempt、冻结上限、结构化 promotion stop 和 stop 后禁止继续。
- `hf_only_threshold_fit_records.py`：hf_only_reference_validation 阈值拟合单元记录 v2 强制保存
  `execution_evidence_kind`；历史 v1 记录不得恢复或汇总为 v2 正式结果，CPU 合成记录也不得由
  正式 finalizer 接受。
- `internal_case.py`：保存写入器和 runner 共同消费的冻结逐 case 输入清单；
  每个 unit 在执行前固定 artifact/attack/metric、routing、key/control、detector
  binding、raw/rectified detector 与 threshold、content/geometry callable 的显式
  formal-runner semantic declaration 摘要、几何操作和 experiments 层可靠性配置
  摘要，并从
  run/case/manifest/unit/attempt 确定性派生 record ID。
- `internal_record_registry.py`：保存随 execution package 分发的精确 record schema
  binding 与允许字段集合。开发侧 Markdown 只镜像登记，不是 runner 的运行时输入。
- `internal_validation.py`：加载并校验
  `configs/experiments/internal_scientific_validation_protocol.json` 的冻结协议身份；
  唯一正式 collection 入口 `validate_run_case_record_collection` 要求精确的
  `FrozenInternalValidationProtocol` 与 `FrozenSplitManifest` 实例，拒绝 duck object
  和 subclass，重算两者摘要并逐 record 核对 provenance 与 unit/split assignment。
- `comparison.py`：定义外部 baseline 比较所需的 `ComparisonMethodSpec`、`ComparisonProtocol`、`PreflightApproval` 和运行前校验。
- `records.py`：定义带完整 provenance 的 `ExperimentRecord` 及其轻量校验。

内部协议已因 detector-mode-aware hf_only_reference_validation 前置门从 v1/1.0.0 显式升为 v2/2.0.0。
v1 record 结构仍可由历史工具读取，但其语义不得在 v2 下重新验证或迁移冒充；
HF-only `content_threshold_fit` 需要 `hf_reference_candidate_frozen`，combined
模式仍需要 `content_branch_promotion_gate_passed`，缺失/未知 mode 一律拒绝。
`hf_detector_reference_gate_passed` 是未来结果门，禁止作为 prerequisite。

内部验证协议与外部 comparison 是两个不同表面，不得互相冒充。当前 internal_validation_protocol 只建立
可执行的协议约束和冻结配置；没有访问 final held-out、没有执行 calibration 或
攻击矩阵，也没有产生科学 records。外部 baseline 与项目方法进入高成本运行前，
仍必须共同固定样本与切分 manifests、生成条件、随机策略、输出规格、攻击与指标集合、
调参与计算预算、失败与排除规则。

每条 record 必须追溯 comparison protocol、sample manifest、方法配置、方法代码、模型 revision、seed 和执行状态。`success` 只能保存有限数值且不得携带失败/排除原因；`failed` 与 `excluded` 不得保存 metric value，并分别只允许对应的原因字段。当前实现是通用协议结构，不是具体实验设计或实验结果。
`held_out_evaluation` 表示最终 held-out evaluation 职责；该语义化名称不含阶段编号。

`salient_local_lf_mask_write_validation.py` 验证 authored 8-cluster roster 的 canonical
authority、2+8/max1 unit roster、future deny axes 与两项 producer-bound historical
negative。它只准入 mask/write/quality development pilot，不准入 masked-LF
whitening、detector、max statistic、Q/K、threshold、FPR、promotion 或 formal stage。
