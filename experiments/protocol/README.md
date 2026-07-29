# Experiment Protocol

此目录保存与具体方法、攻击和运行后端解耦的共享实验契约，不依赖 runtime、methods、attacks、metrics、runners 或治理代码。

## 当前实现

- `internal_splits.py`：定义 unit/case/source-cluster identity、八个互斥职责 split、
  显式 frozen manifest 和当前执行访问门；当前访问
  `held_out_evaluation` 必须 fail closed。
- `internal_matrix.py`：逐项连接 13 项方法职责与科学问题、split、metrics、
  negative controls、promotion gates 和 record field groups，并定义停止/晋升前置门。
- `internal_records.py`：定义逐样本正式记录，保留 raw/rectified detector 身份、
  LF/HF/combined、routing、geometry、threshold、key/control、decision 和 provenance，
  并区分 `success`、`failed`、`excluded`、`retry`。
- `internal_validation.py`：加载并校验
  `configs/experiments/internal_scientific_validation_protocol.json` 的冻结协议身份。
- `comparison.py`：定义外部 baseline 比较所需的 `ComparisonMethodSpec`、`ComparisonProtocol`、`PreflightApproval` 和运行前校验。
- `records.py`：定义带完整 provenance 的 `ExperimentRecord` 及其轻量校验。

内部验证协议与外部 comparison 是两个不同表面，不得互相冒充。当前 A-1 只建立
可执行的协议约束和冻结配置；没有访问 final held-out、没有执行 calibration 或
攻击矩阵，也没有产生科学 records。外部 baseline 与项目方法进入高成本运行前，
仍必须共同固定样本与切分 manifests、生成条件、随机策略、输出规格、攻击与指标集合、
调参与计算预算、失败与排除规则。

每条 record 必须追溯 comparison protocol、sample manifest、方法配置、方法代码、模型 revision、seed 和执行状态。`success` 只能保存有限数值且不得携带失败/排除原因；`failed` 与 `excluded` 不得保存 metric value，并分别只允许对应的原因字段。当前实现是通用协议结构，不是具体实验设计或实验结果。
`held_out_evaluation` 表示最终 held-out evaluation 职责；该语义化名称不含阶段编号。
