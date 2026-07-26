# Experiment Protocol

此目录保存与具体方法、攻击和运行后端解耦的共享实验契约，不依赖 runtime、methods、attacks、metrics、runners 或治理代码。

## 当前实现

- `comparison.py`：定义外部 baseline 比较所需的 `ComparisonMethodSpec`、`ComparisonProtocol`、`PreflightApproval` 和运行前校验。
- `records.py`：定义带完整 provenance 的 `ExperimentRecord` 及其轻量校验。

LF/HF、Q/K 几何和联合判定的内部设计验证协议尚未实现，不得用当前外部 comparison 接口冒充。外部 baseline 与项目方法进入高成本运行前，必须共同固定样本与切分 manifests、生成条件、随机策略、输出规格、攻击与指标集合、调参与计算预算、失败与排除规则。

每条 record 必须追溯 comparison protocol、sample manifest、方法配置、方法代码、模型 revision、seed 和执行状态。`success` 只能保存有限数值且不得携带失败/排除原因；`failed` 与 `excluded` 不得保存 metric value，并分别只允许对应的原因字段。当前实现是通用协议结构，不是具体实验设计或实验结果。
