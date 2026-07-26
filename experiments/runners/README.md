# Governed Experiment Runners

此目录是方法、攻击、指标和协议的唯一组合层，也是 governed records 的唯一写入层。

包含外部 baseline 的高成本运行必须先接收 `experiments.protocol.comparison.PreflightApproval`，重新计算当前 `ComparisonProtocol` digest 并确认一致。runner 必须记录成功、失败和被排除的尝试，禁止静默丢弃不利结果或在运行后改写公平比较条件。

runner 只编排真实实现，不得成为核心方法、攻击或指标的替代实现。当前 CEG-WM 不提供具体 runner。
