# Baseline Registry

外部对比方法在进入实验矩阵前必须登记。空表表示当前 CEG-WM 尚未选择 baseline，不表示 baseline 要求可以跳过。

| baseline_name | source | pinned_version | license | adapter_path | config_path | deviations |
| --- | --- | --- | --- | --- | --- | --- |

## 登记规则

1. `source` 指向论文、官方仓库或正式发布页。
2. `pinned_version` 使用不可变版本、commit 或内容 digest。
3. `adapter_path` 位于 `experiments/methods/baselines/`。
4. `config_path` 位于 `configs/baselines/`。
5. 对上游实现的任何语义修改必须记录在 `deviations`。
6. 可选 vendored 源码位于 `third_party/`，不得成为未登记的项目实现层。

## 公平对比运行前约束

登记 baseline 只解决来源问题，不等于对比已经公平。每次比较必须在 `configs/experiments/` 建立 `ComparisonProtocol`，并在消耗 GPU 或远程资源前通过 preflight。协议至少固定：

- 样本和数据切分 manifests；
- 生成条件、随机策略和输出规格；
- 攻击矩阵与指标集合；
- calibration 与 evaluation 的独立切分；
- 调参与计算预算；
- 失败和排除规则；
- 每个方法的实现 revision、配置 digest 和已声明偏差。

Governed records 必须保存协议 digest 以及样本、方法代码、模型、配置和 seed provenance。没有这些字段的结果不能进入公平对比表格。
