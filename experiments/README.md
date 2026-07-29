# Experiment System

此目录保存研究项目的实验体系。共享协议与可追溯记录位于内层，方法、攻击和指标保持正交，runner 是唯一组合与 governed records 写入层。

| path | responsibility | current_state |
| --- | --- | --- |
| `protocol/` | 内部设计验证、外部比较、preflight 和 records。 | 已实现冻结的内部 split/访问、13 职责验证矩阵、逐样本记录约束，以及通用外部比较结构。 |
| `methods/` | 项目方法及 baseline 的协议适配。 | 仅有边界说明。 |
| `attacks/` | 与方法正交的攻击变换。 | 仅有边界说明。 |
| `metrics/` | 只依赖协议的指标计算。 | 仅有边界说明。 |
| `runners/` | 组合组件、校验 preflight、执行并写 records。 | 仅有边界说明。 |

未来 runner 在写入内部 run/case collection 前必须只调用
`protocol.internal_validation.validate_run_case_record_collection` 这一正式入口；
不得直接调用 `internal_records` 的私有结构 helper。

内部协议实现只关闭 schema、切分隔离、访问、状态和记录语义，不表示任何候选已经
晋升，也不表示 calibration、正式运行或科学有效性已经完成。
