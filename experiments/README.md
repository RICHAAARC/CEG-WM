# Experiment System

此目录保存研究项目的实验体系。共享协议与可追溯记录位于内层，方法、攻击和指标保持正交，runner 是唯一组合与 governed records 写入层。

| path | responsibility | current_state |
| --- | --- | --- |
| `protocol/` | 内部设计验证、外部比较、preflight 和 records。 | 已实现冻结的内部 split/访问、13 职责验证矩阵、逐样本记录约束，以及通用外部比较结构。 |
| `methods/` | 项目方法及 baseline 的协议适配。 | 已实现只委托 `main`/`runtime` 公开 API 的 CEG-WM 内部薄适配器；尚无 baseline。 |
| `attacks/` | 与方法正交的攻击变换。 | 已实现冻结 identity/crop/scale/rotation/组合几何攻击；尚无非几何攻击。 |
| `metrics/` | 只依赖协议的指标计算。 | 已实现内部逐 case/聚合 fixed-FPR、质量、routing、LF/HF、几何、可靠性、回正与 rescue 指标。 |
| `runners/` | 组合组件、校验冻结输入、执行并写 records。 | 已实现内部 governed runner、唯一原子 writer、resume/replay 和显式资源/执行/科学失败分类；不含 baseline、Notebook 或正式执行。 |

内部 runner 在写入 run/case collection 前只调用
`protocol.internal_validation.validate_run_case_record_collection` 这一正式入口；
不得直接调用 `internal_records` 的私有结构 helper。

内部协议、internal_execution_components 组件与当前 runner 只关闭 schema、切分隔离、访问、真实 API 编排、
唯一写入、resume/replay 和失败分类语义，不表示任何候选已经晋升，也不表示
calibration、正式运行、GPU qualification 或科学有效性已经完成。
