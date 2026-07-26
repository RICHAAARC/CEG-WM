# Experiment System

此目录保存研究项目的实验体系。共享协议与可追溯记录位于内层，方法、攻击和指标保持正交，runner 是唯一组合与 governed records 写入层。

| path | responsibility | current_state |
| --- | --- | --- |
| `protocol/` | 内部设计验证、外部比较、preflight 和 records。 | 当前只有通用外部比较与最小 record 结构。 |
| `methods/` | 项目方法及 baseline 的协议适配。 | 仅有边界说明。 |
| `attacks/` | 与方法正交的攻击变换。 | 仅有边界说明。 |
| `metrics/` | 只依赖协议的指标计算。 | 仅有边界说明。 |
| `runners/` | 组合组件、校验 preflight、执行并写 records。 | 仅有边界说明。 |

内部 LF/HF、几何和联合判定验证协议尚未实现。目录存在不表示具体实验已经实现或产生证据。
