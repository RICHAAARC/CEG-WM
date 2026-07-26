# Attack Transformations

CEG-WM 未来在此实现与方法正交的攻击、失真或编辑变换。首个设计范围至少覆盖 crop、scale、rotation 及非几何失真，但当前没有攻击实现。

攻击实现应只依赖 `experiments/protocol/` 并返回协议对象，不直接写 governed records，也不得读取方法私有状态。
