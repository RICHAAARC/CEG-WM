# Experiment Method Adapters

CEG-WM 未来在此把 `main/` 与 `runtime/` 的真实能力适配到 `experiments/protocol/`。适配器负责接口转换和协议对接，不得重新实现或削弱核心机制，也不得直接写 governed records。

当前没有项目方法适配。内部组件验证与外部 baseline 比较使用不同协议表面，但共享 provenance 和失败记录原则。
