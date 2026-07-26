# External Baseline Adapters

此目录只保存将已登记外部方法接入 `experiments/protocol/` 的薄适配器。上游源码不在这里重写；确需 vendoring 时放入 `third_party/` 并固定来源。

每个 baseline 必须：

1. 登记到 `docs/reference/baseline_registry.md`；
2. 在 `configs/baselines/` 固定实现 revision 和参数；
3. 声明相对上游实现的语义偏差；
4. 与项目方法共享 comparison protocol、预算、失败和排除规则；
5. 由 `experiments/runners/` 统一执行并写 governed records。

当前 CEG-WM 不包含具体 baseline 适配器。
