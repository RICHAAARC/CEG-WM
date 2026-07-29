# Runtime Configurations

`runtime_sd35_flowmatch.json` 是首个真实 runtime 候选的冻结配置。它只声明
`runtime_sd35_flowmatch` 已登记的模型、revision、调度器、生成参数、dtype、VAE
协议、callback、Q/K 层和依赖锁，不包含方法实现、凭据或运行结果。

本地 Batch 1 先解析并核验该配置，通过 mock backend 验证设备选择和 adapter
控制流；后续 exact candidate 的真实模型加载、callback、VAE、Q/K 已通过 Colab
GPU qualification。该结果只证明此冻结配置的 runtime 边界，不证明科学效果；
未来配置或 candidate 身份变化必须重新 qualification。
