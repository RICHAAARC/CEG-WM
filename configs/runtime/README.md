# Runtime Configurations

`runtime_sd35_flowmatch.json` 是首个真实 runtime 候选的冻结配置。它只声明
`runtime_sd35_flowmatch` 已登记的模型、revision、调度器、生成参数、dtype、VAE
协议、callback、Q/K 层和依赖锁，不包含方法实现、凭据或运行结果。

本地 Batch 1 只解析并核验该配置，通过 mock backend 验证设备选择和 adapter
控制流。真实模型加载、callback、VAE、Q/K 和 GPU qualification 属于后续批次与
Colab 门，不能由本配置存在或本地测试通过替代。
