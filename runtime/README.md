# Runtime Layer

`runtime/` 保存模型适配、HF carrier/HF detector 执行、Q/K 观测、生成与检测执行
能力，只允许依赖 `main/` 公开接口。模型后端、设备选择和推理流程放在此层；
near-threshold、几何救援和最终判定仍属于 `main/`。

当前 Batch 1 已实现：

- `runtime_sd35_flowmatch` 冻结配置的严格解析与稳定摘要；
- CPU/CUDA 设备选择及不可用设备的 fail-closed 错误；
- 后端协议、实际后端身份核验和单次初始化生命周期；
- 不加载模型的 mock backend 控制流测试。

当前尚未实现模型下载/加载、callback 写入、VAE 编解码、真实 Q/K 捕获、runner、
Notebook 或 GPU qualification。Batch 1 的通过只证明配置和 adapter 控制面可以进入
下一本地构建批次，不是 `runtime_verified` 或科学证据。
