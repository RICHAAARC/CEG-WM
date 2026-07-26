# Runtime Layer

`runtime/` 未来保存模型适配、HF carrier/HF detector 执行、Q/K 观测、生成与检测执行能力，只允许依赖 `main/` 公开接口。

模型后端、设备选择和推理流程放在此层；near-threshold、几何救援和最终判定仍属于 `main/`。当前没有 runtime 实现。
