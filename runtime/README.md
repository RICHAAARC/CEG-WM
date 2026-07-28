# Runtime Layer

`runtime/` 保存模型适配、HF carrier/HF detector 执行、Q/K 观测、生成与检测执行
能力，只允许依赖 `main/` 公开接口。模型后端、设备选择和推理流程放在此层；
near-threshold、几何救援和最终判定仍属于 `main/`。

当前 Batch 1 已实现：

- `runtime_sd35_flowmatch` 冻结配置的严格解析与稳定摘要；
- CPU/CUDA 设备选择及不可用设备的 fail-closed 错误；
- 后端协议、实际后端身份核验和单次初始化生命周期；
- 不加载模型的 mock backend 控制流测试。

Batch 2 的本地非语义基础设施当前已实现：

- clean/watermarked 同基础 float16 latent 的防共享配对与 callback 全序列绑定；
- callback index 18 exactly-once 内容写入、binary16 RNE 独立重放和 actual tensor /
  `delta_content_actual` / realized total-relative L2 测量；
- generation VAE scaling/shift decode 与 detection posterior-mode encode；
- missing/duplicate/wrong callback、pair drift、非有限、overflow、写入消失和禁止
  posterior sampling 的 CPU fake/backend 失败语义。

上述基础设施通过不表示 Batch 2 整体完成。Batch 2 仍不拥有 actual-dtype budget
acceptance rule，结果只登记 `budget_acceptance_status=not_evaluated`；真实
SD3.5 callback、actual dtype 和 VAE 路径仍须 GPU qualification。当前也未实现模型
下载/加载、真实 Q/K 捕获、runner、Notebook 或 GPU qualification；本地 CPU 通过
不是 `runtime_verified` 或科学证据。
