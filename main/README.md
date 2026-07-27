# Core Watermark Method Package

`main/` 是 CEG-WM 可独立抽离的最小方法包。计划包含 `shared/`、`content_chain/`、
`geometry_chain/`、`joint_decision/` 和公开 API。未来实施固定为 13 项职责；
`content_chain` 中的 embedder、LF detector 与 `geometry_chain` 中的 reliability
都有独立路径，不得折叠到 carrier、content detector 或 transform estimator。

此层不得导入 runtime、experiments、paper artifacts、notebooks、infrastructure、tests 或任何外层控制平面，也不得保存实验 records、论文构建或模型后端适配。

当前处于 `method_construction_authorized` 的分批实施期。批次 1 共享
`key_schedule` 与批次 2 HF 内容主干均已完成双重独立审核；本 revision 实现批次 3
的 S/T/R/Q 数值路由、LF carrier、LF blind detector、共同总预算内容组合和
未晋升 C0/C1/C2 诊断。正式 content detector 仍保持 HF-only，不能据此选择
`a`、组合函数或阈值。组合写入要求两个 carrier 绑定同一个经重算验证的 route，
其中 routed result 保留实际不可变 S/T/R/Q 观测并据此重演插值与路由公式；
组合诊断只接受同一普通图像编码观测。批次 4 已实现独立几何链：两登记层真实 Q/K
四通道 relation、几何密钥投影、固定 similarity/dihedral 搜索、独立 reliability
合取和 PyTorch 图像回正。该 CPU/synthetic 实现不声称真实 SD3.5 Q/K 捕获已成立，
也不产生水印阳性；联合判定仍未实现。单个批次通过不表示完整方法完成，也不得提前
创建 method readiness。
