# Core Watermark Method Package

`main/` 是 CEG-WM 可独立抽离的最小方法包，包含 `shared/`、`content_chain/`、
`geometry_chain/`、`joint_decision/` 和公开 API，并按冻结架构实现固定的 13 项职责。
`content_chain` 中的 embedder、LF detector 与 `geometry_chain` 中的 reliability
保持独立路径，不得折叠到 carrier、content detector 或 transform estimator。

此层不得导入 runtime、experiments、paper artifacts、notebooks、infrastructure、tests 或任何外层控制平面，也不得保存实验 records、论文构建或模型后端适配。

当前项目处于 `runtime_verified / implemented`。六个方法构建批次已完成
13 项职责实现、CPU/synthetic 行为验证、readiness 收口及绑定实现 revision 的独立语义
审核；唯一 `method_readiness.yaml` 已在第 6 批完成后创建并记录 `approve`。这些事实此前
构成独立方法阶段迁移的前置证据；后续真实 runtime qualification 和本次
stage-only revision 均不修改方法机制。

内容链已经实现 S/T/R/Q 数值路由、LF/HF carrier、blind detector、共同总预算写入组合
及 hf_only_standardized_score/weighted_hf_lf_standardized_score/maximum_hf_lf_standardized_score 诊断，但正式内容判定仍保持 `hf_only`。LF、routing 和组合行为尚未获得
实验晋升，不能据此选择或宣称正式 `a`、组合函数或阈值。组合写入要求两个 carrier
绑定同一个经重算验证的 route；routed result 保留不可变 S/T/R/Q 观测，并据此重演
插值与路由公式；组合诊断只接受同一普通图像编码观测。

几何链实现两登记层 Q/K 四通道 relation、几何密钥投影、固定 similarity/dihedral
搜索、独立 reliability 合取和 PyTorch 图像回正。联合判定先运行原图内容检测，只有
`[tau_rescue,tau)` 内的内容负样本才惰性进入几何链；可靠回正后复用同一 detector
operation、HF branch identity、注册 key、预处理和 `tau` 重判。几何链不产生水印
阳性；不可变结果只允许 raw 或 rectified content score 形成 `joint_content_positive`，
并以 `positive_path` 区分两条阳性控制流。

正式 content detector 仍为 `hf_only`，因此联合结果固定
`full_ceg_wm_eligible=false`，只表示 construction-candidate 行为，不表示 LF/routing、
reduced-scope 身份或完整 CEG-WM 已获实验晋升。独立真实 GPU qualification 已证明
冻结 SD3.5 callback、actual dtype、VAE 和两层 Q/K runtime 边界可用；它不证明完整
FPR、几何恢复效果、鲁棒性或科学效果，FPR 仍须由后续实验层结合 null 标签统计。
