# CEG-WM Research Design

本目录保存 CEG-WM 的权威研究定义和算法边界：

- [research_definition.md](research_definition.md)：研究目标、访问模型、攻击者能力、非目标和证据边界。
- [method_architecture.md](method_architecture.md)：内容链、几何链、联合判定和依赖方向。
- [content_chain.md](content_chain.md)：HF carrier/direct score 冻结职责、LF 候选、路由和组合验证要求。
- [geometry_chain.md](geometry_chain.md)：Q/K 同步、几何估计、可靠性、回正和盲检测边界。
- [joint_decision.md](joint_decision.md)：原图检测、近阈值门控、条件恢复和同阈值重判。
- [evaluation_design.md](evaluation_design.md)：内部设计验证、外部比较、切分、攻击和指标要求。
- [candidate_specifications.md](candidate_specifications.md)：具名候选的输入输出、算法、配置身份、来源边界、失败语义和晋升门。
- [algorithm_primitives.md](algorithm_primitives.md)：密钥、HF carrier/direct score、LF、路由、组合、Q/K、恢复和判定的数学原语。
- [method_mechanism.md](method_mechanism.md)：嵌入、检测、身份、失败传播和组件晋升的端到端机制。
- [research_construction_roadmap.md](research_construction_roadmap.md)：从研究定义到固定 FPR `0.001` 级别论文证据的构建与验证路线。

这些文档定义项目要研究什么以及实现不得偏离什么，不构成方法实现或效果证据。任何与这里冲突的历史代码、配置、Notebook 或实验脚本都不是项目权威。

当前 CPU/synthetic 方法实现采用唯一的 13 项职责组件口径；精确 responsibility、
固定路径和候选绑定见 [candidate_specifications.md](candidate_specifications.md)。
候选 registry 现在固定为 15 个 ID（14 个具名候选加 1 个 mandatory routing
control），该计数不得与 13 项职责混淆。新增的
`routing_inspyrenet_salient_local_lf`、
`content_embedding_global_hf_local_lf`、
`lf_saliency_masked_null_whitened_matched_score` 和
`content_combination_saliency_max_standardized` 统一为
`design_candidate_pending_implementation`，`implementation_admission=NO`；不增加第
14 项职责，也不重签当前 readiness、stage 或既有 runtime qualification。

旧 `routing_stqr` 的 `A`、双 mask、routed/route-disabled 对照与旧
`content_uniform_combination` 的 `a/u_content(a)`、direction dot/c 和函数族只服务
原 producer/package/record 的 historical exact replay。它们的代码或历史证据存在，
不表示 current candidate、执行授权或 downstream dependency；current 显著目标路线
仅使用 `M_embed`、全一 HF support、无 `a/w` 的 global-HF+local-LF 写入和独立
masked-LF/max-statistic 检测身份。

## Derived Method Diagrams

- [Prompt → Watermarked Image](diagrams/prompt_to_watermarked_image.svg)
  ([Drawio](diagrams/prompt_to_watermarked_image.drawio))
- [Watermark Presence And Key Attribution Decision](diagrams/watermark_detection_and_key_attribution.svg)
  ([Drawio](diagrams/watermark_detection_and_key_attribution.drawio))

两张图只是上述十份权威设计的派生可视化，不增加
`.codex/research_state/research_definition.yaml` 的 design path。图与 Markdown、
contract 或 policy 冲突时以后者为准。图中方法职责已有 CPU/synthetic 实现，冻结
SD3.5 runtime 边界也已通过真实 GPU qualification；图中标为 planned 的内容仍表示
尚未完成的候选晋升、calibration 或正式实验机制，不是效果证据。
