# CEG-WM Research Design

本目录保存 CEG-WM 的权威研究定义和算法边界：

- [research_definition.md](research_definition.md)：研究目标、访问模型、攻击者能力、非目标和证据边界。
- [method_architecture.md](method_architecture.md)：内容链、几何链、联合判定和依赖方向。
- [content_chain.md](content_chain.md)：HF carrier/direct score 冻结职责、LF 候选、路由和组合验证要求。
- [geometry_chain.md](geometry_chain.md)：Q/K 同步、几何估计、可靠性、回正和盲检测边界。
- [joint_decision.md](joint_decision.md)：原图检测、近阈值门控、条件恢复和同阈值重判。
- [evaluation_design.md](evaluation_design.md)：内部设计验证、外部比较、切分、攻击和指标要求。
- [candidate_specifications.md](candidate_specifications.md)：具名候选的输入输出、算法、配置身份、失败语义和验证门。
- [algorithm_primitives.md](algorithm_primitives.md)：密钥、HF carrier/direct score、LF、路由、组合、Q/K、恢复和判定的数学原语。
- [method_mechanism.md](method_mechanism.md)：嵌入、检测、身份、失败传播和组件验证的端到端机制。
- [research_construction_roadmap.md](research_construction_roadmap.md)：从研究定义到固定 FPR `0.001` 级别论文证据的构建与验证路线。

这些文档只定义项目要研究什么、方法公式、组件接口和实现不得偏离的不变量。

方法采用唯一的 13 项职责组件口径；精确 responsibility、固定路径和候选绑定见
[candidate_specifications.md](candidate_specifications.md)。live candidate registry 固定为
27 个 ID（26 个具名候选加 1 个 mandatory routing control），不得与 13 项职责混淆；
新增七个 identity 是 `adopted_design_unimplemented / not_yet_tested`，不重签当前
12-identity/17-node readiness snapshot。

内容链权威路线是 InSPyReNet soft semantic probability `M` + deterministic
Sobel/P95 texture `T` 的逐图软路由、独立 keyed LF/HF carrier、共同 `3/250` 总预算
以及 `max(z_hf_soft,z_lf_soft)` 内容统计。几何链负责 Q/K 同步、crop/scale/rotation
估计、可靠性和回正；联合判定只对近阈值负样本调用几何，并在回正后使用同一内容
检测器和同一阈值重判。

## Derived Method Diagrams

- [Prompt → Watermarked Image](diagrams/prompt_to_watermarked_image.svg)
  ([Drawio](diagrams/prompt_to_watermarked_image.drawio))
- [Watermark Presence And Key Attribution Decision](diagrams/watermark_detection_and_key_attribution.svg)
  ([Drawio](diagrams/watermark_detection_and_key_attribution.drawio))

两张图只是上述十份权威设计的派生可视化，不增加
`.codex/research_state/research_definition.yaml` 的 design path。图与 Markdown、
contract 或 policy 冲突时以后者为准。
