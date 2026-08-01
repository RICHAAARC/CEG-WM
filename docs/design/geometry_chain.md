# Geometry Chain Design

真实 runtime 的几何观测边界登记为 `qk_observation`。该语义身份不改变几何链
“只恢复坐标、不得直接产生阳性”的权限边界。

## Objective

几何链利用待检图像中的 Q/K 观测估计 crop、scale 和 rotation，必要时恢复图像坐标，使内容检测器能够在与原图相同的统计语义下重判。

## Blind Detection Boundary

正式几何检测只允许使用：

- 待检图像；
- 检测密钥；
- 冻结方法配置；
- 冻结公共模型资产。

禁止使用：

- 原始参考图或未攻击生成图；
- embed record；
- 嵌入 latent、嵌入端 Q/K 缓存或统计；
- 由样本真实攻击参数直接构造的恢复提示；
- 仅为让当前样本恢复成功而调节的阈值。

## Candidate Stages

1. 从冻结模型位置提取 Q/K 观测。
2. 构造带密钥绑定的同步证据。
3. 估计 rotation、scale 和 crop/translation。
4. 输出参数估计、置信度、可辨识性和失败原因。
5. 只有可靠时生成回正图像。
6. 将回正图交回联合判定，不在几何链内调用阳性规则。

具体层、Q/K 算子、四通道 relation、keyed objective、同步写入、similarity 搜索、可靠性指标和回正协议已由 [candidate_specifications.md](candidate_specifications.md) 的 `qk_relation_similarity` 与 `rectification_similarity` 关闭。实验可以淘汰候选，但实现不得静默替换候选。

嵌入同步目标和盲检都必须从普通图像执行相同的 VAE-mode、公开噪声、
schedule-index-7、三路空文本、无-CFG image-only forward；生成 conditional Q/K
不属于首个候选。`rectification_similarity` 进一步冻结 dihedral-first 连续变换顺序、
W/V 双线性 matrix、四个 coverage/uniqueness deficit、两层四通道聚合、三轮局部邻域
和 first-win 平局规则；实现不得另选 optimizer。

## Reliability

`geometric_transform_estimator` 只输出最佳/次佳候选及 coverage、uniqueness、gap、
key margin、inlier、residual、boundary 和 identity margin 等原始指标。
`geometry_reliability` 是独立组件，只消费这些指标并执行冻结合取门；不得把该门
合并回 estimator，也不得读取或改变内容分数。

可靠性必须预先定义并单独校准，至少区分：

- 未发现同步证据；
- 参数不可辨识；
- 多候选歧义；
- 估计超出支持范围；
- 回正输出无效；
- 估计可靠。

错误 key、低 coverage、低 uniqueness、小 gap、高 residual、boundary 解、非有限量
或任何必需指标缺失都必须 fail closed。可靠性只控制是否允许回正，不能增加内容分数。

## Transform Scope

首个正式范围包括 crop、scale 和 rotation。translation 可以作为 crop/rotation 恢复所需参数处理。perspective、局部形变和生成式编辑不自动进入首个几何闭环，需要独立设计变更。

## Validation Questions

- 无攻击时是否接近恒等变换？
- 单一变换和组合变换下参数误差是多少？
- 错误密钥下是否拒绝几何同步？
- 不可靠样本是否稳定 fail closed？
- 回正是否改善同一内容检测器，而不是只改善几何代理指标？
- 回正是否引入不可接受的图像质量损失或假阳性？

## Current Status

Q/K 同步、变换估计、独立可靠性与图像回正已完成真实 CPU/synthetic 实现和
synthetic crop/scale/rotation、wrong-key、低可靠性拒绝及 inverse-warp 行为验证。
登记 SD3.5 的两层真实 `to_q`/`to_k` observation 已通过 GPU qualification；
该 runtime 事实不证明 crop/scale/rotation 估计、可靠性或回正的真实图像科学效果，
也不产生水印阳性或正式 FPR。实际阶段/status 已由独立 revisions 同步为
`experiment_ready / implemented`；该阶段只登记冻结实验协议与可追溯执行交付，
不提供 `tau`、confirmation 结果、Calibration Locked、正式 evaluation 或科学证据，
也不晋升 geometry 或任何内容分支。
