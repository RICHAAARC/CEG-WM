# CEG-WM Method Architecture

本架构的实验侧方法接线统一使用 `internal_execution_components`，内部受治理
runner 统一使用 `governed_internal_runner`；两者都不得成为弱实现或替代真实方法。

## Authority

CEG-WM 由内容链、几何链和联合判定组成。内容链拥有水印证据；几何链只恢复坐标；联合判定只执行门控和重判。

本文件定义模块边界；数学原语以 [algorithm_primitives.md](algorithm_primitives.md) 为准，端到端调用与身份关系以 [method_mechanism.md](method_mechanism.md) 为准。

## Content Chain

```text
嵌入侧：
callback-18 临时 RGB8 ──> content_router / InSPyReNet mask
                         ├──> lf_carrier ──> local LF direction ──┐
                         └──> hf_carrier ──> global HF direction ─┤
                                                           v
                                                   content_embedder
                                                           |
                                                    delta_content
                                                           |
                                      runtime 在冻结 callback/model 边界物化

检测侧：
普通待检图像 + key + 公共资产
             ├──> InSPyReNet mask ──> lf_detector ──> s_lf_masked ──┐
             └──> hf_detector ──> s_hf ──┤
                                         v
                                 content_detector ──> D_M
```

CEG-WM 内容链的七项正式职责是 `content_router`、`lf_carrier`、
`hf_carrier`、`content_embedder`、`lf_detector`、`hf_detector` 和
`content_detector`。carrier 只提供模板/写入方向，embedder 独占 LF/HF 组合和总预算，
分支 detector 独立产生盲分数，content detector 只组合检测统计；runtime 只在冻结
callback/model/dtype 边界物化 embedder 给出的更新并把实际张量与 realized
combined total norm/relative L2 返回给
embedder 判定，不拥有路由、组合写入、预算判定或检测算法。current router 只输出
`M_embed`、全一 HF support、mask identity/digests；current embedder 只按
`normalize(normalize(T_hf)+normalize(M_embed*T_lf))` 构造方向并拥有
nominal/limit、materialization reconciliation 与 realized combined total
norm/relative L2。`A`、互补双 mask、`a`、direction dot/c(a) 和 routed/
route-disabled records 只属于 historical exact replay。

当前 actual-dtype 内容预算把 nominal 与 hard limit 同时冻结为 `3/250`。对
callback 18 actual baseline `z0`，runtime 只实现
`z_s=cast_actual(fp32(z0)+s*delta_content_nominal)`，返回
`delta_actual_s=fp32(z_s)-fp32(z0)`、binary16 bitwise replay 和 row-major
binary32 realized 测量。`content_embedder` 直接比较
`norm32(delta_actual_s) <= f32((3/250)*norm32(fp32(z0)))`，必要时在 binary32
`[0,1]` 上二分到无新 representable midpoint，选择最大非零可行 scale 或 fail
closed。ratio/utilization 只作诊断，不存在 tolerance、actual 下限或 runtime
budget policy。

首个 `hf_sparse_tail` 候选固定为高频剩余经 sparse tail 后直接 L2 normalize，只在
normalized-correlation 评分时中心化；该顺序具有 historical DirectHF 来源，但历史
名称与成功证据不进入本项目身份。LF 的新白化 matched-score 设计、路由、组合、Q/K
与 runtime 的具体候选已在
[candidate_specifications.md](candidate_specifications.md) 中关闭；尚未冻结的是
候选的实验晋升结果。`routing_stqr` 与旧 uniform combination 路线已形成
producer-bound development 负证据，不再是当前候选。继任设计以
`routing_inspyrenet_salient_local_lf` 重建显著目标内部 mask、以全局 HF 加局部 LF
写入，并以独立 masked-LF null whitening 和 `max(z_hf,z_lf_masked)` 检测；当前状态为
`design_candidate_implementation_authorized`，implementation admission 为 `YES`，且只
授权本地独立 revisions 实施。四者的 CPU/API source implementation 已在
`d88703689a0ea0487ad3a4553d060e5bf1a762cd` 闭合，并由
`independent_salient_local_lf_experiment_adapter_review:019fed21-be70-7803-aca0-6049bb279dfd:d88703689a0ea0487ad3a4553d060e5bf1a762cd:APPROVE`
独立审核；候选 readiness、真实 checkpoint/runtime smoke、实验 protocol、masked-LF W、
quality 定义、科学验证与晋升仍未闭合。
`content_combination_saliency_max_standardized` 保持 `diagnostic_only=true`、
`promoted=false`；正式 detector 保持 HF-only，quality gate 尚未定义。
当前冻结的 `D_M` 候选仍是 HF-only 的 HF direct score；只有新组合候选
通过预登记晋升后，`content_detector` 才可消费 `s_lf` 与 `s_hf` 形成组合 `D_M`。

两链共享的 root-key、KDF、PRG、wrong-key 与 public-noise 语义由 CEG-WM 自有
`key_schedule_sha256_counter` 候选统一定义。内容链与几何链只消费该共享责任的
派生结果，不得各自实现不兼容的随机协议。

完整 CEG-WM 身份要求内容自适应路由、LF、HF、Q/K、回正与联合判定全部通过各自门禁。LF/路由未晋升可以形成诚实负结果，但 HF-only + geometry 不得以完整 CEG-WM 方法成功发布；缩减方法需要重新命名和独立授权。

## Geometry Chain

```text
待检图像
   ↓
冻结模型中的 Q/K 观测
   ↓
同步与 crop / scale / rotation 估计
   ↓
可靠性判断
   ↓
图像回正或显式失败
```

几何链不消费内容分数，不产生水印阳性，不读取嵌入端私有状态。

## Joint Decision

联合判定先运行冻结内容检测器。只有近阈值负样本才进入几何资格检查；只有可靠几何才允许回正；回正后使用同一检测器和同一阈值。

当前 content detector 候选冻结为 HF direct score。LF/HF 组合成为正式 content detector 之前，LF 不得静默改变 near-threshold 或 rescue 语义。

## Dependency Direction

```text
main.shared
   ├── main.content_chain
   └── main.geometry_chain
              ↓
      main.joint_decision
              ↓
         main public API
              ↓
            runtime
              ↓
     experiments.methods
```

`main.content_chain` 与 `main.geometry_chain` 不互相依赖。runtime 不得拥有最终判定规则；experiment adapter 不得复制方法算法。

当前已实现并登记的 readiness 责任固定为以下 13 项，不允许别名、职责折叠或用
单个集中代理文件代替：

```text
main/shared/key_schedule.py                key_schedule
main/content_chain/routing.py              content_router
main/content_chain/lf_carrier.py           lf_carrier
main/content_chain/hf_carrier.py            hf_carrier
main/content_chain/embedder.py              content_embedder
main/content_chain/lf_detector.py           lf_detector
main/content_chain/hf_detector.py           hf_detector
main/content_chain/detector.py              content_detector
main/geometry_chain/qk_sync.py              qk_geometry_sync
main/geometry_chain/transform_estimator.py  geometric_transform_estimator
main/geometry_chain/reliability.py          geometry_reliability
main/geometry_chain/rectifier.py            image_rectifier
main/joint_decision/detector.py             conditional_recovery_decision
```

`content_router` 仍是唯一 mask/route 职责；继任候选只允许其拥有冻结 InSPyReNet
`M_embed`、全一 HF support 与 identity/digests，不增加组件职责。
`content_embedder` 独占无 `a/w` 的 global-HF/local-LF 写入、共同总预算、
nominal/limit、materialization scale/attempt/integrity/
budget status 与 realized combined total norm/relative L2，以及 active/combined
零方向失败；
`lf_detector` 独占盲 LF 分数；旧 `lf_null_whitened_matched_score` 与新
`lf_saliency_masked_null_whitened_matched_score` 是不同身份，后者必须独立拟合
32-clean null `W`，
不改变 `lf_carrier` 或现有 readiness；`geometry_reliability` 独占 estimator 原始指标上的
合取门。候选 registry 现在是 15 个 ID（14 个具名候选加 1 个 routing 强制对照）；
CPU/synthetic 实现不等于实验晋升，该计数与这里的 13 项实现职责不是同一计数。

current records/controls 固定为 clean、HF-only、masked-LF causal、
global-HF+local-LF、LF-disabled 和失败。historical `routing_stqr` 的 `A`/双-mask/
disabled-uniform 与旧 combination 的 `a/u_content(a)/dot/c/routed` 只可按原 producer
和 package/record identity 重放，不得重签为 current readiness、stage 或 control。

`content_direction`、`active_lf_direction`、`active_hf_direction` 及 target
components 只绑定 nominal 组合公式；actual hard limit 只作用于最终 combined
content delta，不定义或观测 LF/HF actual branch decomposition。geometry 写入与
geometry/total budget 仍是独立职责。

每个责任还必须绑定 `candidate_specifications.md` 中 policy 规定的候选 ID 和独立方法特异性验收节点；路径存在或 AST 结构通过都不等于方法完成。

## Configuration Identity

能够改变检测统计解释的配置必须进入方法配置摘要，包括：

- HF carrier、HF direct score 与 content detector 配置；
- root-key encoding、stable serialization、KDF/PRG、职责域和 quantile-table digest；
- LF 载体、路由和组合配置；
- 图像预处理与尺寸；
- Q/K 层、头、步和特征聚合配置；
- 几何可靠性规则；
- 原图阈值、近阈值区间和重判阈值身份。

不同配置摘要产生的分数和阈值不得混用。

## Current Implementation Status

固定 13 项职责与 27 个 CPU/synthetic 方法行为节点已经实现，并由唯一 readiness
绑定候选摘要、真实 symbol、测试节点和独立语义审核。实际 stage/status 已由
独立 revisions 同步为 `experiment_ready / implemented`。冻结 SD3.5 candidate 的
callback、actual dtype、VAE、两层真实 Q/K、registered-key 重复确定性和
negative-key identity control 已通过真实 GPU qualification。当前正式 detector
仍为 HF-only；旧 routing/combination 是 producer-bound 历史负结果，新显著目标
四候选只完成 d887 CPU/API source implementation 及候选专属 overlay，尚未闭合正式
checkpoint/runtime smoke、实验 protocol、masked-LF W、quality 定义或科学晋升，
`full_ceg_wm_eligible=false`。既有 runtime 证据与实验准备基础设施都不是 `tau`、
confirmation 结果、Calibration Locked、完整联合 FPR、几何恢复效果、正式 evaluation
或科学效果证据，也不晋升
LF/routing/组合/geometry。后续门序见
[research_construction_roadmap.md](research_construction_roadmap.md)。
