# CEG-WM Method Architecture

本架构的实验侧方法接线统一使用 `internal_execution_components`，内部受治理
runner 统一使用 `governed_internal_runner`；两者都不得成为弱实现或替代真实方法。

## Authority

CEG-WM 由内容链、几何链和联合判定组成。内容链拥有水印证据；几何链只恢复坐标；联合判定只执行门控和重判。

本文件定义模块边界；数学原语以 [algorithm_primitives.md](algorithm_primitives.md) 为准，端到端调用与身份关系以 [method_mechanism.md](method_mechanism.md) 为准。

## Content Chain

```text
嵌入侧：
callback-18 临时 RGB8 ──> content_router / semantic M + texture T
                         ├──> m_lf ──> lf_carrier ──> LF direction ──┐
                         └──> m_hf ──> hf_carrier ──> HF direction ──┤
                                                           v
                                                   content_embedder
                                                           |
                                                    delta_content
                                                           |
                                      runtime 在冻结 callback/model 边界物化

检测侧：
普通待检图像 + key + 公共资产
             └──> 重建 M/T 与 m_lf/m_hf
                         ├──> lf_detector ──> z_lf_soft ──┐
                         └──> hf_detector ──> z_hf_soft ──┤
                                                         v
              content_detector / soft max ──> diagnostic candidate（formal/default/joint D_M 仍为 HF-only）
```

CEG-WM 内容链的七项正式职责是 `content_router`、`lf_carrier`、
`hf_carrier`、`content_embedder`、`lf_detector`、`hf_detector` 和
`content_detector`。carrier 只提供模板/写入方向，embedder 独占 LF/HF 组合和总预算，
分支 detector 独立产生盲分数，content detector 只组合检测统计；runtime 只在冻结
callback/model/dtype 边界物化 embedder 给出的更新并把实际张量与 realized
combined total norm/relative L2 返回给
embedder 判定，不拥有路由、组合写入、预算判定或检测算法。router 只输出
`M/T`、`m_lf/m_hf` 和 identity/digests；embedder 只按
`normalize(normalize(m_hf*T_hf)+normalize(m_lf*T_lf))` 构造方向并拥有
nominal/limit、materialization reconciliation 与 realized combined total
norm/relative L2。`m_lf/m_hf` 是空间调制，不是 actual branch energy。

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
[candidate_specifications.md](candidate_specifications.md) 中关闭。内容路线使用
InSPyReNet soft probability `M`、deterministic Sobel/P95 texture `T`、两张和为一的
正软路由图、soft-routed LF/HF branch scores 和
`max(z_hf_soft,z_lf_soft)` 内容统计。

两链共享的 root-key、KDF、PRG、wrong-key 与 public-noise 语义由 CEG-WM 自有
`key_schedule_sha256_counter` 候选统一定义。内容链与几何链只消费该共享责任的
派生结果，不得各自实现不兼容的随机协议。

完整 CEG-WM 身份要求内容自适应路由、LF、HF、Q/K、回正与联合判定共同存在；
任一缩减方法都必须使用不同的方法身份。

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

当前正式/default/joint `D_M` 仍冻结为 HF-only 和既有阈值；原图与回正图共同调用
同一 HF-only detector/config identity 和同一阈值。语义—纹理 soft max 仅为
`implemented_not_scientifically_validated` diagnostic、未晋升候选，不进入正式判定。
未来晋升必须先完成独立分支 calibration、max threshold fit、固定 FPR/科学确认与
显式 promotion；晋升后的原图/回正图必须使用同一个新的 detector/config identity
和新阈值，不得继承旧 W/CDF、`tau` 或 HF-only threshold。

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

方法责任固定为以下 13 项，不允许别名、职责折叠或用
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

`content_router` 是唯一 observation/route 职责；它拥有 `M/T`、`m_lf/m_hf` 与
identity/digests，不增加组件职责。
`content_embedder` 独占无 `a/w` 的 soft-routed LF/HF 写入、共同总预算、
nominal/limit、materialization scale/attempt/integrity/
budget status 与 realized combined total norm/relative L2，以及 active/combined
零方向失败；
`lf_detector` 独占盲 LF 分数和 soft-route 专属 32-clean null `W` 消费；
`hf_detector` 独占盲 soft-routed HF 分数；`geometry_reliability` 独占 estimator
原始指标上的合取门。live candidate registry 是 27 个 ID（26 个具名候选加 1 个
routing 强制对照），与这里的 13 项职责不是同一计数。新增七个 identity 仅为
`adopted_design_unimplemented / not_yet_tested`；当前 readiness 仍绑定原 12 个
implemented identities、17 个 behavior nodes 和 13 项职责。

records/controls 固定为 clean、HF-only、LF-only、soft-routed LF/HF、
route-disabled 和显式失败。

`content_direction`、`active_lf_direction`、`active_hf_direction` 及 target
components 只绑定 nominal 组合公式；actual hard limit 只作用于最终 combined
content delta，不定义或观测 LF/HF actual branch decomposition。geometry 写入与
geometry/total budget 仍是独立职责。

每个责任必须绑定 `candidate_specifications.md` 中规定的候选 ID，并以方法特异性行为
检查验证；路径存在或 AST 结构本身不能代替职责实现。

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

方法验证门序见
[research_construction_roadmap.md](research_construction_roadmap.md)。
