# CEG-WM 分层治理架构契约

## 项目定位

CEG-WM 是内容证据主判、几何条件恢复的双链生成式图像水印研究项目。此前已合法
进入 `method_construction_authorized`；当前 readiness 已在固定 13 项职责上绑定
12 个唯一候选身份、17 个 CPU/synthetic 方法行为节点和唯一 readiness，并经独立
语义审计。旧的 11 候选/28 节点 readiness 快照继续保留在 exact revision
`0258ccb2100bfe8b58d1a12079876841192528b3`，不再是当前权威。

阶段实施路径继续为
`research_defined → method_construction_authorized → method_implemented → runtime_verified → experiment_ready`。
实际阶段/status 已由独立 revisions 同步为 `experiment_ready / implemented`。
CPU/synthetic readiness 本身不证明 runtime；当前另有独立真实 GPU qualification
支持冻结 SD3.5 runtime 边界，但不证明正式 FPR 或科学效果；
正式 detector 仍为 HF-only；旧 routing/combination 只保留 producer-bound historical
replay。语义—纹理软路由五候选已由 producer
`02fe5dcc2b74482c9eb1e0b192b4a2ce79e0d9eb` 实现并经 revision-bound independent
exact audit 批准，状态为 `implemented_not_scientifically_validated`；soft max 仅为
diagnostic 且未晋升。hard salient-object
local-LF 四候选为 `superseded_without_scientific_adjudication`，
`full_ceg_wm_eligible=false`。实验准备基础设施闭环仅表示冻结协议与可追溯执行交付
已经就位，不提供 `tau`、confirmation 结果、Calibration Locked、正式 evaluation
或科学证据，也不晋升 LF/routing/组合/geometry。

`method_implemented` 的 readiness AST 审计只承担候选绑定、固定模块路径、symbol 调用和断言接线检查；它不能单独证明非代理实现。readiness 中的候选规格摘要是独立语义复核 revision 上的实现快照，必须从该 exact Git blob 重放，不能用当前工作树字节重签。当前候选规格仍是 live design authority，必须继续保留 policy/readiness 已绑定的全部实现身份，但可在不改写旧 readiness 快照的前提下登记尚未准入实现的设计候选。独立复核后的实现路径和方法特异性登记测试仍是 stale-protected surface；live candidate-spec 文件本身不因此成为旧 readiness 的不可变源码路径。

## 方法层

以下路径是计划边界；路径不存在或仅有说明文件时不得解释为能力已实现。

| path | responsibility | allowed project dependencies |
| --- | --- | --- |
| `main/shared/` | 两链共享的不可变类型、密钥语义和基础数值接口。 | none |
| `main/content_chain/` | router、LF/HF carrier、独立 content embedder、LF/HF detector 和 content detector。 | `main.shared` |
| `main/geometry_chain/` | Q/K 同步、变换估计、独立 reliability 和图像回正。 | `main.shared` |
| `main/joint_decision/` | 近阈值门控和恢复后同检测器同阈值重判。 | `main.shared`、`main.content_chain`、`main.geometry_chain` |
| `main/` public surface | 对外导出稳定方法 API，不重写内部机制。 | 上述 `main` 子层 |

内容链和几何链相互正交。几何链不得导入内容检测器；内容链不得读取几何可靠性。联合判定只消费两链公开结果，不得把几何分数转换成内容阳性。

`main/shared/key_schedule.py` 独占 root-key encoding、stable serialization、
KDF/PRG、职责域、wrong-key/public-noise 和 golden-vector 责任。LF、HF 与 Q/K
组件必须绑定同一 `key_schedule_sha256_counter` 候选，不能各自调用框架 RNG。

方法完成面固定为 13 项职责组件：共享 1 项、内容链 7 项、几何链 4 项、联合判定
1 项。`content_embedder`、`lf_detector`、`geometry_reliability` 各有独立路径，
不得折回 carrier、content detector 或 transform estimator。候选 registry 的
20 个 ID（19 个具名候选加 1 个 mandatory control）与这 13 项职责是不同计数；
新增候选不增加第 14 项职责，也不重签当前 readiness 或 stage。

当前内容侧所有权固定为：`content_router` 只输出 `M/T`、`m_lf/m_hf` 与
identity/digests；不输出攻击标签、`a/w` 或预算。
`content_embedder` 只按
`normalize(normalize(m_hf*T_hf)+normalize(m_lf*T_lf))` 形成 combined
direction，独占共同总预算、HF-only/LF-only/soft-routed/route-disabled delta
与 realized total norm/relative-L2 核验及零方向失败；不存在 `a/w` grid。
clean、HF-only、LF-only、soft-routed LF/HF、route-disabled 和显式失败是
current control/record surface。

历史 `routing_stqr` 的 observations、`A`、互补双 mask、disabled-uniform control，
以及旧组合的 `a`、`u_content(a)`、direction dot/product 与 routed/route-disabled
records 只服务 exact producer replay；不得重新成为 current method authority、
readiness 或 candidate-selection surface。
runtime 只在冻结 callback/model/dtype 边界物化 delta，并把实际张量及 combined
delta 的 total norm/relative L2 返回给 embedder 判定，不得改变预算或方向，也不得
声称可观测分支级实际写入量。

## 研究与运行层

| path | responsibility | allowed project dependencies |
| --- | --- | --- |
| `runtime/` | 生成模型、Q/K 观测、设备和 dtype 适配；物化 embedder delta 并返回实际张量及 realized combined total norm/relative L2。 | `main` public surface |
| `experiments/protocol/` | 内部设计验证、外部 comparison、records 和共享接口。 | none |
| `experiments/methods/` | 将项目方法与 baseline 薄适配到实验协议。 | `main`、`runtime`、`experiments.protocol` |
| `experiments/attacks/` | 与方法正交的攻击实现。 | `experiments.protocol` |
| `experiments/metrics/` | 与方法实现解耦的指标计算。 | `experiments.protocol` |
| `experiments/runners/` | 唯一实验组合层和 governed records 写入层。 | 上述研究与运行层 |

runtime 只提供执行能力，不决定 near-threshold、几何救援或最终水印判定。实验 methods 不得复制核心算法。

## 联合判定数据流

```text
待检图像 + 检测密钥
          ↓
同一内容检测器 ── 达阈值 ─────────────→ 内容阳性
          │
          └── 近阈值负样本
                    ↓
             Q/K 几何估计
                    ↓
          可靠才允许图像回正
                    ↓
       同一内容检测器 + 同一阈值
                    ↓
                 最终判定
```

几何链失败、不可靠或样本不在救援区间时，必须保留原内容判定，不得降阈值。

## 外围层

- `paper_artifacts/` 只从冻结 records 和 manifests 重建 tables、figures 和 reports。
- `notebooks/` 只保存薄编排入口。
- `infrastructure/` 保存环境、调度和远程执行入口。
- `models/` 是本地、非权威、不审计的模型资产/缓存根；checkpoint 与下载附属
  元数据不得进入 Git，也不得作为方法、readiness 或科学结论的权威证据。
- model/repository/name/revision、checkpoint blob SHA/size 与环境版本/设备是非权威
  selection/observation metadata；方法强身份只绑定候选职责和行为协议。Notebook 唯一
  Drive 模型输入为 regular non-symlink `MyDrive/CEG-WM/models/inspyrenet/ckpt_base.pth`，
  复制到 fresh `/content` 后仍须 `weights_only`、strict state_dict 与 public API fail closed。
- `.agents/`、`.codex/` 与 `governance/` 可整体删除；研究和交付代码不得导入它们。

## 协议与证据链

内部组件设计验证与外部 baseline 比较是两个不同协议表面：

- 内部设计验证用于 LF/HF 职责、错误密钥、路由、组合、几何估计和救援门控消融。
- 外部比较要求项目方法与已登记 baseline 共享样本、预算、攻击和指标条件。

```text
固定配置 + code/model revision + sample/split manifests
                            ↓
                    protocol preflight
                            ↓
                     experiment runner
                            ↓
              governed records + provenance
                            ↓
                     frozen manifest
                            ↓
                 tables / figures / reports
                            ↓
                    supported claims
```

结构、测试、harness 或研究定义通过不能替代真实 records。

## 扩展规则

新增顶级目录前必须登记到 `governance/policies/project_roots.yaml`。改变方法层依赖、正式检测器身份、阈值语义、盲检测边界、records 解释或证据链时，必须同步更新权威设计、policy、测试和必要的 decision record。

本轮阻断修订后通用治理平面冻结。只有真实 method、runtime、experiment 或 evidence 工作暴露具体可复现缺口时，才允许最小修改现有治理规则；不得继续以新增通用 policy、skill、schema 或 harness 代替研究推进。
