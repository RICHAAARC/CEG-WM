# CEG-WM 分层治理架构契约

## 项目定位

CEG-WM 是内容证据主判、几何条件恢复的双链生成式图像水印研究项目。当前已合法
进入 `method_construction_authorized`；随后独立 revisions 已完成固定 13 项职责、
27 个 CPU/synthetic 方法行为节点和唯一 readiness，并经独立语义审计。

阶段实施路径固定为 `research_defined → method_construction_authorized → method_implemented`。
实际阶段/status 仍为 `method_construction_authorized / not_implemented`，等待独立
阶段迁移。CPU/synthetic readiness 不证明 runtime、GPU、正式 FPR 或科学效果；
正式 detector 仍为 HF-only，LF/routing 未实验晋升，
`full_ceg_wm_eligible=false`。

`method_implemented` 的 readiness AST 审计只承担候选绑定、固定模块路径、symbol 调用和断言接线检查；它不能单独证明非代理实现。该阶段还必须有绑定候选规格摘要、实现路径、方法特异性测试节点和 repository revision 的独立语义复核 `approve`，且复核后这些受保护路径没有变化。

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
10 个 ID 与这 13 项职责是不同计数。

内容侧所有权固定为：`content_router` 只输出 observations、`A`、两 mask、
route identity/digests 和 disabled uniform control；LF/HF carrier 只输出模板和
masked unit direction；`content_embedder` 独占冻结 `a`、方向组合、共同总预算、
HF-only/LF-only/combined delta、方向内积/组合归一因子、target total 与 realized
combined total norm/relative L2 核验及零方向失败。mixing coefficients 不是可加
方向份额。
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
