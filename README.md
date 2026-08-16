# CEG-WM

CEG-WM 是一个双链生成式图像水印研究项目。项目以内容证据为主判，以几何链为条件恢复机制；几何证据不得直接产生水印阳性。

## 方法路线

```text
内容链
├── 内容自适应路由
├── LF / HF 载体
├── LF / HF 组合嵌入
├── 独立 LF / HF 盲分数
└── 内容统计组合与正式 content detector

几何链
├── Q/K 几何同步
├── crop / scale / rotation 估计
├── 独立几何可靠性合取门
└── 几何恢复与图像回正

联合判定
├── soft-route max content detector 原图检测
├── 近阈值判断
├── 必要时启动几何恢复
└── 回正后使用同一检测器与同一阈值重判
```

当前冻结的是方法职责和判定边界，不是方法效果：

- CEG-WM 已按自有 `HF carrier`、`HF direct score` 和 `content detector` 身份完成
  CPU/synthetic 实现；这不是 runtime 或效果验证。
- 当前 readiness 在 13 项职责上绑定 12 个唯一候选身份和 17 个行为节点。语义—纹理
  软路由五候选由 producer `02fe5dcc2b74482c9eb1e0b192b4a2ce79e0d9eb`
  实现并经独立 exact audit 批准，状态为
  `implemented_not_scientifically_validated`；旧 routing/combination 已形成
  producer-bound development negative，只保留历史精确重放。
- 方法完成面固定为 13 项职责组件；组合写入、LF 盲分数和几何可靠性各有独立
  路径。设计 registry 为 20 个 ID（19 个具名候选加 1 个 mandatory control），
  readiness-bound 当前候选身份12个，三种计数不得混用。旧的11候选/28节点
  readiness 快照仍由 exact revision
  `0258ccb2100bfe8b58d1a12079876841192528b3` 保存为历史事实。
- 方法设计固定使用 InSPyReNet soft `M` + Sobel/P95 `T` 的逐图软路由、无 `a/w`
  的 `normalize(normalize(m_hf*T_hf)+normalize(m_lf*T_lf))` 写入和
  `max(z_hf_soft,z_lf_soft)` 检测统计，不存在权重、函数或攻击条件 selection。
  calibration 只在独立职责中拟合分支 primary-null/CDF 标准化、max statistic 的
  `tau` 及 rescue/geometry/end-to-end 各自声明的量；历史 `a/w/function` 只按原
  producer replay。
- 几何链只允许恢复坐标与图像，不得直接给出水印阳性。
- 检测不得依赖原始参考图、嵌入端 record 或嵌入端私有统计量。
- 恢复后不得切换到更宽松的救援分类器或阈值。

详细方法定义见 [docs/design/](docs/design/README.md)；采用/失败/待验证状态见
[docs/project_state/](docs/project_state/README.md)。

## 当前阶段

- `project_stage`: `experiment_ready`
- `implementation_status`: `implemented`
- 13 项真实职责、12 个唯一候选身份、17 个方法特异性 CPU/synthetic 节点和唯一 readiness 已在后续
  revisions 完成并经独立语义审计；独立阶段迁移已经完成，但不是由 readiness 自动改阶段。
- `experiment_ready_infrastructure_closure` 明确包含七项职责：研究范围与阶段治理、
  方法架构与证据边界、内容/几何/联合判定设计、算法与候选冻结、真实方法实现、
  runtime qualification，以及实验协议与可追溯交付入口。该闭环已完成；这只表示实验执行准备就绪，
  不授权 calibration、hf_only_reference_validation 晋升、GPU 高成本运行或正式实验。
- 正式 detector 仍为 HF-only；语义—纹理 soft-route 五候选为
  `implemented_not_scientifically_validated`，soft max 仅作 diagnostic 且尚未实验
  晋升，没有 calibration、固定 FPR、GPU 或科学效果证据。hard salient-object
  local-LF 四候选只作为 `superseded_without_scientific_adjudication` 历史候选保留；
  `full_ceg_wm_eligible=false`。
- 真实 SD3.5 runtime 边界已由 candidate
  `8b2344756c4c247906ff0d4eab68e46a773e13f5` 的 `qualification / passed` run
  `20260729T110628Z` 验证；result ZIP SHA-256 为
  `d9b7d91d41cc963098c077268445ad80e9994c809227ca2f68615a37ac93ac37`。
  这不表示 LF/routing/组合、几何恢复效果、完整联合 FPR、攻击鲁棒性、正式实验
  records 或论文效果已经成立。

`SLM-WM`、`SLM-WM-FlowHF`、`CEG-WM-OLD-main` 和 `CEG-O-master` 仅作为历史来源与偏离案例；它们的固定 LF/HF 融合、reference-based 几何、嵌入端私有状态依赖和系统级 attestation 路线不自动成为本项目方法。

## 分层结构

| path | responsibility |
| --- | --- |
| `main/` | 可独立抽离的 CEG-WM 核心方法包；当前包含已审计的 CPU/synthetic 方法实现。 |
| `runtime/` | 生成模型、Q/K 观测和检测执行适配；不得拥有联合判定语义。 |
| `experiments/protocol/` | 内部设计验证、外部比较和 governed records 的共享契约。 |
| `experiments/methods/` | 项目方法和外部 baseline 的薄实验适配。 |
| `experiments/attacks/` | 与方法正交的攻击变换。 |
| `experiments/metrics/` | 内容检测、几何估计、图像质量和资源指标。 |
| `experiments/runners/` | 唯一实验组合层和 governed records 写入层。 |
| `paper_artifacts/` | 从冻结 records 与 manifests 重建论文产物。 |
| `models/` | 本地、非权威、不审计的模型资产/缓存；checkpoint 不进入 Git。 |
| `docs/` | 研究定义、方法设计、决策、协议与证据说明。 |
| `governance/` | 可拆卸的外层契约、policy、harness 和控制平面自测。 |
| `.agents/skills/` | 随项目维护的 Codex 工作流。 |
| `.codex/` | 可拆卸的项目阶段和构建期门禁元数据。 |

研究代码和交付代码不得依赖 `.agents/`、`.codex/` 或 `governance/`。

## 证据边界

研究定义、代码、测试或 harness 通过都不能证明水印有效。正式结论必须来自独立 calibration/evaluation 切分下的 governed records、冻结 manifests 和可重建 artifacts。

## 治理验证

方法和完整门禁使用登记的 `CEG-WM` CPU Conda 环境：

```bash
conda env create --file configs/environments/ceg_wm_cpu.yaml
```

按变更范围选择验证档位：

```bash
conda run -n CEG-WM python governance/tools/run_validation_profile.py governance
conda run -n CEG-WM python governance/tools/run_validation_profile.py method
conda run -n CEG-WM python governance/tools/run_validation_profile.py full
```

先运行最小受影响测试；治理任务不再无条件收集方法测试。阶段、登记设计、pytest
选择和跨治理/研究层变更仍使用 `full`。治理合同测试可能导入研究代码，因此三个
正式 profile 都使用 Conda；`.venv` 只运行已知不导入研究代码的定向轻量检查。

环境边界见 [CPU validation guide](docs/guides/cpu_validation_environment.md)。
