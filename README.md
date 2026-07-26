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
├── HF direct score 与 content detector 原图检测
├── 近阈值判断
├── 必要时启动几何恢复
└── 回正后使用同一检测器与同一阈值重判
```

当前冻结的是方法职责和判定边界，不是方法效果：

- CEG-WM 以自有 `HF carrier`、`HF direct score` 和 `content detector` 命名当前职责；首个 HF 候选尚未实现或验证。
- LF 的职责、载体、路由和 LF/HF 组合规则仍需在本项目内独立设计验证，不预设固定权重。
- 未来方法实现固定为 13 项职责组件；组合写入、LF 盲分数和几何可靠性各有独立
  路径。候选 registry 仍是 10 个 ID，两个计数不得混用。
- 几何链只允许恢复坐标与图像，不得直接给出水印阳性。
- 检测不得依赖原始参考图、嵌入端 record 或嵌入端私有统计量。
- 恢复后不得切换到更宽松的救援分类器或阈值。

详细研究定义见 [docs/design/](docs/design/README.md)。

## 当前阶段

- `project_stage`: `research_defined`
- 当前已经建立项目研究定义、双链架构边界、威胁模型和验证问题。
- 当前没有 LF、HF、几何链或联合判定的项目实现。
- 当前没有 runtime、攻击实现、正式实验 records 或论文效果证据。

`SLM-WM`、`SLM-WM-FlowHF`、`CEG-WM-OLD-main` 和 `CEG-O-master` 仅作为历史来源与偏离案例；它们的固定 LF/HF 融合、reference-based 几何、嵌入端私有状态依赖和系统级 attestation 路线不自动成为本项目方法。

## 分层结构

| path | responsibility |
| --- | --- |
| `main/` | 可独立抽离的 CEG-WM 核心方法包；当前尚无方法实现。 |
| `runtime/` | 生成模型、Q/K 观测和检测执行适配；不得拥有联合判定语义。 |
| `experiments/protocol/` | 内部设计验证、外部比较和 governed records 的共享契约。 |
| `experiments/methods/` | 项目方法和外部 baseline 的薄实验适配。 |
| `experiments/attacks/` | 与方法正交的攻击变换。 |
| `experiments/metrics/` | 内容检测、几何估计、图像质量和资源指标。 |
| `experiments/runners/` | 唯一实验组合层和 governed records 写入层。 |
| `paper_artifacts/` | 从冻结 records 与 manifests 重建论文产物。 |
| `docs/` | 研究定义、方法设计、决策、协议与证据说明。 |
| `governance/` | 可拆卸的外层契约、policy、harness 和控制平面自测。 |
| `.agents/skills/` | 随项目维护的 Codex 工作流。 |
| `.codex/` | 可拆卸的项目阶段和构建期门禁元数据。 |

研究代码和交付代码不得依赖 `.agents/`、`.codex/` 或 `governance/`。

## 证据边界

研究定义、代码、测试或 harness 通过都不能证明水印有效。正式结论必须来自独立 calibration/evaluation 切分下的 governed records、冻结 manifests 和可重建 artifacts。

## 治理验证

项目使用根目录下的 `.venv`。首次创建：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --requirement requirements_cpu.txt
```

验证：

```bash
.venv/bin/python -m pytest -q -s
.venv/bin/python -m pytest -q -s -c governance/pytest.ini
.venv/bin/python governance/harness/run_all_audits.py
```

环境边界见 [CPU validation guide](docs/guides/cpu_validation_environment.md)。
