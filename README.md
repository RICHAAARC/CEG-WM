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

- CEG-WM 已按自有 `HF carrier`、`HF direct score` 和 `content detector` 身份完成
  CPU/synthetic 实现；这不是 runtime 或效果验证。
- LF 的职责、载体、路由和 LF/HF 组合候选已有实现，但仍需实验晋升，不预设正式权重。
- 方法完成面固定为 13 项职责组件；组合写入、LF 盲分数和几何可靠性各有独立
  路径。候选 registry 仍是 10 个 ID，两个计数不得混用。
- 几何链只允许恢复坐标与图像，不得直接给出水印阳性。
- 检测不得依赖原始参考图、嵌入端 record 或嵌入端私有统计量。
- 恢复后不得切换到更宽松的救援分类器或阈值。

详细研究定义见 [docs/design/](docs/design/README.md)。

## 当前阶段

- `project_stage`: `method_construction_authorized`
- `implementation_status`: `not_implemented`
- 13 项真实职责、27 个方法特异性 CPU/synthetic 节点和唯一 readiness 已在后续
  revisions 完成并经独立语义审计；等待独立阶段迁移，不能由 readiness 自动改阶段。
- 正式 detector 仍为 HF-only；LF/routing 尚未实验晋升，
  `full_ceg_wm_eligible=false`。
- 当前没有真实 runtime/GPU qualification、完整联合 FPR、攻击矩阵、正式实验
  records 或论文效果证据。

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
| `docs/` | 研究定义、方法设计、决策、协议与证据说明。 |
| `governance/` | 可拆卸的外层契约、policy、harness 和控制平面自测。 |
| `.agents/skills/` | 随项目维护的 Codex 工作流。 |
| `.codex/` | 可拆卸的项目阶段和构建期门禁元数据。 |

研究代码和交付代码不得依赖 `.agents/`、`.codex/` 或 `governance/`。

## 证据边界

研究定义、代码、测试或 harness 通过都不能证明水印有效。正式结论必须来自独立 calibration/evaluation 切分下的 governed records、冻结 manifests 和可重建 artifacts。

## 治理验证

完整方法门禁使用登记的 `CEG-WM` CPU Conda 环境：

```bash
conda env create --file configs/environments/ceg_wm_cpu.yaml
```

完整验证：

```bash
conda run -n CEG-WM python -m pytest -q -s
conda run -n CEG-WM python -m pytest -q -s -c governance/pytest.ini
conda run -n CEG-WM python governance/harness/run_all_audits.py
```

`.venv` 仅支持不依赖 PyTorch 的轻量治理检查；缺少 `torch` 时不得用 `.venv`
执行会收集默认方法测试的根 pytest 命令。

环境边界见 [CPU validation guide](docs/guides/cpu_validation_environment.md)。
