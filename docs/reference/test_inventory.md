# Test Inventory

## 实时权威列表

研究测试节点会随方法和实验实现增长，不维护逐函数的第二份静态真相。使用以下命令获得项目测试列表：

```bash
conda run -n CEG-WM python -m pytest --collect-only -q -s
```

默认方法节点导入 PyTorch；缺少 `torch` 的 `.venv` 不得用于该根级收集命令。

## 当前测试入口

| test_path | level | default_run | responsibility |
| --- | --- | --- | --- |
| `tests/unit/test_runtime_configuration_and_adapter.py` | `unit` | yes | 冻结 SD3.5 runtime 配置解析、稳定摘要、设备选择、mock backend 身份核验和 fail-closed 初始化控制流；不加载模型、不访问网络或 GPU。 |
| `tests/unit/test_runtime_content_write_and_vae.py` | `unit` | yes | CPU tensor/fake backend 覆盖 clean/watermarked 同基础 latent、callback index 序列、float16 deterministic replay、actual delta/realized 测量、写入消失/非有限/overflow 失败以及 VAE decode/posterior-mode encode；不判定预算合格，不加载模型、不访问网络或 GPU。 |
| `tests/unit/test_geometry_chain.py` | `unit`, `quick` | yes | 真实 Q/K 数值关系、geometry-key 投影、actual-dtype 同步回溯、冻结 similarity 搜索、8 个 wrong-key、独立 reliability fail-closed 与 PyTorch 回正坐标协议；完整搜索节点 CPU 成本较高但仍是默认 synthetic 方法验证，不属于 real-model、`slow` 或 `formal`。 |
| `tests/unit/test_lf_routing_combination.py` | `unit` | yes | LF carrier/blind detector、从实际 S/T/R/Q 观测重演公式的 routing、同 route/mask 绑定的 LF/HF embedder、同普通图像观测约束与未晋升 C0/C1/C2 组合诊断。 |
| `tests/unit/test_hf_content_backbone.py` | `unit` | yes | HF sparse-tail 载体、HF-only 共同总预算、盲 direct score、wrong-key 与当前 HF-only content detector。 |
| `tests/unit/test_key_schedule.py` | `unit` | yes | 冻结 key schedule 的 root/domain、counter/quantile golden、wrong/public 派生、不可变身份与失败边界。 |
| `tests/unit/test_joint_decision.py` | `unit` | yes | 近阈值门控、几何不直接阳性、同 detector/key/preprocess/threshold 回正重判和 raw/rescue 阳性路径。 |
| `tests/unit/test_comparison_preflight.py` | `unit` | yes | 公平对比协议、切分隔离与 baseline 完整性。 |
| `tests/unit/test_internal_scientific_validation_protocol.py` | `unit` | yes | 内部 unit/case/source-cluster identity、八 split 无泄漏、伪造 grant 与 held-out evaluation fail-closed、四类执行状态及 resource/execution/scientific failure class、精确 protocol/manifest dataclass trust anchors（拒绝 duck object/subclass）、全部 canonical 协议语义、实际 digest 与 assignment 绑定、所有非初始 outcome 的 retry lineage/stop 集合状态机、协议版本绑定、13 职责验证矩阵、同 detector/config/threshold 和内容证据阳性/阴性约束；仅 CPU schema/constraint，不宣称科学有效。 |
| `tests/unit/test_internal_experiment_components.py` | `unit` | yes | A-2 顶层 `main` re-export 与 subpackage public object identity、adapter 顶层依赖 AST、真实 `main` 小张量内容链、wrong-key 两步及四种 key public-callable provenance、Q/K 同步与 transform estimator、真实 reliability/rectifier/conditional recovery 适配调用及同 detector/threshold 约束、13 个 canonical binding 三元组及 callable/identity 伪造负例、冻结组件 registry、确定性 identity/crop/scale/rotation/组合攻击、fixed-FPR threshold 自校验与构造后篡改负例、逐 case/聚合 fixed-FPR、wrong-key、质量、routing、LF/HF、几何、可靠性、回正与 rescue 指标、全部聚合的同一合法 split 约束及 rescue 真实轨迹，以及 held-out/非有限/身份漂移 fail-closed；不运行真实模型、GPU、runner 或科学验证。 |
| `tests/unit/test_internal_governed_runner.py` | `quick` | yes | 冻结 input manifest 到真实 adapter、attack、joint decision 和 metric-case replay 的 CPU/synthetic 编排；唯一 writer 的包内 executable field registry/schema/provenance 校验、canonical 快照/原子写入、确定性 ID/sequence/attempt/parent、完成态幂等 resume、资源 retry lineage、执行/科学失败终止、routing candidate 摘要，以及逐 unit artifact/attack/metric、routing、key/control、detector/config/preprocess、threshold/tau、几何操作/experiments reliability config 与冻结期望一致性；覆盖构造后 context/registry、content/geometry callable、binding、threshold、reliability、artifact 像素/digest、attack 参数/digest 漂移在 attack/method/write 前拒绝，也覆盖 canonical 但伪造的 completed record 与同步改写 retry lineage 在 load/replay/resume 前拒绝；同时覆盖预登记排除、held-out evaluation、冲突/partial-write fail-closed；不产生科学 records 或效果结论。 |
| `tests/functional/test_governed_artifact_structures.py` | `quick` | yes | records provenance 与 artifact manifest 结构。 |

## 后续测试等级

| level | default_run | intended_use |
| --- | --- | --- |
| `unit` | yes | 纯函数、核心算法局部行为与 schema。 |
| `constraint` | yes | 架构、命名、字段与文件组织约束。 |
| `quick` | yes | 小型合成输入上的轻量跨函数行为。 |
| `integration` | no | 跨层或真实组件集成。 |
| `smoke` | no | 真实 backend 的关键执行路径。 |
| `slow` | no | 耗时但不承担正式证据口径的检查。 |
| `formal` | no | 冻结协议下的正式门禁和发布证据测试。 |

新增或移动研究测试时，应同步本页的“当前测试入口”；项目 pytest 规则以 `pyproject.toml` 为准。

本清单只说明测试入口和执行成本，不把测试通过解释为方法或实验效果证据。

## 几何 CPU/synthetic 环境

方法默认测试需要真实 CPU PyTorch 与 NumPy，使用项目登记的 Conda 环境。只修改
研究代码/测试时使用 `method`，跨层、阶段、登记设计或测试选择变化使用 `full`：

```bash
conda run -n CEG-WM python governance/tools/run_validation_profile.py method
conda run -n CEG-WM python governance/tools/run_validation_profile.py full
```

纯治理或非研究语义文档任务使用
`conda run -n CEG-WM python governance/tools/run_validation_profile.py governance`，
不收集上述方法节点。治理合同测试可能导入 `main`，所以正式治理档位不能使用缺少
PyTorch 的 `.venv`。

该环境只证明 CPU/synthetic 方法行为，不替代冻结 SD3.5 revision、真实登记层 Q/K
捕获或 GPU/runtime qualification。
