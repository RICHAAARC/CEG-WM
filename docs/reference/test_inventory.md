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
| `tests/unit/test_geometry_chain.py` | `unit`, `quick` | yes | 真实 Q/K 数值关系、geometry-key 投影、actual-dtype 同步回溯、冻结 similarity 搜索、8 个 wrong-key、独立 reliability fail-closed 与 PyTorch 回正坐标协议；完整搜索节点 CPU 成本较高但仍是默认 synthetic 方法验证，不属于 real-model、`slow` 或 `formal`。 |
| `tests/unit/test_lf_routing_combination.py` | `unit` | yes | LF carrier/blind detector、从实际 S/T/R/Q 观测重演公式的 routing、同 route/mask 绑定的 LF/HF embedder、同普通图像观测约束与未晋升 C0/C1/C2 组合诊断。 |
| `tests/unit/test_hf_content_backbone.py` | `unit` | yes | HF sparse-tail 载体、HF-only 共同总预算、盲 direct score、wrong-key 与当前 HF-only content detector。 |
| `tests/unit/test_key_schedule.py` | `unit` | yes | 冻结 key schedule 的 root/domain、counter/quantile golden、wrong/public 派生、不可变身份与失败边界。 |
| `tests/unit/test_joint_decision.py` | `unit` | yes | 近阈值门控、几何不直接阳性、同 detector/key/preprocess/threshold 回正重判和 raw/rescue 阳性路径。 |
| `tests/unit/test_comparison_preflight.py` | `unit` | yes | 公平对比协议、切分隔离与 baseline 完整性。 |
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
