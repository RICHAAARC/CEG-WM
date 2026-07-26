# Test Inventory

## 实时权威列表

研究测试节点会随方法和实验实现增长，不维护逐函数的第二份静态真相。使用以下命令获得项目测试列表：

```bash
.venv/bin/python -m pytest --collect-only -q -s
```

## 当前测试入口

| test_path | level | default_run | responsibility |
| --- | --- | --- | --- |
| `tests/unit/test_hf_content_backbone.py` | `unit` | yes | HF sparse-tail 载体、HF-only 共同总预算、盲 direct score、wrong-key 与当前 HF-only content detector。 |
| `tests/unit/test_key_schedule.py` | `unit` | yes | 冻结 key schedule 的 root/domain、counter/quantile golden、wrong/public 派生、不可变身份与失败边界。 |
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
