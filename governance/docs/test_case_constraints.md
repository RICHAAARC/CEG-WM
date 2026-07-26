# Test Case Governance

## 项目测试

| path | marker | default_run | responsibility |
| --- | --- | --- | --- |
| `tests/unit/` | `unit` | yes | 方法、协议和纯函数行为。 |
| `tests/functional/` | `quick` | yes | 小型合成输入上的研究功能。 |
| `tests/integration/` | `integration` | no | 跨层或真实组件集成。 |
| `tests/smoke/` | `smoke` | no | 真实 backend 的关键路径。 |
| `tests/formal/` | `formal` | no | 冻结协议下的正式门禁。 |
| `tests/helpers/` | none | not applicable | 测试辅助模块。 |
| `tests/fixtures/` | none | not applicable | 小型测试数据。 |

根 `python -m pytest -q` 只运行项目测试，不收集 `governance/tests/`。

## 外层治理自测

`governance/tests/` 只验证 policy、harness、拆包和可拆卸性，使用：

```bash
python -m pytest -q -c governance/pytest.ini
```

两套测试都必须轻量；integration、smoke、slow 和 formal 默认排除。测试输出只写临时目录，测试通过不能替代真实实验 records。
