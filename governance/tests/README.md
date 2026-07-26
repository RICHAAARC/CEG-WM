# Governance Self-Tests

此目录验证 policies、harness、命名、字段、Notebook、拆包、方法就绪与外层可拆卸性。它不进入根项目默认 pytest：

```bash
.venv/bin/python -m pytest -q -s -c governance/pytest.ini
```

测试必须轻量，不启动真实研究运行，也不作为论文证据。

当前入口包括 policy/harness、命名与字段、Notebook、方法就绪、拆包、项目复制和外层可拆卸性测试；实时列表以 `.venv/bin/python -m pytest --collect-only -q -s -c governance/pytest.ini` 为准。
