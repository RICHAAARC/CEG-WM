# Governance Self-Tests

此目录验证 policies、harness、命名、字段、Notebook、拆包、方法就绪与外层可拆卸性。它不进入根项目默认 pytest：

```bash
conda run -n CEG-WM python -m pytest -q -s -c governance/pytest.ini governance/tests
```

治理任务的完成入口是
`conda run -n CEG-WM python governance/tools/run_validation_profile.py governance`；
该入口显式收集 `governance/tests/`，随后运行完整 harness，但不会收集根项目方法
测试。部分拆包与可拆卸性合同测试会导入 `main`，所以缺少 PyTorch 的 `.venv` 只能
运行已知不触发这些合同路径的定向测试，不能完成正式治理档位。

测试必须轻量，不启动真实研究运行，也不作为论文证据。

当前入口包括 policy/harness、命名与字段、Notebook、方法就绪、拆包、项目复制和外层可拆卸性测试；实时列表以 `conda run -n CEG-WM python -m pytest --collect-only -q -s -c governance/pytest.ini governance/tests` 为准。
