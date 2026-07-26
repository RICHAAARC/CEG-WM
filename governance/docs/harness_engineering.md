# Harness Engineering Guide

Harness 是可整体拆卸的外层护栏，只把合同和机器 policy 转化为结构化检查。它不得成为方法、runtime、实验或 artifact rebuild 的依赖。

```bash
python -m pytest -q -c governance/pytest.ini
python governance/harness/run_all_audits.py
```

新增检查必须对应已知偏移风险、读取已登记 policy、限制扫描范围并返回统一报告。Harness 通过不能作为研究证据。
