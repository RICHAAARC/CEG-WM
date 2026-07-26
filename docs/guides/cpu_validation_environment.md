# CPU Validation Environment

CEG-WM 使用项目根目录下的 `.venv` 执行 pytest、governance 自测和 harness。该环境
当前只服务 `method_construction_authorized / not_implemented` 状态的轻量 CPU
治理验证，不包含 Torch、模型权重、GPU runtime 或实验依赖。

## Create

从项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --requirement requirements_cpu.txt
```

当前依赖清单固定 pytest 版本；新增 CPU 测试依赖时必须先更新 `requirements_cpu.txt`，不得只在本机环境中临时安装。

## Validate

不要求激活环境，直接使用项目解释器：

```bash
.venv/bin/python -m pytest -q -s
.venv/bin/python -m pytest -q -s -c governance/pytest.ini
.venv/bin/python governance/harness/run_all_audits.py
```

`-s` 用于规避部分 WSL/Windows 临时目录上的 pytest capture 文件异常；它不改变测试选择或断言语义。

## Boundary

- `.venv/` 是未提交的 `local_environment`，不进入治理扫描、拆包或实验 provenance。
- `requirements_cpu.txt` 是当前 CPU 验证依赖的版本权威。
- 方法、runtime 或正式实验所需依赖必须在相应阶段另行设计和固定。
- 环境存在、pytest 通过或 harness 通过不能证明水印机制有效。
