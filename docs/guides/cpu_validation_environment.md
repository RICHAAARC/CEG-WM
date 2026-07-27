# CPU Validation Environment

CEG-WM 的完整 CPU 三门使用已登记、可复建的 `CEG-WM` Conda 环境。项目根目录
`.venv` 只用于不依赖 PyTorch 的轻量治理检查；缺少 `torch` 时不能运行会收集默认
方法节点的根 pytest 命令。两套环境都只服务
`method_construction_authorized` 阶段的 CPU 验证，不包含模型权重、GPU runtime
或正式实验依赖。

## Create

从项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --requirement requirements_cpu.txt
```

当前依赖清单固定 pytest 版本；`.venv` 的轻量治理依赖变化必须更新
`requirements_cpu.txt`。PyTorch 方法依赖变化必须更新下述已登记 Conda YAML；
两类变化都不得只在本机环境中临时安装。

批次 4 方法验证所需 PyTorch CPU 依赖固定在
`configs/environments/ceg_wm_cpu.yaml`，从项目根目录执行：

```bash
conda env create --file configs/environments/ceg_wm_cpu.yaml
```

不得以本机临时安装替代该 YAML。`requirements_cpu.txt` 仍是 `.venv` 的版本权威；
上述 YAML 是获授权 CPU 方法验证环境的版本权威。

## Validate

不要求激活环境，完整三门直接使用已登记 Conda 环境：

```bash
conda run -n CEG-WM python -m pytest -q -s
conda run -n CEG-WM python -m pytest -q -s -c governance/pytest.ini
conda run -n CEG-WM python governance/harness/run_all_audits.py
```

`.venv` 可以运行明确不收集方法测试的轻量治理入口，例如：

```bash
.venv/bin/python -m pytest -q -s -c governance/pytest.ini
.venv/bin/python governance/harness/run_all_audits.py
```

`-s` 用于规避部分 WSL/Windows 临时目录上的 pytest capture 文件异常；它不改变测试选择或断言语义。

## Boundary

- `.venv/` 是未提交的 `local_environment`，不进入治理扫描、拆包或实验 provenance。
- `requirements_cpu.txt` 是轻量 `.venv` CPU 治理验证依赖的版本权威。
- `configs/environments/ceg_wm_cpu.yaml` 是获授权 CPU 方法验证依赖的版本权威；
  它不是 runtime 或 GPU 依赖锁。
- 真实 runtime、GPU 或正式实验所需依赖必须在相应阶段另行设计和固定。
- 环境存在、pytest 通过或 harness 通过不能证明水印机制有效。
