# CPU Validation Environment

CEG-WM 的三个正式验证档位都使用已登记、可复建的 `CEG-WM` Conda 环境。项目根目录
`.venv` 只用于明确不导入研究代码的定向轻量检查；治理合同测试可能通过拆包或
可拆卸性验证导入 `main`，因此缺少 `torch` 的 `.venv` 不能完成 `governance`、
`method`、`full` 或根 pytest 命令。两套环境都只服务
方法构建/readiness 的 CPU 验证，并继续适用于 `runtime_verified` 的 CPU
验证面；不包含模型权重、GPU runtime 或正式实验依赖。

## Create

从项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --requirement requirements_cpu.txt
```

当前依赖清单固定 pytest 版本；`.venv` 的轻量检查依赖变化必须更新
`requirements_cpu.txt`。正式 profile 的 PyTorch 依赖变化必须更新下述已登记 Conda YAML；
两类变化都不得只在本机环境中临时安装。

批次 4 方法验证所需 PyTorch CPU 依赖固定在
`configs/environments/ceg_wm_cpu.yaml`，从项目根目录执行：

```bash
conda env create --file configs/environments/ceg_wm_cpu.yaml
```

不得以本机临时安装替代该 YAML。`requirements_cpu.txt` 仍是 `.venv` 的版本权威；
上述 YAML 是三个正式 CPU profile 的环境版本权威。

正式环境同时固定 `pytest=9.1.1` 与 `pytest-xdist=3.8.0`。`xdist` 仅用于下述登记的
文件级并行分片；不得用 `-n auto` 或把完整目录直接交给多个 worker。

## Validate

先运行最小受影响测试，再按任务范围选择一个档位：

| profile | pytest boundary | environment |
| --- | --- | --- |
| `governance` | governance 固定并行文件 → governance 串行补集 → harness。 | `CEG-WM` Conda |
| `method` | project 固定并行文件 → project 串行补集 → harness。 | `CEG-WM` Conda |
| `full` | project 并行/串行 → governance 并行/串行 → harness。 | `CEG-WM` Conda |

```bash
conda run -n CEG-WM python governance/tools/run_validation_profile.py governance
conda run -n CEG-WM python governance/tools/run_validation_profile.py method
conda run -n CEG-WM python governance/tools/run_validation_profile.py full
```

普通治理、skill 或非研究语义文档变更使用 `governance`；只影响研究代码/测试的变更
使用 `method`；阶段、research state、登记设计、候选/readiness、pytest 选择、验证
规则和跨层修改使用 `full`。不能确定时使用 `full`。

`-s` 用于规避部分 WSL/Windows 临时目录上的 pytest capture 文件异常；它不改变测试选择或断言语义。

## Registered scheduling

`governance/tools/run_validation_profile.py` 是分片执行权威。project 的 28 个已审查文件和
governance 的 6 个已审查文件使用固定 4 workers 与 `--dist=loadfile`。每个目录随后以
完整目录作为 pytest 收集根，并用逐路径 `--ignore` 排除已经执行的并行文件，从而形成
无重复的串行补集。新增测试文件不会被并行白名单自动吸收，而会自然进入串行补集；这既
保持 fail-closed 收集，也避免未来文件被未经审核地并行化。精确正向白名单登记在
`docs/reference/test_case_registry.md`。

所有 pytest 分片均带 `-x`。`full` 的固定顺序为：

1. project 固定文件并行分片；
2. project 目录串行补集；
3. governance 固定文件并行分片；
4. governance 目录串行补集；
5. 单进程 harness。

首个非零 return code 立即作为 profile return code 返回；后续分片和 harness 不再启动。
stdout/stderr 直接继承调用进程，以便保留真实失败输出。每个子进程都强制使用
`TMPDIR=/tmp`、`TEMP=/tmp`、`TMP=/tmp`、`OMP_NUM_THREADS=1` 和
`MKL_NUM_THREADS=1`。不得自动回退到串行重跑，不得使用 `-n auto`，不得同时运行两个
pytest suite，也不得并行运行 harness。

## Boundary

- `.venv/` 是未提交的 `local_environment`，不进入治理扫描、拆包或实验 provenance。
- `requirements_cpu.txt` 是 `.venv` 定向轻量检查依赖的版本权威，不保证满足正式
  `governance` profile。
- `configs/environments/ceg_wm_cpu.yaml` 是 `governance`、`method`、`full` CPU
  验证依赖的版本权威；
  它不是 runtime 或 GPU 依赖锁。
- 真实 runtime、GPU 或正式实验所需依赖必须在相应阶段另行设计和固定。
- 环境存在、pytest 通过或 harness 通过不能证明水印机制有效。
