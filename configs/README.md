# Governed Configurations

此目录保存研究与运行边界上的受治理配置。配置负责声明参数和协议，不得承载方法实现、实验结果或凭据。

## 当前目录

| path | responsibility | current_state |
| --- | --- | --- |
| `baselines/` | 外部 baseline 的固定 revision、参数和适配配置。 | 仅有边界说明。 |
| `environments/` | 获授权 CPU/synthetic 方法验证环境的可复建依赖记录。 | 批次 4 CPU 环境已登记。 |
| `experiments/` | 实验矩阵与公平对比协议实例。 | 仅有边界说明。 |

CEG-WM 在对应实现和协议出现后按需增加 `methods/`、`runtime/` 和 `artifacts/`。本目录只保存研究或运行实际消费的配置，不保存构建期检查元数据。

所有持久化研究字段应登记到 `docs/reference/field_registry.md`。密钥、token、私有数据路径和本机绝对路径不得提交。

## Method CPU/synthetic environment

批次 4 几何方法测试使用独立 CPU Conda 环境，按已登记 YAML 创建：

```bash
conda env create --file configs/environments/ceg_wm_cpu.yaml
```

`configs/environments/ceg_wm_cpu.yaml` 是这套获授权 CPU 方法验证环境的版本权威。
该依赖记录只服务默认 CPU/synthetic 方法测试及同一 checkout 的治理门禁。它不是
`runtime_sd35_flowmatch` 的真实模型依赖锁，也不支持 GPU/runtime 已验证声明。
