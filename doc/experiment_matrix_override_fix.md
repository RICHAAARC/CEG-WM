# experiment_matrix 参数注入修复总结

## 问题诊断

### 根本原因
experiment_matrix 在为 calibrate 和 evaluate 阶段生成子命令时，注入了 CLI override 参数 `calibrate.detect_records_glob` 和 `evaluate.detect_records_glob`，但这些参数没有在以下任何地方注册：
1. runtime_whitelist.yaml 的 allowed_overrides
2. 配置文件（default.yaml / paper_full_cuda.yaml）中的字段定义
3. arg_name_enum 枚举

导致错误：`override_not_allowed: key=calibrate.detect_records_glob`

## 解决方案对比

| 问题版本 | 错误类型 | 修复方案 |
|---------|--------|--------|
| v1 (01:19) | `override_not_allowed: key=seed` | 在 runtime_whitelist 中添加 seed/model_id |
| v2 (01:26) | YAML 语法错误 | 修复行 189-192 的缩进 |
| v3 (01:34) | `override_field_path_missing: field_path=seed` | 在配置文件中添加 `seed: null` |
| v4 (01:43) | `calibration.detect_records_glob is required` | 在 experiment_matrix.py 中注入参数 |
| **v5 (本次修复)** | **所有 8 个实验失败** | **完整的三层验证系统** |

## 修改详情

### 1. runtime_whitelist.yaml
添加三个新的 override 定义到 `override.allowed_overrides`：
```yaml
- arg_name: "calibrate_detect_records_glob"
  field_path: "calibrate.detect_records_glob"
  override_mode: "set"
  source: "cli"
  description: "实验矩阵：允许设置 calibrate 阶段的 detect 记录 glob 路径"

- arg_name: "evaluate_detect_records_glob"
  field_path: "evaluate.detect_records_glob"
  override_mode: "set"
  source: "cli"
  description: "实验矩阵：允许设置 evaluate 阶段的 detect 记录 glob 路径"

- arg_name: "evaluate_thresholds_path"
  field_path: "evaluate.thresholds_path"
  override_mode: "set"
  source: "cli"
  description: "实验矩阵：允许设置 evaluate 阶段的阈值工件路径"
```

并更新 `arg_name_enum.allowed`：
```yaml
- "calibrate_detect_records_glob"
- "evaluate_detect_records_glob"
- "evaluate_thresholds_path"
```

### 2. default.yaml 和 paper_full_cuda.yaml
添加新的配置块（在 impl 部分之前）：
```yaml
calibrate:
  detect_records_glob: null

evaluate:
  detect_records_glob: null
  thresholds_path: null
```

### 3. experiment_matrix.py
修改 `_run_experiment_grid_item()` 函数中的参数构建逻辑：

```python
# 使用 arg_name 而不是 field_path
if stage_name in ["calibrate", "evaluate"]:
    detect_record_path = run_root / "records" / "detect_record.json"
    arg_name = f"{stage_name}_detect_records_glob"
    stage_overrides.append(f"{arg_name}={json.dumps(str(detect_record_path))}")

if stage_name == "evaluate":
    thresholds_path = run_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
    stage_overrides.append(f"evaluate_thresholds_path={json.dumps(str(thresholds_path))}")
```

## 验证方法

运行 override 系统验证代码（已测试通过）：
```bash
python -c "
from main.policy.runtime_whitelist import load_runtime_whitelist
from main.policy import override_rules

whitelist = load_runtime_whitelist('configs/runtime_whitelist.yaml')
overrides = [
    'calibrate_detect_records_glob=\"/path/to/detect.json\"',
    'evaluate_detect_records_glob=\"/path/to/detect.json\"',
    'evaluate_thresholds_path=\"/path/to/thresholds.json\"',
]
parsed_list = [override_rules._parse_override_arg(o) for o in overrides]
resolved = override_rules._resolve_overrides(parsed_list, whitelist)
print('✓ 所有 override 都被允许')
"
```

结果：✅ 验证通过

## 预期效果

修复后 experiment_matrix 的子命令调用将成功：
- calibrate 阶段：接收 `--override calibrate_detect_records_glob=...`
- evaluate 阶段：接收 `--override evaluate_detect_records_glob=...` 和 `--override evaluate_thresholds_path=...`
- 所有参数都通过三层验证

## 关键学习

1. **override 系统的三层验证机制**：
   - 第一层：arg_name 白名单检查 (arg_name_enum)
   - 第二层：配置文件字段存在检查 (config 中的 field_path)
   - 第三层：override 权限检查 (allowed_overrides)

2. **CLI override 的键名规范**：
   - 使用 arg_name（如 "calibrate_detect_records_glob"）而不是 field_path（如 "calibrate.detect_records_glob"）
   - override_rules 首先尝试 arg_name 匹配，然后回退到 field_path 匹配

3. **实验矩阵的参数注入模式**：
   - 对动态用于 stage 构建的参数，需要在 whitelist 和配置中注册
   - 确保所有四个 stage（embed/detect/calibrate/evaluate）都能接收必要的参数

## 相关文件

- [configs/runtime_whitelist.yaml](../../configs/runtime_whitelist.yaml) - 白名单定义
- [configs/default.yaml](../../configs/default.yaml) - 基础配置
- [configs/paper_full_cuda.yaml](../../configs/paper_full_cuda.yaml) - 纸质配置
- [main/evaluation/experiment_matrix.py](../../main/evaluation/experiment_matrix.py) - 实现
- [main/policy/override_rules.py](../../main/policy/override_rules.py) - override 系统

