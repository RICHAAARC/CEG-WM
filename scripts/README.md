# 审计和测试脚本使用说明

本目录包含冻结前对抗式审计的所有审计脚本和回归测试。

## 目录结构

```
scripts/
├── run_all_audits.py          # 审计聚合器（主入口）
└── audits/
    ├── audit_write_bypass_scan.py                  # B1/B5: 写盘旁路扫描
    ├── audit_yaml_loader_uniqueness.py             # A6: YAML 加载安全
    ├── audit_freeze_surface_integrity.py           # A1-A7: 冻结面完整性
    ├── audit_registry_injection_surface.py         # C1/C4: 注册表注入面
    ├── audit_policy_path_semantics_binding.py      # B3/D1/D2: 路径策略绑定
    ├── audit_dangerous_exec_and_pickle_scan.py     # D9: 危险执行扫描
    └── audit_network_access_scan.py                # D10: 网络访问扫描

tests/
├── conftest.py                                     # pytest fixture 配置
├── test_schema_requires_interpretation.py          # A2: schema 权威化
├── test_records_write_must_enforce_freeze_gate.py  # B1/A1: 写盘门禁
├── test_registry_seal_is_immutable.py              # C1: 注册表封闭
├── test_artifacts_semantic_bypass_guard.py         # B6/B5: artifacts 语义旁路
├── test_run_closure_must_exist_on_failure.py       # F1/F2: 失败闭包产出
└── test_records_bundle_anchor_consistency.py       # F3: bundle 一致性
```

## 使用方法

### 1. 运行所有审计脚本

在仓库根目录执行：

```powershell
python scripts/run_all_audits.py --repo-root . --output audit_report.json
```

**参数说明：**
- `--repo-root`：仓库根目录（默认当前目录）
- `--output`：报告输出路径（默认输出到 stdout）
- `--strict`：严格模式，对抗式扫描未输出命中列表视为 FAIL

**退出码：**
- `0`：允许冻结（FreezeSignoffDecision = ALLOW_FREEZE）
- `1`：阻止冻结（FreezeSignoffDecision = BLOCK_FREEZE）

### 2. 运行单个审计脚本

```powershell
python scripts/audits/audit_write_bypass_scan.py .
```

每个审计脚本接受一个参数（仓库根目录），输出 JSON 格式的审计结果。

### 3. 运行测试

安装依赖：
```powershell
pip install pytest
```

运行所有测试：
```powershell
pytest tests/ -v
```

运行特定测试：
```powershell
pytest tests/test_schema_requires_interpretation.py -v
```

## 审计结果格式

每个审计脚本输出统一的 JSON 格式：

```json
{
  "audit_id": "B1.write_bypass_scan",
  "gate_name": "gate.write_bypass",
  "category": "B",
  "severity": "BLOCK",
  "result": "PASS",
  "rule": "禁止绕过受控写盘路径直接写入 records 或关键产物",
  "evidence": {
    "matches": [],
    "fail_count": 0
  },
  "impact": "未发现阻断级写盘旁路",
  "fix": "N.A."
}
```

**字段说明：**
- `audit_id`：审计项唯一标识
- `gate_name`：门禁名称
- `category`：类别（A-G）
- `severity`：严重性（BLOCK / NON_BLOCK）
- `result`：结果（PASS / FAIL / N.A.）
- `rule`：规则描述
- `evidence`：证据（对抗式扫描包含 matches 命中列表）
- `impact`：影响说明
- `fix`：修复建议

## 聚合报告格式

`run_all_audits.py` 输出的聚合报告包含：

```json
{
  "summary": {
    "FreezeSignoffDecision": "ALLOW_FREEZE",
    "BlockingReasons": [],
    "RiskSummary": "LOW",
    "counts": {
      "PASS": 7,
      "FAIL": 0,
      "N.A.": 0,
      "BLOCK_fails": 0,
      "NON_BLOCK_fails": 0
    }
  },
  "results": [
    // 所有审计结果明细
  ],
  "metadata": {
    "repo_root": "D:\\Code\\WM-Framework",
    "audit_count": 7
  }
}
```

## 判定规则

### FreezeSignoffDecision
- **ALLOW_FREEZE**：无 BLOCK 级 FAIL，允许进入下一阶段
- **BLOCK_FREEZE**：存在 BLOCK 级 FAIL，必须修复后才能冻结

### RiskSummary
- **HIGH**：存在 BLOCK 级 FAIL
- **MED**：存在 NON_BLOCK FAIL，但无 BLOCK 级
- **LOW**：全部 PASS 或仅 N.A.

## 注意事项

1. **对抗式扫描必须输出命中列表**：即使为空列表也必须显式输出，否则聚合器会标记为 FAIL。

2. **BLOCK vs NON_BLOCK**：审计脚本不得自行升级严重性等级，必须与清单保持一致。

3. **证据定位**：所有 FAIL 必须给出"路径 + 行号区间或函数名"，行号必须由脚本实际计算。

4. **tests/ 覆盖度**：当前测试集为最小回归闭环，覆盖关键门禁与冻结不可旁路检查项。E2E 测试不在此范围内。

## 集成到 CI/CD

可将审计脚本集成到 pre-commit 或 CI 流水线：

```yaml
# .github/workflows/audit.yml
name: Freeze Audit
on: [pull_request]
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: python scripts/run_all_audits.py --repo-root . --output audit_report.json
      - uses: actions/upload-artifact@v3
        with:
          name: audit-report
          path: audit_report.json
```

## 故障排查

### 审计脚本执行失败
检查 Python 版本（需要 3.8+）和依赖：
```powershell
python --version
pip install pyyaml
```

### 测试失败
某些测试依赖实际实现，当前标记为 `pytest.skip` 或 `pytest.xfail`。随着实现推进，逐步取消跳过标记。

### 命中列表为空但仍 FAIL
检查分类逻辑，确认 `classify_match` 函数是否正确区分 ALLOWLISTED / WARNING / FAIL。

## 后续扩展

当前审计集合覆盖 A-G 类别的关键 BLOCK 项。后续可新增：
- E 类（统计与判决语义）的细化审计
- G 类（recommended_enforce）的执行态三态化审计
- 集成测试脚本用于 E2E 验证
