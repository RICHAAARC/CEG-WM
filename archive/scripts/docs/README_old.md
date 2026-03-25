# 审计与工作流脚本使用说明

本目录包含工作流编排、冻结前对抗式审计和复现实验相关脚本。

## 目录结构

```
scripts/
├── run_onefile_workflow.py                             # 一键全链路（embed/detect/calibrate/evaluate/audits/signoff）
├── run_cpu_first_e2e_verification.py                   # CPU smoke 闭环验收
├── run_paper_full_workflow_verification.py             # paper_full CUDA 正式验收
├── run_all_audits.py                                    # 审计聚合器（主入口）
├── run_freeze_signoff.py                                # 冻结签署工具（baseline/paper/publish profile）
├── run_experiment_matrix.py                             # 试验矩阵聚合器
├── run_publish_workflow.py                              # publish 工作流编排
├── run_repro_pipeline.py                                # 复现实验流水线
├── workflow_acceptance_common.py                        # workflow 验收摘要共用工具
└── audits/
  ├── audit_write_bypass_scan.py                         # records.write_path_is_unbypassable（legacy_code=B1/B5）
  ├── audit_yaml_loader_uniqueness.py                    # config.yaml_loader_is_safe_and_unique（legacy_code=A6）
  ├── audit_freeze_surface_integrity.py                  # freeze_surface.integrity_and_single_source_loading（legacy_code=A1-A7）
  ├── audit_registry_injection_surface.py                # registry.seal_and_runtime_injection_resistance（legacy_code=C1/C4）
  ├── audit_policy_path_semantics_binding.py             # policy.path_semantics_binding_and_audit_evidence（legacy_code=B3/D1/D2）
  ├── audit_dangerous_exec_and_pickle_scan.py            # runtime.dangerous_execution_and_deserialization_blocked（legacy_code=D9）
  ├── audit_network_access_scan.py                       # runtime.network_access_is_audited_or_blocked（legacy_code=D10）
  ├── audit_evaluation_report_schema.py                  # evaluation_report 锚点字段完整性（signoff BLOCK）
  ├── audit_records_fields_append_only.py                # append-only 字段注册一致性（signoff BLOCK）
  ├── audit_attack_protocol_implementable.py             # attack protocol 协议—实现一致性（paper/publish BLOCK）
  ├── audit_attack_protocol_report_coverage.py           # attack protocol 声明与报告覆盖率对齐（paper/publish BLOCK）
  ├── audit_experiment_matrix_outputs_schema.py          # experiment matrix 工件 schema（paper/publish BLOCK）
  └── audit_repro_bundle_integrity.py                    # reproduction bundle 完整性（paper/publish BLOCK）

tests/
├── conftest.py
├── test_*.py                                            # 回归测试集（以仓库当前 tests 目录为准）
└── ...
```

## 使用方法

### 验收入口

CPU smoke 闭环验收：

```powershell
python scripts/run_cpu_first_e2e_verification.py --config configs/smoke_cpu.yaml
```

paper_full CUDA 正式验收：

```powershell
python scripts/run_paper_full_workflow_verification.py --config configs/paper_full_cuda.yaml
```

### 0. 一键评测复现与审计流程（稳定锚点）

**完整工作流：从评测执行 → 报告生成 → 审计验证**

```powershell
# 步骤 1: 在仓库根目录执行（需配置好 conda 环境）
cd d:\Code\CEG-WM

# 步骤 2: 运行评测流程（produce evaluation_report.json）
# 假设已完成 embed/detect/calibrate，此步仅执行 evaluate
python -m main.cli.evaluate_cli --config configs/default.yaml --output outputs/smoke_detect

# 步骤 3: 运行审计聚合器（验证所有冻结约束）
python scripts/run_all_audits.py --repo-root . --output audit_report.json

# 步骤 4: 检查 FreezeSignoffDecision
# 如果值为 "ALLOW_FREEZE"，说明当前代码满足冻结条件
python -c "import json; r=json.load(open('audit_report.json')); print(f\"Decision: {r['summary']['FreezeSignoffDecision']}\")"

# 步骤 5: 运行完整回归测试（确保未引入破坏性变更）
pytest tests/ -q
```

**关键字段对应支撑锚点：**

| 锚点 | 对应文件 | 关键字段 | 语义 |
|------|------|------|------|
| evaluation_report.json | outputs/smoke_detect/ | `cfg_digest`, `attack_protocol_digest` | 评测前后配置一致性 |
| metrics_by_attack_condition[*] | evaluation_report.json | `group_key` = "family::params_version" | 声明的攻击条件被执行且上报 |
| audit_report.json | . | `summary.FreezeSignoffDecision` | 冻结审计集合决议 |
| protocol_conditions_count | audit report (coverage) | 与 reported_conditions_count 对比 | 协议完整性：声明=执行 |

**预期输出示例（PASS 状态）：**
```json
{
  "summary": {
    "FreezeSignoffDecision": "ALLOW_FREEZE",
    "BlockingReasons": [],
    "RiskSummary": "LOW"
  },
  "metadata": {
    "profile": "paper",
    "audit_count": 16
  }
}
```

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
  "audit_id": "records.write_path_is_unbypassable",
  "gate_name": "records.write_path_is_unbypassable",
  "legacy_code": "B1",
  "formal_description": "受控写盘路径必须不可旁路。",
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
- `result`：结果（PASS / FAIL / SKIP / N.A.）
- `rule`：规则描述
- `evidence`：证据（对抗式扫描包含 matches，协议对齐包含 condition 对比）
- `impact`：影响说明（可选）
- `fix`：修复建议（可选）

### 新增审计：audit_attack_protocol_report_coverage

**用途**：验证 attack_protocol.yaml 中声明的所有攻击条件（family::params_version）是否都被执行并上报到 evaluation_report.json。

**规则**：
- ✅ PASS：protocol_conditions_count == reported_conditions_count 且集合相等
- ❌ FAIL：存在在协议中声明但未出现在报告中的条件（missed_conditions 非空）
- ⚠️ FAIL：报告中包含未声明的条件（extra_reported_conditions 非空）
- ⊘ SKIP：evaluation_report.json 尚未生成或路径不可达

**证据字段**：
```json
{
  "audit_id": "audit.attack_protocol_report_coverage",
  "gate_name": "gate.attack_protocol_report_coverage",
  "category": "G",
  "severity": "BLOCK",
  "result": "PASS|FAIL|SKIP",
  "rule": "all declared attack conditions must be executed and reported; no undeclared conditions in report",
  "evidence": {
    "protocol_version": "attack_protocol_v1",
    "protocol_spec_path": "configs/attack_protocol.yaml",
    "eval_report_path": "outputs/smoke_detect/evaluation_report.json",
    "protocol_conditions_count": 8,
    "reported_conditions_count": 8,
    "declared_conditions": ["composite::rotate_resize_v1", "crop::v1", "gaussian_blur::v1", ...],
    "reported_conditions": [...],
    "missed_conditions": [],
    "extra_reported_conditions": []
  }
}
```

**集成位置**：
- run_all_audits.py AUDIT_SCRIPTS（15 个脚本，排序：协议实现后）
- paper profile：自动包含（PAPER_PROFILE_ADDITIONAL_AUDITS）
- publish profile：自动包含（继承 paper）

**失败排查**：
1. 若 evaluation_report.json 缺失：检查是否已运行 evaluate 步骤
2. 若 missed_conditions 非空：检查 attack_runner 是否真实执行了声明的攻击族
3. 若 extra_reported_conditions 非空：协议版本可能滞后，检查是否需要更新 attack_protocol.yaml

## 聚合报告格式

`run_all_audits.py` 输出的聚合报告包含：

```json
{
  "summary": {
    "FreezeSignoffDecision": "ALLOW_FREEZE",
    "BlockingReasons": [],
    "RiskSummary": "LOW",
    "counts": {
      "PASS": 15,
      "SKIP": 0,
      "FAIL": 0,
      "BLOCK_fails": 0
    }
  },
  "results": [
    // 所有审计结果明细（15 个审计的逐个输出）
  ],
  "metadata": {
    "repo_root": ".",
    "audit_count": 15,
    "profile": "paper"
  }
}
```

### evaluation_report.json 字段映射表

评测报告（由 main/evaluation/report_builder.py::build_evaluation_report 生成）包含以下必要字段及其用途：

| 字段 | 必需 | 类型 | 用途 | 绑定审计 |
|------|------|------|------|------|
| `report_type` | ✅ | str | 报告类型标识（固定 "evaluation_report"） | schema 完整性 |
| `evaluation_version` | ✅ | str | 报告版本（如 "v1"） | schema 完整性 |
| `cfg_digest` | ✅ | str | 配置摘要（SHA256）| 配置不变性 |
| `plan_digest` | ✅ | str | 攻击计划摘要 | 计划不变性 |
| `thresholds_digest` | ✅ | str | 阈值工件摘要 | thresholds 只读 |
| `threshold_metadata_digest` | ✅ | str | 阈值元数据摘要 | thresholds 只读 |
| `impl_digest` | ✅ | str | 实现摘要 | 实现不变性 |
| `ablation_digest` | ✅ | str | Ablation 摘要（可选 offset） | ablation 完整性 |
| `attack_trace_digest` | ✅ | str | 攻击执行跟踪摘要 | 执行追溯 |
| `attack_protocol_version` | ✅ | str | 协议版本（来自 attack_protocol.yaml） | 协议版本追踪 |
| `attack_protocol_digest` | ✅ | str | 协议规范摘要 | 协议不变性 |
| `fusion_rule_version` | ✅ | str | 融合规则版本 | 融合规则追踪 |
| `policy_path` | ✅ | str | 策略路径标识 | 策略绑定 |
| `metrics` | ✅ | dict | 整体统计（n_total, n_rejected, n_rejected_by_reason） | 统计完整性 |
| `metrics_by_attack_condition` | ✅ | list | 按条件分组的指标列表（每项含 group_key） | **协议覆盖率验证** |
| `anchors` | ✅ | dict | 分组计数锚点（n_total, n_rejected_*） | 锚点完整性 |

**关键字段：metrics_by_attack_condition**

```json
"metrics_by_attack_condition": [
  {
    "group_key": "crop::v1",
    "n_total": 120,
    "n_passed": 85,
    "n_rejected": 35,
    "precision": 0.71,
    "recall": 0.68,
    "f1": 0.69
  },
  {
    "group_key": "gaussian_blur::v1",
    "n_total": 100,
    "n_passed": 72,
    "n_rejected": 28,
    "precision": 0.72,
    "recall": 0.71,
    "f1": 0.715
  }
  // ... 更多条件
]
```

**group_key 格式强制约束**：
- 格式：`"{family_name}::{params_version_name}"`（例：`"rotate::v1"`）
- 排序：按 group_key 字母序排列（report_builder.py 第 190 行保证）
- 必须项：来自 attack_protocol.yaml::families 与 params_versions 的所有条件
- 长期稳定性：条件键一旦声明不可改变（append-only 原则）

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

当前审计集合覆盖 A-G 类别的关键 BLOCK 项。按版本化计划可追加：
- E 类（统计与判决语义）的细化审计
- G 类（recommended_enforce）的执行态三态化审计
- 集成测试脚本用于 E2E 验证
