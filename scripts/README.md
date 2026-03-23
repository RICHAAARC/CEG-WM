# scripts 目录说明

当前 README 仅保留一条推荐链路：

- scripts/run_paper_full_cuda.py

## 用途

scripts/run_paper_full_cuda.py 是 paper_full_cuda 的 output-only 编排入口。

职责边界：

- 顺序执行 embed、detect、calibrate、evaluate。
- 当配置启用 experiment_matrix 时，补充执行 experiment_matrix。
- 直接消费 configs/paper_full_cuda.yaml。
- 不执行 formal acceptance。
- 不执行 signoff。
- 不生成旧 formal 验收链路中的补洞型控制逻辑。

## 使用方法

在仓库根目录执行：

```powershell
python scripts/run_paper_full_cuda.py --config configs/paper_full_cuda.yaml --run-root outputs/colab_run_paper_full_cuda
```

如果当前环境已经就绪，也可以直接使用默认参数：

```powershell
python scripts/run_paper_full_cuda.py
```

## 输入

- 配置文件：configs/paper_full_cuda.yaml
- 运行根目录：默认输出到 outputs/colab_run_paper_full_cuda

## 主要输出

脚本成功执行后，run_root 下的核心产物包括：

- records/embed_record.json
- records/detect_record.json
- records/calibration_record.json
- records/evaluate_record.json
- artifacts/thresholds/thresholds_artifact.json
- artifacts/evaluation_report.json
- artifacts/run_closure.json

如果 experiment_matrix 启用，还会额外生成：

- outputs/experiment_matrix/artifacts/grid_summary.json

## 适用场景

- notebook/Paper_Full_Cuda.ipynb 的主执行入口
- 本地或远端 GPU 环境下的 paper_full_cuda 项目输出生成

## 当前约束

本 README 暂不展开说明 scripts 目录中的其他脚本，后续按需要再补充。
  "metadata": {
    "repo_root": ".",
    "audit_count": 15,
    "profile": "paper"
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
