# 测试与回归套件说明

本目录包含冻结面完整性与门禁执行的回归测试闭环。

## 测试覆盖矩阵

| 测试文件 | 覆盖检查项 | 严重性 | 描述 |
|---------|-----------|--------|------|
| `test_schema_requires_interpretation.py` | schema.interpretation_is_required（legacy_code=A2） | BLOCK | schema 校验必须要求 interpretation |
| `test_records_write_must_enforce_freeze_gate.py` | records.write_path_enforces_freeze_gate（legacy_code=B1/A1） | BLOCK | records 写盘必须经过 freeze_gate |
| `test_registry_seal_is_immutable.py` | registry.seal_and_immutability（legacy_code=C1） | BLOCK | 注册表 seal 后不可变 |
| `test_artifacts_semantic_bypass_guard.py` | artifacts.semantic_bypass_is_blocked（legacy_code=B6/B5） | BLOCK | artifacts 不得包含 records 语义字段 |
| `test_run_closure_must_exist_on_failure.py` | evidence.run_closure_emitted_on_failure（legacy_code=F1/F2） | BLOCK | 失败时 run_closure 必产出 |
| `test_records_bundle_anchor_consistency.py` | evidence.records_bundle_anchor_consistency（legacy_code=F3） | BLOCK | bundle 检测 anchors 一致性 |

## 运行方式

### 运行全部测试
```powershell
pytest tests/ -v
```

### 运行特定测试文件
```powershell
pytest tests/test_schema_requires_interpretation.py -v
```

### 运行特定测试函数
```powershell
pytest tests/test_schema_requires_interpretation.py::test_schema_requires_interpretation_parameter -v
```

### 显示详细输出
```powershell
pytest tests/ -v -s
```

## Fixture 说明（conftest.py）

### `tmp_run_root`
创建临时 run_root 目录，包含标准子目录结构：
- `records/`
- `artifacts/`
- `logs/`

**用法：**
```python
def test_example(tmp_run_root):
    records_dir = tmp_run_root / "records"
    # 使用临时目录进行测试
```

### `minimal_cfg_paths`
创建最小配置文件集合：
- `configs/frozen_contracts.yaml`
- `configs/runtime_whitelist.yaml`
- `configs/policy_path_semantics.yaml`

**用法：**
```python
def test_example(minimal_cfg_paths):
    frozen_contracts = minimal_cfg_paths["frozen_contracts"]
    # 使用配置文件路径
```

### `monkeypatch_no_network`
禁用网络访问，确保测试隔离性。

**用法：**
```python
def test_example(monkeypatch_no_network):
    # 测试中任何网络访问都会抛异常
    import requests
    # requests.get(...) 会失败
```

### `monkeypatch_no_gpu`
模拟 CPU-only 环境，使 `torch.cuda.is_available()` 返回 False。

**用法：**
```python
def test_example(monkeypatch_no_gpu):
    import torch
    assert not torch.cuda.is_available()
```

### `mock_interpretation`
创建模拟的 interpretation 对象用于测试。

**用法：**
```python
def test_example(mock_interpretation):
    assert mock_interpretation["contract_version"] == "v1.0.0"
```

### `mock_registry_sealed`
创建模拟的 sealed registry 用于测试封闭性。

**用法：**
```python
def test_example(mock_registry_sealed):
    registry = mock_registry_sealed
    registry.seal()
    # 此后 register 应该抛异常
```

## 测试策略

### 1. 失败路径优先
测试首先覆盖“应该失败”的场景：
- 缺少 interpretation → fail-fast
- 绕过 freeze_gate → 写盘拒绝
- seal 后 register → 抛异常

### 2. 异常定位验证
每个失败测试验证异常信息包含：
- 门禁名称（gate_name）
- 字段路径（field_path）
- 失败原因（reason）

### 3. 正例覆盖
在失败路径验证后，提供正例：
- 正确提供 interpretation → 通过校验
- seal 前 register → 成功注册
- anchors 一致 → bundle 成功

### 4. xfail 与 skip 标记
- `pytest.skip`：依赖的模块或实现尚不存在
- `pytest.xfail`：实现存在但尚未完全符合规范

随着实现推进，逐步移除这些标记。

## 预期测试状态

### 当前行为（可审计）
部分测试可能 `skip` 或 `xfail`，这是正常的。关键是：
1. 测试逻辑正确表达了规范要求
2. 实现完成后可直接取消跳过标记
3. 失败路径的异常检查已到位

升级条件：新增或调整规则时，必须通过版本化方式演进，不影响冻结红线语义。

### 冻结前最终状态
所有 BLOCK 级测试必须 PASS，且：
- 无 skip（除非该功能明确不在 v1 范围内）
- 无 xfail
- 覆盖率达到关键门禁的 fail-fast 要求

## 扩展测试

当前测试集为回归测试闭环。按版本化计划可追加：

### 集成测试
- 端到端运行 `embed → detect → evaluate`
- 验证 run_closure 完整性
- 验证 records_bundle 可复算

### 性能测试
- 大规模 records bundle 性能
- 审计脚本扫描性能

### 对抗测试
- 构造恶意输入尝试绕过门禁
- 验证所有防护措施生效

## 故障排查

### ImportError: No module named 'main'
确保仓库根目录在 PYTHONPATH 中：
```powershell
$env:PYTHONPATH = "D:\Code\CEG-WM"
pytest tests/ -v
```

### 测试发现实现缺失
检查 skip message，确认是否需要补齐对应模块。

### 测试通过但不应该通过
检查是否使用临时方案（直接写入而非调用真实函数）。这些测试应标记 `xfail` 并在实现完成后移除标记。

## CI/CD 集成

可将测试集成到 CI 流水线：

```yaml
# .github/workflows/test.yml
name: Tests
on: [pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install pytest pyyaml
      - run: pytest tests/ -v --tb=short
```

## 测试编写规范

新增测试时遵循：

1. **函数命名**：`test_<module>_<behavior>`
2. **文档字符串**：包含中文功能描述和英文 docstring
3. **断言清晰**：每个 assert 附带失败消息
4. **隔离性**：使用 fixture 提供临时目录，不污染仓库
5. **异常检查**：验证异常类型和异常消息内容

示例：
```python
def test_schema_rejects_invalid_record(mock_interpretation):
    """
    Test that schema rejects records missing required fields.
    
    缺少必需字段的 record 必须被 schema 拒绝。
    """
    invalid_record = {"run_id": "test_001"}
    
    with pytest.raises(ValueError) as exc_info:
        schema.validate_record(invalid_record, interpretation=mock_interpretation)
    
    assert "required" in str(exc_info.value).lower(), \
        "Exception should mention missing required fields"
```
