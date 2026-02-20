# CEG-WM Colab Notebook 使用指南

本目录包含用于在 Google Colab 环境中验证"真实算法完全对齐（paper-faithful）"的端到端 Notebook。

## 文件清单

- **NOTEBOOK_GENERATION_GUIDE.md**: Notebook 生成完整指南（推荐阅读）
- **README.md**: 本文件（快速开始）

## 快速开始

### 方式1: 使用生成指南（推荐）

查看 [NOTEBOOK_GENERATION_GUIDE.md](NOTEBOOK_GENERATION_GUIDE.md)，其中包含：
- 11 个 Cell 的完整代码
- 复制粘贴即可使用
- 详细的故障排查步骤

### 方式2: 使用简化脚本

创建一个 Colab Notebook，在第一个 Cell 中执行：

```bash
# Windows PowerShell
Compress-Archive -Path * -DestinationPath CEG-WM.zip

# Linux/Mac
zip -r CEG-WM.zip . -x '*.git*' -x '__pycache__/*'

# 或使用 Git （推荐）
git archive --format=zip --output=CEG-WM.zip HEAD
```

## 配置 HF Token（可选）

若模型为 gated repository (如 SD3)：

1. 在 [Hugging Face](https://huggingface.co/settings/tokens) 创建 Token
2. 在 Colab 左侧边栏点击 🔑 **Secrets**
3. 添加：
   - Name: `HF_TOKEN`
   - Value: 你的 token

## 验收标准

检查下载的 `alignment_acceptance_summary.json`：

```json
{
  "final_verdict": "ACCEPT",  // 应为 ACCEPT
  "embed_status": {
    "alignment_overall_status": "PASS"  // 应为 PASS
  },
  "audit_status": {
    "freeze_decision": "ALLOW_FREEZE"  // 应为 ALLOW_FREEZE
  },
  "pytest_status": {
    "all_passed": true  // 应为 true
  }
}
```

## 可调参数（Cell E）

```python
# 模型配置
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
HF_REVISION = "main"

# 推理参数
INFERENCE_PROMPT = "a serene mountain landscape with a lake"
INFERENCE_NUM_STEPS = 20  # 推荐 20-28
INFERENCE_GUIDANCE_SCALE = 4.5  # 推荐 4.5-7.0
SEED = 42

# 水印配置
ENABLE_PAPER_FAITHFULNESS = True  # 必须为 True
ENABLE_CONTENT = True
ENABLE_GEOMETRY = False
```

## 输出产物

### 下载文件

- `sd3_paper_faithfulness_run_bundle.zip`: 完整证据包

### ZIP 内容

```
run_root/
  ├── records/           # embed 记录（embed_*.json）
  ├── artifacts/         # 生成图像
  └── logs/              # 运行日志
run_root_detect/
  ├── records/           # detect 记录（detect_*.json）
  └── ...
run_logs/
  ├── embed.log          # embed 执行日志
  ├── detect.log         # detect 执行日志
  ├── pytest.log         # 测试日志
  ├── audits_strict.log  # 审计日志
  └── audit_result.json  # 审计结果 JSON
alignment_acceptance_summary.json  # 验收摘要
```

## 常见问题

### Q1: 模型下载失败

**原因**: 未配置 HF_TOKEN 或 Token 无权限

**解决**:
1. 确保在 [Hugging Face](https://huggingface.co/settings/tokens) 创建了 Token
2. 在 Colab Secrets 中正确配置 `HF_TOKEN`
3. 确保 Token 有权限访问目标模型

### Q2: Embed 失败

**检查步骤**:
1. 查看 `run_logs/embed.log` 最后 50 行
2. 确认 GPU 可用：`torch.cuda.is_available()` 应返回 `True`
3. 检查配置参数是否合理（steps、guidance_scale 等）

### Q3: 对齐验证未通过

**原因**: `alignment_overall_status != "PASS"`

**排查**:
1. 打开 `run_root/records/embed_*.json`
2. 查看 `content_evidence.alignment_report.failures` 字段
3. 检查第一条 failure 的 `path` 和 `reason`

### Q4: 审计 BLOCK

**原因**: 违反了某个门禁规则

**排查**:
1. 打开 `run_logs/audit_result.json`
2. 查找第一条 `severity=="BLOCK" && result=="FAIL"` 的审计
3. 查看其 `audit_id`、`rule`、`impact` 字段

### Q5: Pytest 失败

**排查**:
1. 查看 `run_logs/pytest.log` 最后 100 行
2. 找到第一个 `FAILED` 的测试用例
3. 检查失败原因（通常是环境差异或缺少依赖）

## 技术说明

### SD3 架构

本 Notebook 使用 **Stable Diffusion 3.5 Medium**，采用 **DiT (Diffusion Transformer)** 架构，而非传统 U-Net。

模型通过 `diffusers.StableDiffusion3Pipeline` 加载，由仓库的 `main.diffusion.sd3.pipeline_factory` 统一管理。

### 门禁机制

仓库实施严格的冻结门禁：

- **YAML 加载唯一入口**: `main.core.config_loader.load_yaml_with_provenance()`
- **Digest 计算唯一入口**: `main.core.digests.canonical_sha256()`
- **Records 写入唯一入口**: `main.core.records_io.write_*`

所有修改必须符合冻结契约（`configs/frozen_contracts.yaml`），否则审计 BLOCK。

### Paper Faithfulness

对齐验证包括：

1. **Pipeline Inspector**: 检查 pipeline 结构与 paper spec 一致性
2. **Diffusion Tracer**: 追踪扩散过程关键状态
3. **Injection Site Binder**: 绑定注入点到 manifest
4. **Alignment Evaluator**: 评估整体对齐状态

最终在 `embed_record.json` 中生成 `alignment_report.overall_status`。

## 版本兼容性

| 组件 | 版本 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.9.1 |
| Diffusers | 0.32.0 |
| Transformers | 4.45.2 |
| CUDA | 11.8+ |

## 支持

如遇问题，请检查：

1. Colab 运行时类型是否为 **GPU**（T4 或更高）
2. 依赖版本是否与上表一致
3. 仓库代码是否与 Notebook 版本匹配

## 许可

遵循主仓库许可协议。
