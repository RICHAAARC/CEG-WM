# Google Colab 端到端验证 - 5 分钟快速上手

## 目标

在 Google Colab 中运行完整的 CEG-WM 水印系统，生成可审计的证据包。

## 前置准备

### 1. 打包仓库（2 分钟）

```bash
# 在本地仓库根目录
cd d:/Code/CEG-WM

# 使用 Git 打包（推荐）
git archive --format=zip --output=CEG-WM.zip HEAD

# 或使用压缩工具
# Windows: 右键 → 发送到 → 压缩(zipped)文件夹
# PowerShell: Compress-Archive -Path * -DestinationPath CEG-WM.zip
```

### 2. 获取 HF Token（可选，1 分钟）

如需使用 SD3.5 Medium（gated model）：

1. 访问 https://huggingface.co/settings/tokens
2. 创建新 token（Read 权限即可）
3. 复制 token 备用

## 在 Colab 中操作

### 第 1 步：创建 Notebook

1. 打开 https://colab.research.google.com/
2. 点击 **File → New Notebook**
3. 点击 **Runtime → Change runtime type → GPU (T4)**

### 第 2 步：复制代码

打开 [`NOTEBOOK_GENERATION_GUIDE.md`](NOTEBOOK_GENERATION_GUIDE.md)，依次复制 **11 个 Cell** 的代码到 Colab。

> **提示**: 每个 Cell 标题为 "### Cell X (Code/Markdown)"

### 第 3 步：配置 HF Token（可选）

如果需要访问 SD3：

1. 点击 Colab 左侧边栏的 🔑 **Secrets**
2. 点击 **+ Add new secret**
3. Name: `HF_TOKEN`
4. Value: 粘贴你的 token
5. **Enable notebook access**

### 第 4 步：执行所有 Cell

按顺序执行（Runtime → Run all，或 Ctrl+F9）：

| Cell | 操作 | 预计耗时 |
|------|------|----------|
| 1-2 | 上传 CEG-WM.zip | 1 分钟 |
| 3 | 安装依赖 | 3 分钟 |
| 4-6 | 配置与自检 | 1 分钟 |
| 7 | Embed（首次下载模型） | 8 分钟 |
| 8 | Detect | 2 分钟 |
| 9-10 | 测试与审计 | 5 分钟 |
| 11 | 打包下载 | 1 分钟 |

**总计**: 约 20 分钟（首次运行），后续约 10 分钟。

### 第 5 步：验收

下载 `run_bundle.zip`，解压后检查 `summary.json`:

```json
{
  "embed_alignment": "PASS",         // ✅ 对齐通过
  "audit_decision": "ALLOW_FREEZE"   // ✅ 审计通过
}
```

## 常见问题（1 分钟快速修复）

### Q1: 上传 ZIP 后报错 "未找到 configs/ 目录"

**原因**: ZIP 文件包含额外的文件夹层级

**修复**: Cell 2 会自动检测并进入嵌套目录，无需手动操作

### Q2: 模型下载失败

**错误信息**: `401 Unauthorized` 或 `gated repo`

**修复**:
1. 确认已配置 `HF_TOKEN` (见第 3 步)
2. 确认 token 有 Read 权限
3. 访问 https://huggingface.co/stabilityai/stable-diffusion-3.5-medium 接受许可

### Q3: Embed 执行时间过长

**原因**: 首次下载模型（约 6GB）

**正常行为**: 
- 首次: 8-10 分钟
- 后续: 2-3 分钟（模型已缓存）

**加速**:
- 使用 Colab Pro（更快的 GPU 和网络）
- 或等待模型下载完成后重新运行

### Q4: 对齐验证未通过

**检查**: `run_logs/embed.log` 最后 50 行

**常见原因**:
- 模型版本不匹配 → 检查 `MODEL_ID` 是否正确
- 配置参数超出范围 → 使用默认值（steps=20, guidance=4.5）

### Q5: 审计 BLOCK

**检查**: `run_logs/audit_result.json`

**定位**:
```python
# 在 Colab 新 Cell 中执行
import json
result = json.load(open('/content/run_logs/audit_result.json'))
for r in result['results']:
    if r['result'] == 'FAIL' and r['severity'] == 'BLOCK':
        print(f"BLOCK: {r['audit_id']} - {r['impact']}")
```

## 调整参数（可选）

在 Cell 6 修改配置：

```python
# 模型选择
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"  # 或 3-medium

# 推理参数
PROMPT = "your custom prompt"  # 自定义提示词
STEPS = 20                      # 推荐 20-28
GUIDANCE = 4.5                  # 推荐 4.5-7.0
SEED = 42                       # 固定随机种子

# 水印强度（在 overrides 中调整）
# "enable_content=true"
# "enable_geometry=false"  # 当前版本关闭几何链
```

## 输出说明

### run_bundle.zip 内容

```
run_root/
  ├── records/embed_xxx.json      # ✅ 包含对齐报告
  ├── artifacts/output_xxx.png    # ✅ 生成的水印图像
  
run_root_detect/
  ├── records/detect_xxx.json     # ✅ 检测结果

run_logs/
  ├── embed.log                   # 完整执行日志
  ├── pytest.log                  # 测试结果
  ├── audit_result.json           # ✅ 审计决策

summary.json                      # ✅ 验收摘要
```

### 验收标准

两项核心指标：

1. **embed_alignment**: `"PASS"` → paper_faithfulness 对齐成功
2. **audit_decision**: `"ALLOW_FREEZE"` → 所有门禁通过

## 下一步

- 查看生成的图像: `run_root/artifacts/output_*.png`
- 分析对齐报告: `run_root/records/embed_*.json` → `content_evidence.alignment_report`
- 检查审计详情: `run_logs/audit_result.json` → `results[]`

## 完整文档

- 详细说明: [NOTEBOOK_GENERATION_GUIDE.md](NOTEBOOK_GENERATION_GUIDE.md)
- 完整 README: [README.md](README.md)
- 技术总结: [../doc/Colab_Notebook交付总结.md](../doc/Colab_Notebook交付总结.md)

---

**预计总耗时**: 25 分钟（打包 2 min + 执行 20 min + 验收 3 min）

**难度**: ⭐⭐ (需要基本的 Colab 使用经验)
