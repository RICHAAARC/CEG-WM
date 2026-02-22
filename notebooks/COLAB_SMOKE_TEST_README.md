# CEG-WM Colab Smoke Test 使用说明

## 概述

本文档描述如何在 Google Colab 上运行 CEG-WM 项目的端到端 smoke test。

**Notebook 文件**: `colab_smoke_test_sd_pipeline.ipynb`

---

## 目标与覆盖范围

本 smoke test 验证以下内容链工程闭合：

1. **Pipeline 构建**：在 GPU 环境成功构建 Stable Diffusion 3 pipeline
2. **Embed 流程**：真实输出产物 + records + run_closure
3. **Detect 流程**：读取 embed 产物，输出 content_score（或严格 failure semantics）
4. **Calibrate 流程**：生成 thresholds 工件（wrong-key null 分布）
5. **Evaluate 流程**：只读 thresholds 工件，输出 TPR@FPR 指标
6. **路径合规性**：全流程 artifacts 均在 path_policy 控制的 run_root 下
7. **一致性验证**：输出关键 digests 并验证一致性
8. **依赖检查**：统计并输出缺失依赖分支触发率

---

## 前置要求

### 1. Google Colab 环境
- **GPU 运行时**：必须选择 GPU 运行时（T4 或更高）
  - 菜单：`运行时` → `更改运行时类型` → `硬件加速器` 选择 `GPU`
- **磁盘空间**：至少 15GB 可用空间（用于模型下载与生成）

### 2. Hugging Face Token
- 访问 https://huggingface.co/settings/tokens
- 创建或复制一个 **读权限** token
- 需要接受 Stable Diffusion 3 模型的许可协议：
  - 访问 https://huggingface.co/stabilityai/stable-diffusion-3-medium
  - 点击 "Agree and access repository"

### 3. 项目文件准备
- 将 CEG-WM 项目打包成 `CEG-WM.zip`
- 确保包含以下关键文件和目录：
  ```
  CEG-WM/
  ├── configs/
  │   ├── frozen_contracts.yaml
  │   ├── default.yaml
  │   ├── runtime_whitelist.yaml
  │   └── policy_path_semantics.yaml
  ├── main/
  │   ├── cli/
  │   ├── core/
  │   ├── diffusion/
  │   ├── policy/
  │   ├── registries/
  │   └── watermarking/
  ├── requirements.txt
  └── ...
  ```

---

## 使用步骤

### Step 1: 上传 Notebook 到 Colab

1. 访问 https://colab.research.google.com/
2. 点击 `文件` → `上传笔记本`
3. 选择 `colab_smoke_test_sd_pipeline.ipynb`
4. 或者直接在 GitHub 上打开（如果仓库是公开的）：
   - `文件` → `在 GitHub 中打开` → 输入仓库 URL

### Step 2: 选择 GPU 运行时

1. 菜单：`运行时` → `更改运行时类型`
2. 硬件加速器：选择 `GPU`（推荐 T4）
3. 点击 `保存`

### Step 3: 运行 Cell A - 环境设置

按顺序运行 Cell A 的所有子 Cell：

#### A-1: 检测 GPU
- 验证 GPU 可用性
- 打印 Python 版本

#### A-2: 上传项目 Zip
- 如果项目目录不存在，按提示上传 `CEG-WM.zip`
- 方法 1：使用 Colab 文件浏览器手动上传
- 方法 2：使用代码上传（取消注释相关代码）

#### A-3: 设置 PYTHONPATH
- 验证项目文件完整性
- 设置 Python 导入路径

#### A-4: 安装依赖
- 安装固定版本的依赖包
- 可能需要 3-5 分钟

#### A-5: Hugging Face 登录
- **重要**：此步骤需要交互式输入 HF token
- 不会在 Notebook 中硬编码或显示 token
- 如果模型已缓存，可以跳过

### Step 4: 运行 Cell B - 预检

- 打印版本信息
- 验证核心模块可导入
- 验证事实源一致性
- 执行 pipeline preflight 检查

### Step 5: 运行 Cell C - 准备测试输入

- 创建 run_root 目录结构
- 定义 smoke test 配置：
  - 4 个 prompts
  - 2 个 seeds
  - 512×512 分辨率
  - 128 个 wrong-key 样本
- 准备 watermark keys

### Step 6: 运行 Cell D - Embed

- **最耗时的步骤**（5-15 分钟，取决于 GPU）
- 下载 SD3 模型（首次运行）
- 执行真实 watermark embed
- 验证产物与锚点

### Step 7: 运行 Cell E - Detect

- 从 embed 产物中检测 watermark
- 验证失败语义（status!=ok 时无 content_score）
- 统计 pipeline_missing 计数

### Step 8: 运行 Cell F - Calibrate

- 使用 wrong-key 样本执行 NP 校准
- 生成 thresholds 工件
- 输出 thresholds_digest 和 threshold_metadata_digest

### Step 9: 运行 Cell G - Evaluate

- 只读模式加载 thresholds
- 输出 TPR@FPR 和 reject_rate
- 验证 thresholds_digest 一致性

### Step 10: 运行 Cell H - 一致性断言

执行硬断言（失败即停止）：
1. cfg_digest 在所有阶段都存在
2. thresholds_digest 在 evaluate 与 calibrate 一致
3. status!=ok 时不存在 content_score
4. pipeline_missing 计数为 0
5. 输出目录未发生目录逃逸

### Step 11: 可选 - 运行 Cell I - pytest

- 可选步骤，可能耗时较长
- 默认已注释，如需运行请取消注释

### Step 12: 运行 Cell J - 总结

- 生成并打印 smoke test 总结表
- 保存总结到 JSON 文件
- 最终判定：PASS 或 FAIL
- 可选：下载结果到本地

---

## 预期输出

### 成功标志

如果 smoke test 成功，最后应看到：

```
================================================================================
🎉 SMOKE TEST 全部通过！
================================================================================
```

### 关键 Digests

总结中应包含以下关键 digests：

- **cfg_digest**: 各阶段的配置摘要
- **plan_digest**: Embed 阶段的子空间规划摘要
- **impl_digest**: 实现标识摘要
- **thresholds_digest**: 阈值工件摘要
- **threshold_metadata_digest**: 阈值元数据摘要

### 一致性检查

- ✓ 所有阶段都有 cfg_digest
- ✓ thresholds_digest 在 calibrate 和 evaluate 一致
- ✓ pipeline_missing 计数 = 0
- ✓ 路径合规性通过

---

## 常见问题

### Q1: GPU 内存不足 (OOM)

**症状**: CUDA out of memory 错误

**解决方案**:
1. 减少 smoke test 规模：
   - 减少 prompts 数量（从 4 降到 2）
   - 减少分辨率（从 512 降到 256）
   - 减少 wrong-key 样本数（从 128 降到 64）
2. 重启运行时并清理缓存：
   - `运行时` → `重启并运行所有单元格`

### Q2: HF Token 登录失败

**症状**: 模型下载失败，提示需要认证

**解决方案**:
1. 确认在 https://huggingface.co/stabilityai/stable-diffusion-3-medium 接受了许可协议
2. 检查 token 是否有 read 权限
3. 重新运行 Cell A-5

### Q3: 项目文件缺失

**症状**: Cell A-3 报告缺失关键文件

**解决方案**:
1. 检查 zip 文件是否完整
2. 确保 zip 根目录是 `CEG-WM/`，而不是嵌套目录
3. 重新打包并上传

### Q4: 依赖版本冲突

**症状**: 导入模块时报错

**解决方案**:
1. 确保按顺序运行 Cell A-4
2. 如果仍有问题，重启运行时
3. 检查 Colab 预装的包是否与项目冲突

### Q5: Pipeline 构建失败

**症状**: Embed 阶段报错 "pipeline_status: fail"

**解决方案**:
1. 检查模型下载是否完整
2. 查看详细错误日志
3. 确认 GPU 运行时已启用
4. 检查 configs/default.yaml 中的 model_id 是否正确

### Q6: Smoke Test 部分失败

**症状**: 某个阶段 status = FAIL

**解决方案**:
1. 查看该阶段的详细输出日志
2. 检查 run_closure.json 中的 failure_reason
3. 确认是否违反了冻结契约或一致性要求
4. 如果是预期的失败语义（如 absent/mismatch），验证是否符合规范

---

## 约束与限制

### 红线（禁止触碰）

1. ❌ 禁止修改 `configs/frozen_contracts.yaml` 既有字段语义
2. ❌ 禁止修改 canonical JSON/digest 口径
3. ❌ 禁止新增写盘旁路
4. ❌ Detect 侧不得旁路写 decision
5. ❌ Notebook 不得要求手动改代码
6. ❌ 不得输出明文 token

### Smoke Test 规模限制

为避免 Colab 超时，本 smoke test 使用小规模参数：

- Prompts: 4 条
- Seeds: 2 个
- 分辨率: 512×512
- Wrong-key 样本: 128 个
- 推理步数: 28 步

**总运行时间**: 约 10-20 分钟（取决于 GPU 和网络）

### 模型缓存

SD3 模型约 10GB，首次运行需要下载。后续运行会使用缓存。

---

## 输出文件结构

Smoke test 完成后，会在 `/content/runs/smoke_sd/` 下生成以下结构：

```
/content/runs/smoke_sd/
├── embed/
│   ├── records/
│   │   ├── run_closure.json
│   │   └── embed_records.jsonl
│   ├── artifacts/
│   │   └── (生成的图像等)
│   └── logs/
├── detect/
│   ├── records/
│   │   ├── run_closure.json
│   │   └── detect_records.jsonl
│   └── ...
├── calibrate/
│   ├── records/
│   │   └── run_closure.json
│   ├── artifacts/
│   │   ├── thresholds.json
│   │   └── threshold_metadata.json
│   └── ...
├── evaluate/
│   ├── records/
│   │   ├── run_closure.json
│   │   └── evaluation_results.json
│   └── ...
├── keys/
│   ├── correct_key.txt
│   └── wrong_keys.json
└── smoke_test_summary.json
```

---

## 高级用法

### 修改 Smoke Test 参数

在 Cell C-2 中修改 `SMOKE_CONFIG`：

```python
SMOKE_CONFIG = {
    "prompts": [...],          # 修改 prompt 列表
    "seeds": [42, 123, 456],   # 增加 seed 数量
    "resolution": 256,         # 降低分辨率以加速
    "num_inference_steps": 20, # 减少推理步数
    "target_fpr": 0.01,        # 修改目标 FPR
    "num_wrong_key_samples": 64,  # 减少校准样本数
}
```

### 启用 HF 通道

在 embed CLI 命令中添加 override：

```python
"--override", "watermark.hf.enabled=true",
```

### 下载完整结果

取消 Cell J-2 的注释，运行后会下载：
- `smoke_test_summary.json`
- `smoke_test_results.tar.gz`（完整 run_root）

---

## 故障排查

### 查看详细日志

每个阶段的详细日志位于：
- `{STAGE}_RUN/logs/`

### 检查 Run Closure

每个阶段的 `run_closure.json` 包含关键诊断信息：
- `status_ok`: 是否成功
- `status_reason`: 失败原因枚举
- `pipeline_status`: Pipeline 构建状态
- 各种 digests 和锚点

### 验证配置

检查配置是否正确加载：
```python
from main.core import config_loader
cfg = config_loader.load_yaml_raw("configs/default.yaml")
print(cfg)
```

---

## 联系与反馈

如遇到问题或需要支持，请：

1. 检查本 README 的常见问题部分
2. 查看项目文档：`doc/真实方法安全注入流程.md`
3. 提交 Issue 并附上：
   - Colab Notebook 输出截图
   - `smoke_test_summary.json` 内容
   - 失败阶段的 `run_closure.json`

---

## 版本历史

- **v1.0** (2026-02-22): 初始版本，支持 SD3 pipeline 端到端 smoke test
