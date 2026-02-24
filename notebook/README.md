# Colab 完整工作流 Notebook 使用指南

## 文件位置
- Notebook 文件：`notebook/colab_complete_workflow.ipynb`

## 使用步骤

### 1. 在 Google Colab 中打开 Notebook

1. 访问 [Google Colab](https://colab.research.google.com/)
2. 选择 "上传" → "上传 .ipynb 文件"
3. 选择 `notebook/colab_complete_workflow.ipynb` 文件

或者：
```
File → Open notebook → Upload
```

### 2. 准备 CEG-WM 代码仓库

**方式 A：从 Google Drive 上传（推荐）**
1. 将 CEG-WM 代码打包成 zip：
   ```bash
   zip -r CEG-WM.zip CEG-WM/ -x "CEG-WM/outputs/*" "CEG-WM/.git/*"
   ```

2. 上传到 Google Drive 的根目录

3. 在 Notebook 中执行上传步骤的代码

**方式 B：从 GitHub 克隆**
```python
!git clone https://github.com/your-repo/CEG-WM.git /tmp/ceg_wm_workspace/CEG-WM
```

### 3. 逐步执行 Notebook

按照顺序执行各个单元格：

1. **环境配置**：安装依赖和配置环境
2. **上传代码**：加载 CEG-WM 代码仓库
3. **下载模型**：获取必要的模型权重
4. **数据准备**：准备输入数据
5. **工作流执行**：运行完整的 embed → detect → calibrate → evaluate 流程
6. **结果打包**：压缩并下载结果

### 4. 配置选项

修改数据准备步骤中的配置选择：

```python
CONFIG_CHOICE = "paper_proof"  # 或 "default"
```

- `default.yaml`：基础配置，快速测试
- `paper_proof.yaml`：论文验证配置，启用完整功能

### 5. 下载结果

工作流完成后，会自动下载压缩的 run_root 目录（.zip 文件）。

## 预期输出

### run_root 目录结构
```
run_root/
├── artifacts/
│   ├── embed_output.json              # Embed 阶段输出
│   ├── detect_output.json             # Detect 阶段输出
│   ├── calibrate_output.json          # Calibrate 阶段输出
│   ├── evaluation_report.json         # 最终评估报告
│   ├── run_closure.json               # 运行元数据
│   ├── cfg_audit/                     # 配置审计
│   ├── env_audits/                    # 环境审计
│   └── repro_bundle/                  # 可复现性证据包
└── intermediate/                       # 中间缓存文件
```

### 关键输出文件

1. **evaluation_report.json**
   - 包含完整的评估指标
   - 模型性能统计
   - 各阶段的详细结果

2. **run_closure.json**
   - 运行完整性检查
   - 执行元数据
   - 时间戳和环境信息

## 常见问题

### Q：工作流执行超时
**A**：Colab 运行时有时间限制。对于大规模任务：
- 使用更小的数据集测试
- 分阶段执行工作流
- 使用本地 GPU 环境

### Q：GPU 不可用
**A**：在 Colab 中启用 GPU：
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. 保存并重启

### Q：模型下载失败
**A**：
- 检查网络连接
- 验证 Hugging Face 访问权限
- 如需私有模型，配置 HF token

### Q：内存不足（OOM）
**A**：
- 减少 batch size
- 使用 device="cpu" 和 GPU 交替
- 分割数据集进行多次运行

## 高级选项

### 自定义参数

在工作流执行步骤修改命令行参数：

```python
command = [
    "python",
    "scripts/run_repro_pipeline.py",
    "--run-root", str(RUN_ROOT),
    "--config", str(CONFIG_FILE),
    # 添加自定义参数
    "--override", "device='cpu'",  # 使用 CPU
]
```

### 分阶段执行

如果完整工作流失败，可以分别执行各阶段：

```bash
# Embed 阶段
python -m main.cli.run_embed --out outputs/my_run --config configs/default.yaml

# Detect 阶段
python -m main.cli.run_detect --in outputs/my_run --out outputs/my_run

# Calibrate 阶段
python -m main.cli.run_calibrate --in outputs/my_run --out outputs/my_run

# Evaluate 阶段
python -m main.cli.run_evaluate --in outputs/my_run --out outputs/my_run
```

## 环境要求

- **Python**：3.8+
- **GPU**（推荐）：NVIDIA GPU 或 Colab GPU
- **存储**：至少 50GB（用于模型和输出）
- **内存**：16GB+ 推荐用于完整工作流

## 获取帮助

1. 查看 Notebook 中的故障排除指南
2. 检查工作流日志：`RUN_ROOT/logs/workflow.log`
3. 参考配置文档：`doc/paper_proof_config_guide.md`

---

**版本**：1.0  
**更新时间**：2026 年 2 月 24 日  
**适用项目**：CEG-WM（防水印系统）
