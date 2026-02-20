# Notebooks 目录索引

本目录包含用于在 Google Colab 环境中运行 CEG-WM 端到端验证的所有文档与工具。

## 📖 文档清单

### 快速入门（推荐从这里开始）

- **[QUICKSTART.md](QUICKSTART.md)** ⭐
  - 5 分钟快速上手指南
  - 最小化步骤，快速验证
  - 常见问题 1 分钟修复

### 完整指南

- **[NOTEBOOK_GENERATION_GUIDE.md](NOTEBOOK_GENERATION_GUIDE.md)** 📘
  - 11 个 Cell 的完整代码模板
  - 复制粘贴即可使用
  - 详细的技术说明与故障排查

- **[README.md](README.md)** 📄
  - 综合使用说明
  - 验收标准详解
  - 输出产物说明

### 技术总结

- **[../doc/Colab_Notebook交付总结.md](../doc/Colab_Notebook交付总结.md)** 📊
  - 完整的技术实现说明
  - 合规性验证记录
  - 后续改进建议

## 🚀 使用流程

### 第 1 步：选择入口文档

| 需求 | 推荐文档 | 预计耗时 |
|------|----------|----------|
| 快速验证，最小步骤 | **QUICKSTART.md** | 5 min 阅读 + 20 min 执行 |
| 完整理解所有细节 | **NOTEBOOK_GENERATION_GUIDE.md** | 15 min 阅读 + 20 min 执行 |
| 技术原理与设计决策 | **Colab_Notebook交付总结.md** | 20 min 阅读 |

### 第 2 步：准备环境

1. 打包仓库为 ZIP（见 QUICKSTART.md）
2. 打开 Google Colab
3. 配置 HF Token（可选）

### 第 3 步：执行验证

1. 按照文档复制 11 个 Cell 代码
2. 依次执行所有 Cell
3. 下载证据包（run_bundle.zip）

### 第 4 步：验收

检查 `summary.json`：

```json
{
  "embed_alignment": "PASS",
  "audit_decision": "ALLOW_FREEZE"
}
```

## 📦 输出产物

执行完成后将获得：

- **run_bundle.zip**: 完整证据包（约 50-100 MB）
  - run_root/: embed 运行记录与生成图像
  - run_root_detect/: detect 运行记录
  - run_logs/: 所有执行日志
  - summary.json: 验收摘要

## 🔧 工具文件

- **colab_template.ipynb**: 基础 Notebook 模板
- **../scripts/generate_colab_notebook.py**: Notebook 生成脚本（供开发使用）

## ❓ 常见问题

### Q: 为什么没有预生成的 .ipynb 文件？

**A**: Jupyter Notebook JSON 格式复杂，包含大量转义字符，直接生成容易出错。采用"文档 + 代码模板"方式更稳定可靠。

### Q: 能否一键运行？

**A**: 可以。未来可将 11 个 Cell 合并为单个 Python 脚本，通过 `!wget + %run` 方式一键执行。当前版本优先保证可读性与可调试性。

### Q: 支持哪些模型？

**A**: 当前主要支持 SD3.5 Medium。可通过修改 Cell 6 的 `MODEL_ID` 参数切换到其他 SD3 变体。

### Q: 本地能否运行？

**A**: 可以。将 Colab Cell 代码复制到本地 Jupyter Notebook 即可，但需要：
- GPU 环境（CUDA 11.8+）
- 足够内存（推荐 16GB+）
- 完整的 Python 依赖

## 📚 相关文档

- [仓库主 README](../README.md): 项目整体说明
- [configs/frozen_contracts.yaml](../configs/frozen_contracts.yaml): 冻结契约规范
- [scripts/README.md](../scripts/README.md): 审计脚本说明

## 🎯 验收标准

运行成功的 4 个必要条件：

1. ✅ **FreezeSignoffDecision** == `"ALLOW_FREEZE"`
2. ✅ **Pytest** 全部通过（failed=0）
3. ✅ **embed alignment** == `"PASS"`
4. ✅ **detect pf_status** 不为 mismatch/absent/failed

## 📞 技术支持

如遇问题：

1. 查看对应文档的"故障排查"章节
2. 检查执行日志（run_logs/）
3. 验证配置参数（Cell 6）
4. 确认 GPU 可用（torch.cuda.is_available()）

## 🔄 版本历史

- **v1.0** (2026-02-20): 初始版本
  - 支持 SD3.5 Medium
  - 11 Cell 完整流程
  - Paper faithfulness 对齐验证
  - 严格审计集成

---

**推荐阅读顺序**: QUICKSTART → NOTEBOOK_GENERATION_GUIDE → Colab_Notebook交付总结
