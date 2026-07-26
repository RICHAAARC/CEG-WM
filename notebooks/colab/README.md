# Colab Entrypoints

CEG-WM 的 Colab Notebook 放在此目录。文件名必须表达实际用途，例如方法机制探测、实验执行或 artifact 检查，不使用编号阶段或含义不明的状态词。

Notebook 只负责：

1. 安装固定依赖并定位 repository root；
2. 加载受治理配置；
3. 调用 `main/`、`runtime/`、`experiments/protocol/`、`experiments/runners/` 或 `paper_artifacts/` 的公开入口，不把整个 `experiments/` 当作工具箱；
4. 展示轻量诊断并导出未提交输出。

重复的 Colab session 辅助逻辑只有在确实出现后才收敛到 `notebooks/support/`。当前没有具体 Notebook。
