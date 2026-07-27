# Colab Usage Guide

当前 CEG-WM 没有具体 Notebook；以下流程约束后续 Colab 入口。
从 `method_implemented` 到 `runtime_verified` 的本地 CPU 优先、固定 Notebook、
Google Drive 结果包和独立审计流程见
[Runtime And GPU Qualification Workflow](runtime_gpu_qualification_workflow.md)。

1. 从 `notebooks/colab/` 选择语义明确的 Notebook。
2. 在环境准备 cell 中安装固定依赖并定位 repository root。
3. 加载项目配置，只从 `main/`、`runtime/`、`experiments/protocol/`、`experiments/runners/` 或 `paper_artifacts/` 的公开入口导入正式实现；不要把整个 `experiments/` 当作无边界工具箱。
4. 调用 runner 或 artifact builder，不在 cell 中手写正式 records、阈值、比较条件、tables 或 manifests。
5. 将运行输出写入未提交目录；需要迁移到 GPU 服务器时使用 `experiment_execution_package`。
6. 提交 Notebook 前清空 outputs 和 execution count。

只有出现真实重复代码后才创建 `notebooks/support/`，且其中只放环境、I/O 或展示 helper。计算、统计、协议和方法逻辑留在可独立测试的 repository modules 中。
