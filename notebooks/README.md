# Notebook Entrypoints

此目录保存 Jupyter 与 Colab 的薄编排入口。Notebook 只能进行环境准备、配置选择、repository module 调用、轻量检查和展示；协议、方法、攻击、指标、records 和 artifact rebuild 的唯一实现必须留在可测试模块中。

## 目录约定

- `colab/`：Colab 入口。
- `support/`：只有出现真实重复的 Notebook 环境或展示代码后才创建；不得放科学计算核心。

提交的 Notebook 必须清空 cell outputs 和 execution count。运行输出写入未提交目录，正式 records 与 artifacts 通过 repository modules 生成。

当前有三个入口：

- `colab/runtime_qualification.ipynb` 固定服务已审核 runtime qualification；
- `colab/experiment_execution.ipynb` 只收集独立审核的 archive/revision/config/
  input 摘要，单次读取并核对 package 外 bootstrap，建立新的 `/content` 可信
  快照并只调用该快照。
- `colab/development_exploration.ipynb` 固定独立审核的 development execution
  revision，以 Drive 为跨会话 persistent root，并只调用该 revision 中的统一
  development server；结果 ZIP、receipt 与 `SHA256SUMS` 的 Drive export 只是交付
  便利，不是科学完成权威。

三者均不在 Notebook 源中实现 package 安全解包、manifest/allowlist、方法、
records、runner 或结果判定。
