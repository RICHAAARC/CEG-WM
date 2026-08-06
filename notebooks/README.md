# Notebook Entrypoints

此目录保存 Jupyter 与 Colab 的薄编排入口。Notebook 只能进行环境准备、配置选择、repository module 调用、轻量检查和展示；协议、方法、攻击、指标、records 和 artifact rebuild 的唯一实现必须留在可测试模块中。

## 目录约定

- `colab/`：Colab 入口。
- `support/`：只有出现真实重复的 Notebook 环境或展示代码后才创建；不得放科学计算核心。

提交的 Notebook 必须清空 cell outputs 和 execution count。运行输出写入未提交目录，正式 records 与 artifacts 通过 repository modules 生成。

当前有三个保留入口，但只有一个具有本批运行授权：

- `colab/development_exploration.ipynb` 是当前唯一允许 **Run all** 的入口。它绑定
  revision `2ff836f45c4012010092f7075e749507ae2ad9ae` 和 run
  `ceg_wm_thirteen_module_mechanism_screening`，以 Drive 为跨会话 persistent root，
  并只调用该 revision 中的统一 development server。本次入口固定传入
  `--maximum-wiring-clusters 2`，只运行 2 个 preflight 加 2 个 wiring operational
  units，共 4 operational、0 scientific；Agent2/Agent3 验真前不得运行完整筛查。
- `colab/experiment_execution.ipynb` 保留 HF-only 4096 threshold-fit 的既有权威身份，
  但本阶段暂停且未授权运行。
- `colab/runtime_qualification.ipynb` 保留已审核 runtime qualification 的历史权威，
  但当前暂停且未授权重复运行。

机制筛查冻结总预算为 240 scientific + 42 operational = 282 units，最多 846
attempts。旧 506-unit development authority 和旧 run IDs 只作为不可变历史边界，
不是 active 入口或当前预算分母；Git 历史继续保留。

结果 ZIP、receipt 与 `SHA256SUMS` 的 Drive export 只是交付便利，不是科学完成权威。

三者均不在 Notebook 源中实现 package 安全解包、manifest/allowlist、方法、
records、runner 或结果判定。
