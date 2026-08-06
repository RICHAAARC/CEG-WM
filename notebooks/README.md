# Notebook Entrypoints

此目录保存 Jupyter 与 Colab 的薄编排入口。Notebook 只能进行环境准备、配置选择、repository module 调用、轻量检查和展示；协议、方法、攻击、指标、records 和 artifact rebuild 的唯一实现必须留在可测试模块中。

## 目录约定

- `colab/`：Colab 入口。
- `support/`：只有出现真实重复的 Notebook 环境或展示代码后才创建；不得放科学计算核心。

提交的 Notebook 必须清空 cell outputs 和 execution count。运行输出写入未提交目录，正式 records 与 artifacts 通过 repository modules 生成。

当前有四个保留入口；operational 入口已完成并永久暂停，科学筛查入口是唯一待
Agent2/Agent3 独立批准的入口，在批准前不得运行：

- `colab/thirteen_module_mechanism_screening.ipynb` 是唯一待批准的完整科学筛查薄入口。
  它绑定 execution revision `7e449aa29f53ea38e3a044681c75c8f3dccff135` 和全新 run
  `ceg_wm_thirteen_module_mechanism_scientific_screening`，从冻结 roster unit 0 开始，
  不传入 `--maximum-wiring-clusters` 或 `--stop-before-scientific-units`。完整冻结预算
  为 240 scientific + 42 operational = 282 units，最多 846 attempts；跨 session
  恢复、soft stop、records 和结果打包仍完全由 repository server 承担。该入口不读取
  或续跑下述 operational run。
- `colab/development_exploration.ipynb` 已完成 2 个 preflight 和 8/8 wiring smoke，
  到达 `authorized_operational_boundary_reached`，现为 **permanently paused / not
  authorized**。它绑定
  revision `7e449aa29f53ea38e3a044681c75c8f3dccff135` 和全新 run
  `ceg_wm_thirteen_module_mechanism_screening_session_resume_validation`，以 Drive 为跨会话
  persistent root，
  并只调用该 revision 中的统一 development server。本次入口固定传入
  `--maximum-wiring-clusters 2` 与 `--stop-before-scientific-units`。首次最多运行 2 个
  preflight 加 2 个 wiring operational units，即仅 units 0–3，共 4 operational、
  0 scientific；同一新
  namespace 的后续 Run all 验证已结束 session 的即时恢复，每次最多恢复 2 个 wiring。
  累计 10 个 operational 后已停止，没有进入 unit 10 或创建 scientific 产物；不得
  再次 Run all。
- `colab/experiment_execution.ipynb` 保留 HF-only 4096 threshold-fit 的既有权威身份，
  但本阶段暂停且未授权运行。
- `colab/runtime_qualification.ipynb` 保留已审核 runtime qualification 的历史权威，
  但当前暂停且未授权重复运行。

机制筛查冻结总预算为 240 scientific + 42 operational = 282 units，最多 846
attempts。旧 506-unit development authority 和旧 run IDs 只作为不可变历史边界，
不是 active 入口或当前预算分母；Git 历史继续保留。

旧 execution revision `2ff836f45c4012010092f7075e749507ae2ad9ae`、旧 run
`ceg_wm_thirteen_module_mechanism_screening` 及其 dangling intent 是不可变诊断；
未获运行批准的 `ce536f1ad66b5f45c05d7b0a08e5c83fb8fb4b29` /
`6c84cb121030a1190a183955dd4a27798a0eb975` recovery delivery namespace
`ceg_wm_thirteen_module_mechanism_screening_preflight_recovery` 同样保持不变。当前
`b66cb04ebb41f0d5473c498ad5769b467ff26d7e` 的旧 run
`ceg_wm_thirteen_module_mechanism_screening_operational_validation` 及其 4 个已提交
operational units、第二次 active-writer diagnostic 也保持 immutable。当前
session-resume-validation run 不读取、不续跑、不迁移、不改写、不删除或混用上述任一历史 namespace。

结果 ZIP、receipt 与 `SHA256SUMS` 的 Drive export 只是交付便利，不是科学完成权威。

三者均不在 Notebook 源中实现 package 安全解包、manifest/allowlist、方法、
records、runner 或结果判定。
