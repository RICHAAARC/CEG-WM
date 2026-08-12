# Colab Entrypoints

CEG-WM 的 Colab Notebook 放在此目录。文件名必须表达实际用途，例如方法机制探测、实验执行或 artifact 检查，不使用编号阶段或含义不明的状态词。

## Current authorized entrypoint

`content_routing_directional_diagnosis.ipynb` 是当前唯一授权在收到最终固定链接后执行
**Run all** 的入口。它固定拉取 execution revision
`925c2cbc727e3b18e91c0b3981eeed1b470a955a`，使用全新独立 run ID
`ceg_wm_content_routing_positive_reference_support_correction_diagnosis`，并执行冻结的
`2 operational + 32 reference fit + 8 paired scientific = 42 total` units；每个 unit
只允许 attempt zero。Notebook 只挂载 Drive、读取 `HF_TOKEN` 与
`CEG_WM_ROOT_KEY`、核对 detached exact checkout、调用
`content_routing_directional_diagnosis_server.py`，并以 create-only 方式导出 result 或
diagnostic ZIP、execution receipt 与 `SHA256SUMS`。失败时先完成导出再报错。

原始 T/R/Q records 保留 finite nonnegative 值（包括零）。每个 probe fold 从 fold 外 24 条
records 展平完整空间值，仅 strictly positive 子集进入一次 exact nearest-rank P95，且最终
P95 必须大于零。缺少正值支持属于 implementation/dependency blocked，不得解释为科学阴性。

该入口比较同 prompt/seed/key/base latent、共同实际总预算下的 routed 与 uniform control；
全部 42 units 以及 8 个 probe 的两条 arm 都使用固定 `a = 0.50`。该系数只作为本次
routing 因果控制，不是 alpha selection、组合权重选择或跨 alpha 外推。通过时只允许另行
申请 fixed-half routing directional validation。它只形成 development routing directional
diagnosis；不拟合 threshold/FPR，不形成正式组合、candidate promotion、calibration、
evaluation、baseline 或论文 claim。旧 run `ceg_wm_content_routing_backend_binding_correction_diagnosis`
及全部历史 records 保持 producer-bound；当前入口不读取、恢复、迁移、改写或拼接到本次
全新 run 的 8-probe 分母。

`qk_synchronization_write_diagnostic.ipynb` 当前为 **paused / not authorized**。它此前是
唯一授权在收到最终固定链接后执行
**Run all** 的入口。它固定拉取 execution revision
`24042298bef550803c1710b84485c07ca6223cf2`，使用独立 run ID
`ceg_wm_qk_vae_checkpoint_operation_localization`。当前入口仅用于验证已审核的
suffix-memory correction，execution authority 精确为
`1 operational / 0 scientific / 1 total / 1 attempt`：只运行 unit0 attempt0，并在
operational success 或安全 failure diagnostic 后立即停止，不执行休眠的 12 个 ratio
或 16 个 transform units，不产生 scientific aggregate。Notebook 挂载 Drive、读取
`HF_TOKEN` 与 `CEG_WM_ROOT_KEY`、核对 detached
exact checkout、调用 `qk_synchronization_write_diagnostic_server.py`，并把服务器的
result/diagnostic ZIP、execution receipt 与 `SHA256SUMS` 保存到 Drive export 目录。
旧 run ID `ceg_wm_qk_runtime_failure_localization`、
`ceg_wm_qk_synchronization_write_public_rgb8_diagnosis` 与更早的
`ceg_wm_qk_synchronization_write_diagnosis` 下的 records、diagnostics 与 intents
保持不可变；当前入口不读取、迁移、覆盖或混入这些历史执行内容。

依赖安装、基础 GPU 检查、冻结模型 revision 下载、真实 public method/runtime 调用、
operational record、持久化和内部 ZIP 全部属于服务器脚本。该 memory-correction preflight
入口不形成Q/K机制阳性或阴性，也不让
几何产生水印阳性，不形成 formal ratio selection、transform estimator 结论、threshold、
FPR、candidate promotion、calibration、formal evaluation、baseline 或论文 claim。

`lf_whitened_directional_validation.ipynb` 已完成其冻结的 1 个 non-scientific
public-endpoint smoke 与 32 个 LF whitened directional scientific units，当前为
**paused / not authorized**。其 execution revision
`51adb765cdddafcb4c65c357e899c77b4c9f36d2` 与 run ID
`ceg_wm_lf_whitened_directional_validation_prepared_feature_execution` 保持
producer-bound，不读取、不迁移、不改写或混入当前 content-routing 诊断分母。

依赖安装、基础 GPU 检查、冻结模型 revision 下载、真实方法/runtime 调用、正式
records、持久化和内部 ZIP 全部属于服务器脚本。该 development-only LF whitened
directional validation 只读使用 producer revision
`a78c47184cf83ad351bb4442ebd31c218726de25`、run ID
`ceg_wm_lf_whitening_asset_fit_and_score_screening` 已冻结的 whitening asset，不重拟合、
迁移或回写旧 run。它不拟合 threshold，不形成 FPR、candidate promotion、calibration、
formal evaluation、baseline 或论文 claim，也不执行 routing、LF/HF 组合、Q/K 或几何恢复。
旧 execution revision `194eccdd1f16c295528a4d9e1d7c75c2748f061a` 与旧 run ID
`ceg_wm_lf_whitened_directional_validation` 保持 producer-bound 历史身份，当前为
**paused / not authorized**，不得作为当前入口运行或与新 run 混合。
除当前 content-routing 入口外，目录内其余全部 Notebook 均为
**paused / not authorized**，不得作为当前入口运行。

`lf_whitened_score_screening.ipynb` 已完成 1 个 non-scientific operational smoke、32 个
clean null-fit 与 8 个 paired raw-vs-whitened screening units，当前为
**paused / not authorized**。其冻结 whitening asset 仅作为上述方向验证的只读输入；旧
records 保持原 producer/run 身份，且不进入新验证的 scientific 分母。

`lf_transmission_diagnostic.ipynb` 及其 execution revision
`2337f9d7c773a6054d558108e31d07d35fbee42f`、run ID
`ceg_wm_lf_carrier_to_detector_transmission_diagnostic` 已完成历史诊断职责，当前为
**paused / not authorized**。其 records 保持独立，不读取、不迁移、不改写或混入当前
LF whitening fit 与 screening 分母。

`hf_only_detector_directional_validation.ipynb` 及其 execution revision
`0d4253ab2614c642563c566e6268565c337b503f`、run ID
`ceg_wm_hf_only_detector_directional_validation_binary32_budget_authority` 当前为
**paused / not authorized**。更早的 run
`ceg_wm_hf_only_detector_directional_validation_initial_gate` 及其 records 是
immutable partial evidence；当前 LF 入口不读取、不迁移、不改写或混合这些 namespace。

`hf_transmission_diagnostic.ipynb` 已完成其历史传输诊断职责，当前为
**paused / not authorized**；其 revision、run 与 records 保持独立历史证据，不得续跑或
混入本次 detector directional validation。历史 execution revision 为
`af1eea8f55086b583e3e5e4a02586959983db70b`，run ID 为
`ceg_wm_hf_transmission_diagnostic_server_execution`。

`runtime_qualification.ipynb` 保留已审核 runtime qualification 的历史权威，但当前
**paused / not authorized**，不得再次 Run all。其既有职责为：

1. 挂载 Drive，并使用已审核后固定在源中的 candidate revision、`qualification` profile、
   `current` package 路径、完整 package SHA-256 和空 replay source；
2. 检查 GPU 与临时磁盘，从 Colab Secrets 读取 `HF_TOKEN` 和
   `CEG_WM_ROOT_KEY`；
3. 对 Drive bootstrap 单次读取并核对固定完整 SHA-256，以 `xb` 写入新的
   `/content` 快照并复核摘要；
4. 只调用已复核的本地快照，并显示正式结果或 `bootstrap_failure` 诊断包路径。

Notebook 不负责解包、manifest/allowlist/逐文件 identity、依赖安装、runner 正式
结果判定或任何方法/runtime 逻辑。这些职责由 execution package 外、绑定 package
schema version 1 的可信 bootstrap 和包内 runner 分开承担。

保留快照绑定 candidate
`8b2344756c4c247906ff0d4eab68e46a773e13f5` 和 package SHA-256
`8290abeed79931eb7208ac9ca280f1ea401f4725abfead35f12617a0ef54dd38`。在当时已结束的
授权运行中，用户只需 **Run all** 且不得编辑这些固定值；这不是当前运行指令。该
candidate 的既有 smoke 与 qualification
结果仍保存在独立 revision/run-id 目录，本快照不覆盖历史结果；qualification run
`20260729T110628Z` 已通过独立审核，当前快照不得重复运行。
切换 `smoke`、`replay` 或候选 archive
时，必须由实施者形成新的 Notebook revision，并经独立审计者和 gatekeeper 审核。
提交副本必须保持 outputs 为空、execution count 为 null。

`experiment_execution.ipynb` 保留 HF-only 4096 threshold-fit GPU execution 的独立
权威入口，但当前 **paused / not authorized**，不得 Run all。它原有职责是：
它挂载 Drive，从 Colab Secrets 把 `HF_TOKEN` 与 `CEG_WM_ROOT_KEY` 仅桥接到子进程，
从 `https://github.com/RICHAAARC/CEG-WM.git` 拉取并核对 clean exact revision，然后
只调用仓库中的完整服务器入口。依赖安装、冻结模型获取、package/bootstrap 信任链、
正式 runner、records 和 ZIP 打包均不在 Notebook 中实现。

保留入口冻结为 source revision
`7797e78a4da11ee39d5554772b299821ea0019b3` 的 fit shard zero。服务器入口在本地
`/content` 生成 result 或 diagnostic ZIP 与机器 receipt；Notebook 核对 artifact
SHA-256 后，将 ZIP 与 receipt 原子复制到
`MyDrive/CEG-WM/hf_only_threshold_fit_results/<revision>/<run-id>/shard_00/`，并在复制后
重算 SHA-256。此前绑定 Drive 外部 package 的 `93215a9...` Notebook/交付在执行前
已被本 GitHub checkout 流程替代；历史外部目录保留但不得运行或作为证据。

入口每次只运行一个 frozen fit shard，不访问 untouched confirmation、不运行 baseline、
不批准 tau，也不形成科学效果声明。本次 Notebook revision 只绑定服务器 source
revision，不改变候选、阈值、split 或阶段。提交副本必须保持 outputs 为空、execution
count 为 null。

`development_exploration.ipynb` 已完成 13 模块 operational validation，现为
**permanently paused / not authorized**，不得再次 Run all。Notebook 自身由既有
delivery revision 提供，执行权威固定为已完成
独立方法审核的 `7e449aa29f53ea38e3a044681c75c8f3dccff135`；它只从 GitHub 获取该 exact commit，
以 detached checkout 调用 `development_exploration_server.py`，不得改用 mutable branch。

用户只需选择 Colab GPU runtime、设置 `HF_TOKEN` 与 `CEG_WM_ROOT_KEY` 两个 Secrets、
执行 **Run all** 并授权 Drive。固定 run ID 为
`ceg_wm_thirteen_module_mechanism_screening_session_resume_validation`。冻结研究预算为 240 scientific 加
42 operational，共 282 units、最多 846 attempts。本次入口固定传入
`--maximum-wiring-clusters 2` 与 `--stop-before-scientific-units`。新 namespace 的首次
session 最多完成 2 个 preflight 加 2 个 wiring units，即只提交 units 0–3：
4 operational、0 scientific。后续
Run all 验证有效 session receipt 关闭未过期 lease 后的即时恢复，每个 session 最多恢复
2 个 wiring，直至累计 10 个 operational units。入口始终在 unit 10 scientific 边界前
成功停止，完成 10 个 operational 后重复 Run all 也不会创建新 commit、scientific intent、
bundle、marker 或 module outcome。实际运行已完成 2 个 preflight 和全部 8/8 wiring
smoke clusters，并以 `authorized_operational_boundary_reached` 闭合。上述 operational
units 不计入 13 模块科学覆盖。

旧 execution revision `2ff836f45c4012010092f7075e749507ae2ad9ae`、旧 run
`ceg_wm_thirteen_module_mechanism_screening` 及其 dangling intent 仅作为 immutable
diagnostic 保留。未获运行批准的 execution revisions
`ce536f1ad66b5f45c05d7b0a08e5c83fb8fb4b29`、
`6c84cb121030a1190a183955dd4a27798a0eb975` 及 recovery delivery namespace
`ceg_wm_thirteen_module_mechanism_screening_preflight_recovery` 也原样保留。当前
`b66cb04ebb41f0d5473c498ad5769b467ff26d7e` 的旧 run
`ceg_wm_thirteen_module_mechanism_screening_operational_validation`，包括 4 个已提交
operational units 与第二次 active-writer diagnostic，也保持 immutable。当前
session-resume-validation run 不读取、不续跑、不迁移、不改写、不删除或混用上述任一 namespace。

旧 506-unit development authority 不是 active 入口或当前预算分母。旧
`ceg_wm_development_exploration_detector_crossfit_execution` run 及其中已有的
scientific records、operational records 与 diagnostic artifacts 均原样保留，不读取、
不迁移、不改写、不删除；旧
`ceg_wm_development_exploration_science_first_v42` run namespace 与其中已有 records、
dangling attempts、full artifacts 均原样保留，不读取、不迁移、不改写；旧
`ceg_wm_development_exploration_scientific_execution` run 原样保留：2 个
operational commits、0 个 scientific commits、unit 0002 attempt 0 仍为 dangling，诊断为
`builtins.AssertionError`；任何既有
`ceg_wm_development_exploration_joint_record_execution` 目录也保持原样。新入口不读取、
迁移或删除这两个旧 run。每次 session 自动生成唯一 session ID；
Drive 中的 persistent root 用于跨 session 恢复，
`/content` 只保存当次 checkout 和 cache。服务器遵守冻结的 21 小时 soft stop、24 小时
hard cap 与 unit/attempt 总预算，后续 session 只恢复下一未完成 unit。

服务器生成的 result 或 diagnostic ZIP 和 create-only receipt 会在核对 revision、run、
session、artifact path 与 SHA-256 后复制到
`MyDrive/CEG-WM/development_exploration/exports/<execution-revision>/<run-id>/<session-id>/`，
同时写入 `SHA256SUMS`。非零退出也先保留 diagnostic export 再报错。这些 export 是用户
回传便利；科学完成仍只认
`MyDrive/CEG-WM/development_exploration/persistent/<run-id>/` 下经服务器验证的
`COMMITTED` bundles。Notebook 不安装依赖、不下载模型、不实现方法、runner、records、
预算、协议或科学判断，并保持 outputs 为空、execution count 为 null。

从上述 exact execution revision 按服务器现有确定性 tracked-tree 打包逻辑重建的
development execution package 为 4,549,335 bytes，SHA-256 为
`260a76d0e10ddbcf705bbdfda11e5593c688d2b3957d1635b4404b498187067e`。该摘要只绑定
`7e449aa29f53ea38e3a044681c75c8f3dccff135` 执行包；Notebook delivery revision 不改变
包内协议或方法实现。

`thirteen_module_mechanism_screening.ipynb` 已完成历史 13 模块科学筛查，当前为
**paused / not authorized**，不得再次 Run all。它从 GitHub detached checkout 获取 exact
execution revision `7e449aa29f53ea38e3a044681c75c8f3dccff135`，但使用全新 run
`ceg_wm_thirteen_module_mechanism_scientific_screening`，从冻结 roster unit 0 开始。
该入口不传入 `--maximum-wiring-clusters` 或 `--stop-before-scientific-units`，因此由
repository server 按确定性顺序执行完整的 240 scientific + 42 operational = 282 units、
最多 846 attempts，并保持既有跨 session 恢复、21 小时 soft stop 与 24 小时 hard cap。

新入口不读取、续跑、迁移、改写、删除或混用已完成的
`ceg_wm_thirteen_module_mechanism_screening_session_resume_validation` namespace 或任一
更早历史 run。Notebook 仍只挂载 Drive、桥接 Secrets、拉取 exact Git revision、调用
`development_exploration_server.py` 并复制服务器生成的 ZIP/receipt/`SHA256SUMS`；不安装
依赖、不下载模型、不实现 runner、records、预算、协议或科学判定。提交副本保持 outputs
为空、execution count 为 null。本筛查仍属 development，不创建 formal tau、Calibration
Locked、candidate promotion、final-held-out evaluation、external baseline 或论文 claim。
