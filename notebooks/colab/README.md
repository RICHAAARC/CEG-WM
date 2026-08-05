# Colab Entrypoints

CEG-WM 的 Colab Notebook 放在此目录。文件名必须表达实际用途，例如方法机制探测、实验执行或 artifact 检查，不使用编号阶段或含义不明的状态词。

`runtime_qualification.ipynb` 只负责：

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

当前快照绑定 candidate
`8b2344756c4c247906ff0d4eab68e46a773e13f5` 和 package SHA-256
`8290abeed79931eb7208ac9ca280f1ea401f4725abfead35f12617a0ef54dd38`，用户只需
**Run all**，不得编辑这些固定值。该 candidate 的既有 smoke 与 qualification
结果仍保存在独立 revision/run-id 目录，本快照不覆盖历史结果；qualification run
`20260729T110628Z` 已通过独立审核，当前快照不得重复运行。
切换 `smoke`、`replay` 或候选 archive
时，必须由实施者形成新的 Notebook revision，并经独立审计者和 gatekeeper 审核。
提交副本必须保持 outputs 为空、execution count 为 null。

`experiment_execution.ipynb` 是 HF-only threshold-fit GPU execution 的独立薄入口。
它挂载 Drive，从 Colab Secrets 把 `HF_TOKEN` 与 `CEG_WM_ROOT_KEY` 仅桥接到子进程，
从 `https://github.com/RICHAAARC/CEG-WM.git` 拉取并核对 clean exact revision，然后
只调用仓库中的完整服务器入口。依赖安装、冻结模型获取、package/bootstrap 信任链、
正式 runner、records 和 ZIP 打包均不在 Notebook 中实现。

当前入口冻结为 source revision
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

`development_exploration.ipynb` 是 13 模块 development exploration 的独立薄入口。
Notebook 自身由后续 delivery revision 提供，但执行权威固定为已完成独立方法审核的
`7298cef0a6d4c497f2ba7b1a78e4ee4661cebe03`；它只从 GitHub 获取该 exact commit，
以 detached checkout 调用 `development_exploration_server.py`，不得改用 mutable branch。

用户只需选择 Colab GPU runtime、设置 `HF_TOKEN` 与 `CEG_WM_ROOT_KEY` 两个 Secrets、
执行 **Run all** 并授权 Drive。固定 run ID 为
`ceg_wm_development_exploration_science_first_v42`。旧
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
