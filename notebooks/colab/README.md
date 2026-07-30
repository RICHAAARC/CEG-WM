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

`experiment_execution.ipynb` 是独立的 A3b 薄入口。它要求调用者粘贴 package
外独立审核的 archive SHA-256、精确 revision 及 candidate/execution/input
摘要，核对并快照固定 SHA-256 的 package 外 bootstrap，然后只调用该快照并下载
bootstrap 返回的 result 或 diagnostic zip。其 package-contained 默认入口只做
CPU/synthetic development wiring，不运行模型或 GPU、不访问 held-out、不校准
阈值，也不形成科学效果证据。当前占位 trust inputs 未替换前不可执行。
