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

`experiment_execution.ipynb` 是 HF-only threshold-fit GPU execution 的独立薄入口。它只接收
package/bootstrap/sidecar/embedded-manifest 的独立审核 SHA-256、精确 revision、
run ID、shard index 和 Colab Secrets；candidate/execution/fit 摘要由 clean exact
revision 的 builder 派生并通过已独立校验的 sidecar 绑定，Notebook 不接受人工注入。
它核对并快照 package 外 bootstrap，只调用该快照并下载 result 或 diagnostic ZIP。
入口每次只运行一个 frozen fit shard，不访问 untouched confirmation、不运行 baseline、
不批准 tau，也不形成科学效果声明。当前占位 trust inputs 未替换前不可执行，提交副本
必须保持 outputs 为空、execution count 为 null。
