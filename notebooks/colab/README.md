# Colab Entrypoints

CEG-WM 的 Colab Notebook 放在此目录。文件名必须表达实际用途，例如方法机制探测、实验执行或 artifact 检查，不使用编号阶段或含义不明的状态词。

当前唯一 Notebook 是 `runtime_qualification.ipynb`，只负责：

1. 挂载 Drive 并在运行时收集 profile、package 路径、独立审核提供的完整 package
   SHA-256 以及可选 replay source；
2. 检查 GPU 与临时磁盘，从 Colab Secrets 读取 `HF_TOKEN` 和
   `CEG_WM_ROOT_KEY`；
3. 对 Drive bootstrap 单次读取并核对固定完整 SHA-256，以 `xb` 写入新的
   `/content` 快照并复核摘要；
4. 只调用已复核的本地快照，并显示正式结果或 `bootstrap_failure` 诊断包路径。

Notebook 不负责解包、manifest/allowlist/逐文件 identity、依赖安装、runner 正式
结果判定或任何方法/runtime 逻辑。这些职责由 execution package 外、绑定 package
schema version 1 的可信 bootstrap 和包内 runner 分开承担。

profile、package 和 expected SHA 均是运行时输入；切换
`smoke`/`qualification`/`replay` 或候选 archive 不修改并保存 Notebook 源。
提交副本必须保持 outputs 为空、execution count 为 null。
