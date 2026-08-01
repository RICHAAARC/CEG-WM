# Colab Usage Guide

Runtime qualification 入口是
`notebooks/colab/runtime_qualification.ipynb`。
从 `method_implemented` 到 `runtime_verified` 的本地 CPU 优先、固定 Notebook、
Google Drive 结果包和独立审计流程见
[Runtime And GPU Qualification Workflow](runtime_gpu_qualification_workflow.md)。

1. 当前冻结入口固定绑定 candidate
   `8b2344756c4c247906ff0d4eab68e46a773e13f5`、`PROFILE="qualification"`、`current`
   execution package 路径、独立审核的 archive SHA-256
   `8290abeed79931eb7208ac9ca280f1ea401f4725abfead35f12617a0ef54dd38` 和
   `REPLAY_SOURCE=None`；用户只需 **Run all**，不编辑 Notebook cell。该
   candidate 的 smoke 与 qualification 结果继续保存在各自的 revision/run-id
   目录，不被当前快照覆盖；qualification run `20260729T110628Z` 已通过独立审核。
2. Notebook 对 Drive bootstrap 只读取一次，先核对完整 SHA-256，再以 `xb` 写入
   全新的 `/content` 快照并复核摘要；只执行该本地快照，摘要不符时不启动
   subprocess。随后检查 GPU/磁盘并读取 Colab Secrets。bootstrap 位于 Drive 的
   `CEG-WM/runtime_qualification/bootstrap/package_schema_1/`，不属于待验证
   execution package。
3. bootstrap 在任何 pip、package import 或 runner 启动前，把 archive 单次流式
   复制到唯一 ephemeral `xb` 快照并同步核对调用者提供的 SHA-256；不匹配即删除
   快照且不解包。匹配后只处理本地快照，完成 ZIP 安全、manifest schema version 1、
   allowlist、完整文件集及逐文件 hash/size 检查。
4. 全部预信任检查通过后，bootstrap 才安装冻结依赖并调用 package runner。
   runner 仍唯一负责 `smoke`、`qualification`、`replay` 及正式结果。
5. runner 已启动且形成正式结果时，无论通过或失败都保存
   `ceg_wm_runtime_qualification_<run_id>.zip`；runner 启动前的 ingress、解包或
   pip 失败保存独立的
   `ceg_wm_runtime_bootstrap_failure_<run_id>.zip`，不能当作 qualification 结果。
6. 模型、HF/pip cache、解包目录和临时 tensor 只放 `/content`；Secret、模型与
   cache 不写 Drive。提交 Notebook 前清空 outputs 和 execution count。

expected package SHA 必须来自 package 外的独立审核结果，不能从 archive 同目录的
可替换 sidecar 自动信任。可信 bootstrap 本身以固定完整 SHA-256 绑定，其版本只
支持当前 package schema version 1；schema 改变时必须另行审核 bootstrap 与
Notebook trust anchor。

当前 qualification 快照不得重复运行。后续切换 `smoke`、`replay` 或其他候选
package 时，不由用户临时编辑
Notebook；必须由实施者修改固定快照，经独立审计者和 gatekeeper 审核后形成新的
Notebook revision。

独立的 `notebooks/colab/experiment_execution.ipynb` 服务 experiment_execution_delivery package
接线验证。它只核对并调用 package 外 bootstrap，要求 archive SHA、精确 revision
及 candidate/execution/input digest 来自独立审核；当前占位值未冻结前不能执行。
包内默认入口仅生成 CPU/synthetic development record 与 replay，不执行模型或
GPU、不访问 held-out，也不支持 runtime 或科学效果 claim。
