# Audit Modules

此目录中的每个模块执行一个独立、只读的治理检查，并返回统一的 `audit_name`、`decision`、`violations`、`checked_paths` 和 `summary` 结构。完整集合由上级 `run_all_audits.py` 调用。

审计必须依据已登记 policy 限定扫描范围，不能递归检查本地环境、缓存、第三方源码或运行输出。

`audit_dependency_boundaries.py` 还落实 `record_writer_layers`：实验 protocol、
methods、attacks 与 metrics 中出现显式文件写调用会失败；当前唯一允许的内部正式
records 写入层是 `experiments.runners`。
