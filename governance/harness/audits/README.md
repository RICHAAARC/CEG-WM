# Audit Modules

此目录中的每个模块执行一个独立、只读的治理检查，并返回统一的 `audit_name`、`decision`、`violations`、`checked_paths` 和 `summary` 结构。完整集合由上级 `run_all_audits.py` 调用。

审计必须依据已登记 policy 限定扫描范围，不能递归检查本地环境、缓存、第三方源码或运行输出。
