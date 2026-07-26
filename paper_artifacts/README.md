# Paper Artifact Rebuild Layer

此目录从冻结 governed records 与 manifests 重建 tables、figures 和 reports。它只依赖 `experiments/protocol/`，不得重新运行方法、模型或攻击。

## 当前实现

- `digest.py`：对 JSON 兼容配置计算稳定 SHA-256 digest。
- `manifests.py`：定义和校验 `ArtifactManifest` 的最小结构。
- `artifact_manifest.py`：根据输入、输出与配置构造 manifest。

当前 CEG-WM 没有具体 table、figure 或 report builder，也不保存正式实验输出。
