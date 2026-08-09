# Executable Extraction Profiles

以下三项与外层 `governance/tools/extract_release_package.py` 中的 `PROFILES` 一一对应。完整开发仓库不是脚本 profile；拆包工具本身不进入任何交付包。

| profile_name | purpose | governance | runtime | artifact_builders |
| --- | --- | --- | --- | --- |
| `minimal_method_package` | 生成核心方法发布候选 | excluded | excluded | excluded |
| `experiment_execution_package` | 部署 Colab 或 GPU 服务器执行 | excluded | included | excluded |
| `paper_artifact_rebuild_package` | 从冻结 records 重建论文产物 | control plane excluded | excluded | included |

## `minimal_method_package`

Required includes:

```text
main/
pyproject.toml
```

存在时可加入 `configs/methods/`、`tests/unit/method/` 与 `tests/functional/method/`。runtime、experiments、paper artifacts、notebooks、infrastructure、scripts、governance、skills、baseline 配置、`third_party/` 和生成目录被排除。输出根 README 来自 `templates/release_readmes/minimal_method_package.md`，不复制框架根 README。

## `experiment_execution_package`

Required includes:

```text
main/
runtime/
experiments/
configs/
infrastructure/
tests/integration/
tests/smoke/
pyproject.toml
```

Optional include:

```text
third_party/  # 仅显式传入 --include-third-party
scripts/experiment_execution/  # 存在真实服务器执行脚本时
```

只有 baseline registry 存在有效登记时才能显式纳入 `third_party/`。Governance、拆包工具、docs、notebooks、paper artifacts、unit/constraint/functional/formal tests、helpers、fixtures 和生成目录被排除。不会复制整个 `scripts/`；只有目的明确的 `scripts/experiment_execution/` 可按存在性纳入。该 profile 执行已经由开发仓库定义的真实 modules 与配置，不把 Notebook 作为服务器实现。输出使用包内 README 模板。

## `paper_artifact_rebuild_package`

Required includes:

```text
configs/
experiments/protocol/
paper_artifacts/
docs/guides/artifact_rebuild.md
docs/reference/field_registry.md
docs/reference/artifact_evidence.md
tests/functional/test_governed_artifact_structures.py
pyproject.toml
```

存在真实重建脚本时，可纳入 `scripts/artifact_rebuild/`。包内只保留直接验证 records 与 artifact manifest 重建结构的 `test_governed_artifact_structures.py`；其他 functional tests 属于方法或运行验证，不进入论文产物包。main、runtime、method/attack/metric/runner implementations、notebooks、infrastructure、`third_party/`、拆包工具、extraction profile 说明、治理控制平面、integration tests 和生成目录被排除。输出使用包内 README 模板。纯治理说明不进入该包；artifact evidence 文档保留是因为它定义研究证据链语义。

## 就绪判定与安全检查

所有 profile 都在 manifest 中给出 `safety_violations`、`validation_violations` 和 `release_candidate_ready`。安全问题会阻止实际复制；缺少实质实现、对应测试或打包元数据不会伪装成发布完成，而会令候选状态为未就绪。`release_candidate_ready: true` 只表示包边界与独立验证前置条件齐全，不是方法有效性或论文证据。

## Dry-run

```bash
python3 governance/tools/extract_release_package.py \
  --profile experiment_execution_package \
  --root . \
  --output /tmp/generative_watermark_package_check \
  --dry-run
```

源仓库必须先满足自身分层与 records 写入权约束；拆包 profile 不改变这些研究语义。
