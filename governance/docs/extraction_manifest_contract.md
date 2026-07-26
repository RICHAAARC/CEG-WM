# Extraction Manifest Contract

本页登记 `governance/tools/extract_release_package.py` 生成的外层拆包清单字段。它们不是研究配置、实验 records、artifact provenance 或方法接口字段，因此不进入 `docs/reference/field_registry.md`。

| field_name | responsibility |
| --- | --- |
| `profile_name` | 本次使用的 extraction profile。 |
| `copied_files` | 实际复制的仓库相对文件列表。 |
| `missing_paths` | 缺失的必需路径。 |
| `safety_violations` | 敏感文件、配置、本机路径或 baseline 来源问题。 |
| `validation_violations` | 独立安装、实质实现或测试前置条件缺口。 |
| `third_party_included` | 是否显式纳入 vendored baseline 源码。 |
| `release_candidate_ready` | 交付边界与独立验证前置条件是否齐全。 |
| `excluded_parts` | profile 明确排除的路径部分。 |
| `dry_run` | 是否只生成清单而未复制文件。 |

这些字段只描述外层工具执行，不得进入研究 claims 或方法完成证据。
