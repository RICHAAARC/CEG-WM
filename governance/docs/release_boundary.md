# Release And Extraction Boundary

`governance/tools/extract_release_package.py` 提供 `minimal_method_package`、`experiment_execution_package` 和 `paper_artifact_rebuild_package`。所有 profile 都排除 `.agents/`、`.codex/`、`governance/`、拆包工具本身、审计报告、缓存和运行输出。

拆包只改变交付边界，不改变研究语义或证据状态。先 dry-run 检查 `copied_files`、`missing_paths`、`safety_violations`、`validation_violations` 和 `release_candidate_ready`；包内 README、导入与适用测试必须独立通过。

`third_party/` 只能由实验执行 profile 显式纳入，且必须具有 baseline provenance。`release_candidate_ready` 不是方法有效性、论文证据或发布批准。

逐项交付清单见 `docs/reference/extraction_profiles.md`；工具 manifest 字段见 `governance/docs/extraction_manifest_contract.md`。
