# paper_workflow Stage 00-05

## 作用范围

`paper_workflow` 是论文评测编排层，不重新实现 `main/` 下的方法主逻辑，而是在其之上补齐以下阶段边界：

1. family manifest 冻结
2. source / attack shard 执行
3. merge 与 materialize
4. canonical metrics 与 paper-facing exports
5. release packaging 与 signoff

当前真实实现覆盖 `PW00` 到 `PW05`：

1. `PW00`：family manifest、source split、attack event grid 与 shard plan 冻结。
2. `PW01`：`positive_source`、`clean_negative` 与可选 `control_negative` 的 source shard 隔离执行。
3. `PW02`：source merge、全局 thresholds、clean formal / derived exports、payload clean summary，以及 clean quality pair manifest。
4. `PW03`：`attacked_positive` shard 执行、attacked image materialization 与 staged detect records。
5. `PW04`：attack shard merge、canonical attack metrics、clean / attack unified quality、paper-facing tables / figures / supplemental metrics，以及可选 tail estimation。
6. `PW05`：artifact-only release packaging、formal readiness、signoff 与 frozen family bundle export。

## 当前设计要点

### 1. 会话隔离

所有 notebook 默认运行在彼此隔离的 Colab 会话中。下游阶段只允许依赖已经落盘的工件，不能依赖上游会话中的内存状态、临时缓存或当前环境里的临时依赖安装结果。

### 2. PW02 与 PW04 的 quality 边界

当前 clean / attack quality 责任边界已经调整为：

1. `PW02` 不再输出旧式 formal clean quality summary CSV/JSON。
2. `PW02` 只输出 `clean_quality_pair_manifest.json`，把 clean 图像对、prompt 与 event 绑定冻结下来。
3. `PW04` 统一负责 clean 与 attack 的 `PSNR`、`SSIM`、`LPIPS`、`CLIP text similarity` 聚合结果。

### 3. PW05 的 signoff 语义

当前 `PW05` 是 artifact-only signoff：

1. `quality_clean` 读取 `PW04 clean_quality_metrics.json`
2. `quality_attack` 读取 `PW04 attack_quality_metrics.json`
3. 不再根据当前 notebook 会话能否导入 `lpips` 或 `open_clip` 决定是否通过 signoff

## 目录布局

family 运行目录位于：

```text
paper_workflow/families/<family_id>/
```

典型子目录包括：

```text
manifests/
snapshots/
source_shards/positive/
source_shards/negative/
source_shards/control_negative/
attack_shards/
runs/
logs/
runtime_state/
exports/
exports/pw02/
exports/pw04/
exports/pw05/
```

## 主要阶段输出

### PW02

关键输出位于 `exports/pw02/`：

1. `paper_source_finalize_manifest.json`
2. `thresholds/content/thresholds.json`
3. `thresholds/attestation/thresholds.json`
4. `formal_final_decision_metrics.json`
5. `derived_system_union_metrics.json`
6. `quality/clean_quality_pair_manifest.json`
7. `payload/payload_clean_summary.json`

`runtime_state/pw02_summary.json` 会继续向下游暴露：

1. `paper_source_finalize_manifest_path`
2. `clean_pair_artifacts_dir`
3. `clean_quality_pair_manifest_path`
4. `payload_clean_summary_path`
5. `analysis_only_artifact_paths`

### PW04

`PW04` 的主导出位于 `exports/pw04/`，其中兼容主线输出包括：

1. `manifests/attack_merge_manifest.json`
2. `attack_positive_pool_manifest.json`
3. `attack_negative_pool_manifest.json`
4. `formal_attack_final_decision_metrics.json`
5. `formal_attack_attestation_metrics.json`
6. `derived_attack_union_metrics.json`
7. `formal_attack_negative_metrics.json`
8. `per_attack_family_metrics.json`
9. `per_attack_condition_metrics.json`
10. `tables/attack_event_table.jsonl`
11. `tables/attack_family_summary.csv`
12. `tables/attack_condition_summary.csv`
13. `clean_attack_overview.json`

当前 unified quality 流水线还会额外写出：

1. `quality/quality_pair_plan.json`
2. `quality/shards/quality_shard_XXXX.json`
3. `quality/quality_finalize_manifest.json`
4. `metrics/clean_quality_metrics.json`
5. `metrics/attack_quality_metrics.json`

append-only 的 paper-facing exports 包括：

1. `metrics/paper_metric_registry.json`
2. `metrics/content_chain_metrics.json`
3. `metrics/event_attestation_metrics.json`
4. `metrics/system_final_metrics.json`
5. `metrics/bootstrap_confidence_intervals.json`
6. `tables/main_metrics_summary.csv`
7. `tables/attack_family_summary_paper.csv`
8. `tables/attack_condition_summary_paper.csv`
9. `tables/rescue_metrics_summary.csv`
10. `tables/bootstrap_confidence_intervals.csv`
11. `figures/attack_tpr_by_family.png`
12. `figures/clean_vs_attack_scope_overview.png`
13. `figures/rescue_breakdown.png`
14. `tradeoff/clean_imperceptibility.json`
15. `tradeoff/attack_distortion.json`
16. `tradeoff/quality_robustness_tradeoff.csv`
17. `tradeoff/quality_robustness_frontier.png`
18. `tail/estimated_tail_fpr_1e4.json`
19. `tail/estimated_tail_fpr_1e5.json`
20. `tail/tail_fit_diagnostics.json`
21. `tail/tail_fit_stability_summary.json`

`PW04 summary` 中当前关键字段包括：

1. `clean_quality_metrics_path`
2. `attack_quality_metrics_path`
3. `quality_pair_plan_path`
4. `quality_shard_paths`
5. `quality_finalize_manifest_path`
6. `quality_shard_count`
7. `paper_scope_registry_path`
8. `analysis_only_artifact_paths`
9. `analysis_only_artifact_annotations`

当前 paper-facing 主作用域只使用 canonical scope names：

1. `content_chain`
2. `event_attestation`
3. `system_final`

### PW05

`PW05` 的 release 输出位于 `exports/pw05/`：

1. `formal_run_readiness_report.json`
2. `signoff_report.json`
3. `release_manifest.json`
4. `workflow_summary.json`
5. `run_closure.json`
6. `stage_manifest.json`
7. `package_manifest.json`
8. `package_index.json`
9. `*.zip`

这里的 release package 会同时包含 canonical source artifacts、analysis-only artifact bindings 与 `PW05` 自身生成的审计工件。

## 关键脚本

当前核心脚本包括：

1. `paper_workflow/scripts/pw_common.py`
2. `paper_workflow/scripts/pw00_build_family_manifest.py`
3. `paper_workflow/scripts/pw01_run_source_event_shard.py`
4. `paper_workflow/scripts/pw01_run_source_event_shard_worker.py`
5. `paper_workflow/scripts/pw01_stage_runtime_helpers.py`
6. `paper_workflow/scripts/pw02_merge_source_event_shards.py`
7. `paper_workflow/scripts/pw02_metrics_extensions.py`
8. `paper_workflow/scripts/pw03_run_attack_event_shard.py`
9. `paper_workflow/scripts/pw_quality_metrics.py`
10. `paper_workflow/scripts/pw04_prepare_quality_pairs.py`
11. `paper_workflow/scripts/pw04_run_quality_shard.py`
12. `paper_workflow/scripts/pw04_finalize_quality_metrics.py`
13. `paper_workflow/scripts/pw04_merge_attack_event_shards.py`
14. `paper_workflow/scripts/pw04_paper_exports.py`
15. `paper_workflow/scripts/pw04_metrics_extensions.py`
16. `paper_workflow/scripts/pw05_release_signoff.py`

其中：

1. `pw_quality_metrics.py` 是 clean / attack unified quality 的底层计算器。
2. `pw04_prepare_quality_pairs.py`、`pw04_run_quality_shard.py`、`pw04_finalize_quality_metrics.py` 共同组成 `PW04` 的 quality prepare / shard / finalize 流水线。
3. `pw05_release_signoff.py` 负责最终 package staging、manifest、zip 与 signoff。

## 入口脚本

notebook 对应的 Python 入口包括：

1. `paper_workflow/scripts/PW00_Paper_Eval_Family_Manifest.py`
2. `paper_workflow/scripts/PW01_Source_Event_Shards.py`
3. `paper_workflow/scripts/PW02_Source_Merge_And_Global_Thresholds.py`
4. `paper_workflow/scripts/PW03_Attack_Event_Shards.py`
5. `paper_workflow/scripts/PW04_Attack_Merge_And_Metrics.py`
6. `paper_workflow/scripts/PW05_Release_And_Signoff.py`

`paper_workflow/notebook/` 下的 notebook 仍然保持薄封装角色，主要负责 Colab 会话参数、环境 bootstrap 与脚本调用。
