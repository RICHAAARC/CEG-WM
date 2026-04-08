# Legacy Mainline Retirement

旧主线已正式退役，历史材料统一迁入 `archive/legacy_mainline` 相关目录保存。

退役范围：
- `notebook/00_main/`
- 旧主线专用 `scripts/01_*`、`scripts/02_*`、`scripts/03_*`、`scripts/04_*`
- 与旧主线强绑定且不再作为活动路径使用的测试

当前唯一活动主路径：
- `paper_workflow/`

约束：
- 活动代码与活动测试不得再引用 `notebook/00_main`、旧主线 notebook 名称或旧主线 runner/script 路径。
- 共享 helper（如 `scripts/notebook_runtime_common.py`、`scripts/workflow_acceptance_common.py`）若仍被 `paper_workflow` 使用，则继续保留在活动路径。