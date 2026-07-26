# Core Watermark Method Package

`main/` 是 CEG-WM 可独立抽离的最小方法包。计划包含 `shared/`、`content_chain/`、
`geometry_chain/`、`joint_decision/` 和公开 API。未来实施固定为 13 项职责；
`content_chain` 中的 embedder、LF detector 与 `geometry_chain` 中的 reliability
都有独立路径，不得折叠到 carrier、content detector 或 transform estimator。

此层不得导入 runtime、experiments、paper artifacts、notebooks、infrastructure、tests 或任何外层控制平面，也不得保存实验 records、论文构建或模型后端适配。

当前处于 `method_construction_authorized` 的分批实施期。共享层
`main/shared/key_schedule.py` 已进入批次 1 实施与 CPU 验证；其余 12 项职责仍未
实现。单个批次通过不表示完整方法完成，也不得提前创建 method readiness。
