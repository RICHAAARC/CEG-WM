# Core Watermark Method Package

`main/` 是 CEG-WM 可独立抽离的最小方法包。计划包含 `shared/`、`content_chain/`、
`geometry_chain/`、`joint_decision/` 和公开 API。未来实施固定为 13 项职责；
`content_chain` 中的 embedder、LF detector 与 `geometry_chain` 中的 reliability
都有独立路径，不得折叠到 carrier、content detector 或 transform estimator。

此层不得导入 runtime、experiments、paper artifacts、notebooks、infrastructure、tests 或任何外层控制平面，也不得保存实验 records、论文构建或模型后端适配。

当前尚无方法或占位实现；实质 Python 定义只能在合法构建阶段且取得单独后续授权的
revision 中加入。目录与空入口不表示任何组件或能力已经完成。
