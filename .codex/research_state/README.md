# Research Construction State

此目录只保存构建期阶段门禁读取的外层元数据，不属于研究配置或运行输入。

- `research_definition.yaml` 连接当前十份权威设计和九个冻结方法不变量。
- `method_construction_admission.yaml` 已按模板登记候选规格关闭、独立审计批准、
  用户授权引用和完整 authorization base revision；它只证明合法进入
  `method_construction_authorized`，不证明实现完成。
- 当前为 `experiment_ready / implemented`；此前 construction、method 和 runtime
  阶段转换均不含方法实现混入；既有独立 experiment-stage revision 只同步阶段/status。
- 唯一 `method_readiness.yaml` 已从模板实例化，逐项连接固定路径、候选 ID、
  责任、唯一实现 symbol、17 个非同构行为节点、候选摘要和 revision-bound
  独立语义复核。当前 readiness 在13项职责上绑定12个唯一候选身份；设计 registry
  则为20个 ID（19个具名候选加1个 mandatory control），二者不得混淆。语义—纹理
  软路由五候选的当前 reviewed revision 为
  `cd541e5fa7ffeabc1db1f74a3e9f5a925e0112d9`，并经独立 exact audit 批准，但状态仅为
  `implemented_not_scientifically_validated`。旧的11候选/28节点 readiness 快照仍在
  exact revision `0258ccb2100bfe8b58d1a12079876841192528b3` 保留为历史事实。
  `02fe5dcc2b74482c9eb1e0b192b4a2ce79e0d9eb` 只保留为 semantic-domain 变更前的
  历史 exact provenance，不是当前 readiness 权威；completion profile 为
  `profile_pending`。
  hard salient-object local-LF 四候选已由方法设计替代，但不据此作科学失败裁决。
- readiness 只记录方法构建闭合，不是阶段文件。experiment_ready_infrastructure_closure 已完成，冻结实验协议与
  可追溯执行入口可用；这只满足 `experiment_ready` 的执行准备边界，不授权
  calibration、hf_only_reference_validation 晋升、GPU 高成本运行或正式实验。既有
  runtime qualification 只保留为原 producer/revision 上的历史窄证据，不是当前
  method/runtime/delivery authority。正式 detector
  仍为 HF-only，soft-route max 只作 diagnostic、未实验晋升且没有 formal threshold，
  `full_ceg_wm_eligible=false`，且没有正式
  FPR、鲁棒性、正式 records 或科学效果证据。

current 内容设计固定为无 `a/w/function` selection 的 max statistic；calibration 只
分别承担分支 primary-null/CDF 标准化、max `tau` 及 rescue/geometry/end-to-end
各自职责。historical `a/w/function` 只服务原 producer replay，不进入 readiness。

删除 `.codex/` 后，研究项目必须仍可运行；这里的元数据不能成为方法输入。
