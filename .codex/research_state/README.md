# Research Construction State

此目录只保存构建期阶段门禁读取的外层元数据，不属于研究配置或运行输入。

- `research_definition.yaml` 连接当前十份权威设计和九个冻结方法不变量。
- `method_construction_admission.yaml` 已按模板登记候选规格关闭、独立审计批准、
  用户授权引用和完整 authorization base revision；它只证明合法进入
  `method_construction_authorized`，不证明实现完成。
- 当前为 `experiment_ready / implemented`；此前 construction、method 和 runtime
  阶段转换均不含方法实现混入；既有独立 experiment-stage revision 只同步阶段/status。
- 唯一 `method_readiness.yaml` 已从模板实例化，逐项连接固定路径、候选 ID、
  责任、唯一实现 symbol、27 个非同构行为节点、候选摘要和 revision-bound
  独立语义复核。该 readiness 绑定旧实现的11个候选身份；设计 registry 则为15个
  ID（14个具名候选加1个 mandatory control），二者都不得与13项职责混淆。新增
  显著目标局部 LF 四候选均为 `design_candidate_implementation_authorized`、
  `implementation_admission=YES`；该授权 token 保持不变。四者的 CPU/API source
  implementation 已在 `d88703689a0ea0487ad3a4553d060e5bf1a762cd` 闭合，并由
  `independent_salient_local_lf_experiment_adapter_review:019fed21-be70-7803-aca0-6049bb279dfd:d88703689a0ea0487ad3a4553d060e5bf1a762cd:APPROVE`
  独立审核。候选专属 `salient_local_lf_candidate_readiness.yaml` overlay 只记录四者
  `source_cpu_api_implementation_ready=true`，不把四者写入既有 11-candidate readiness；
  正式 checkpoint/runtime qualification、实验 protocol、masked-LF W、quality 定义、
  科学验证或晋升。
  `content_combination_saliency_max_standardized` 保持 `diagnostic_only=true`、
  `promoted=false`；正式 detector 保持 HF-only，quality gate 尚未定义。
- readiness 只记录方法构建闭合，不是阶段文件。experiment_ready_infrastructure_closure 已完成，冻结实验协议与
  可追溯执行入口可用；这只满足 `experiment_ready` 的执行准备边界，不授权
  calibration、hf_only_reference_validation 晋升、GPU 高成本运行或正式实验。当前 runtime 边界证据绑定
  candidate `8b2344756c4c247906ff0d4eab68e46a773e13f5`、package SHA-256
  `8290abeed79931eb7208ac9ca280f1ea401f4725abfead35f12617a0ef54dd38`、
  qualification run `20260729T110628Z` 和 result ZIP SHA-256
  `d9b7d91d41cc963098c077268445ad80e9994c809227ca2f68615a37ac93ac37`。
  该 `qualification / passed` 结果只支持真实 SD3.5 runtime 边界；正式 detector
  仍为 HF-only，新 masked-LF/routing/组合仅完成 CPU/API source implementation、未实验晋升，
  `full_ceg_wm_eligible=false`，且没有正式
  FPR、鲁棒性、正式 records 或科学效果证据。

current 内容设计固定为无 `a/w/function` selection 的 max statistic；calibration 只
分别承担分支 primary-null/CDF 标准化、max `tau` 及 rescue/geometry/end-to-end
各自职责。historical `a/w/function` 只服务原 producer replay，不进入 readiness。

删除 `.codex/` 后，研究项目必须仍可运行；这里的元数据不能成为方法输入。
