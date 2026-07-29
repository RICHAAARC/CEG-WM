# Research Construction State

此目录只保存构建期阶段门禁读取的外层元数据，不属于研究配置或运行输入。

- `research_definition.yaml` 连接当前十份权威设计和九个冻结方法不变量。
- `method_construction_admission.yaml` 已按模板登记候选规格关闭、独立审计批准、
  用户授权引用和完整 authorization base revision；它只证明合法进入
  `method_construction_authorized`，不证明实现完成。
- 当前为 `runtime_verified / implemented`；此前 construction 和 method 阶段转换
  均不含方法实现混入；本次独立 revision 也只同步 runtime 阶段/status。
- 唯一 `method_readiness.yaml` 已从模板实例化，逐项连接固定路径、候选 ID、
  责任、唯一实现 symbol、27 个非同构行为节点、候选摘要和 revision-bound
  独立语义复核。10 个候选 ID 与 13 项职责不是同一计数。
- readiness 只记录方法构建闭合，不是阶段文件。当前 runtime 边界证据绑定
  candidate `8b2344756c4c247906ff0d4eab68e46a773e13f5`、package SHA-256
  `8290abeed79931eb7208ac9ca280f1ea401f4725abfead35f12617a0ef54dd38`、
  qualification run `20260729T110628Z` 和 result ZIP SHA-256
  `d9b7d91d41cc963098c077268445ad80e9994c809227ca2f68615a37ac93ac37`。
  该 `qualification / passed` 结果只支持真实 SD3.5 runtime 边界；正式 detector
  仍为 HF-only，LF/routing 未实验晋升，`full_ceg_wm_eligible=false`，且没有正式
  FPR、鲁棒性或科学效果证据。

删除 `.codex/` 后，研究项目必须仍可运行；这里的元数据不能成为方法输入。
