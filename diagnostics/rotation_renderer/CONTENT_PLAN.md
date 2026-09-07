# 内容检出对照：52 路线准备包，尚未执行

本包只准备方案、脚本和轻量测试。没有加载内容模型、执行内容评分、启动 GPU 或推送。
先复用旧八图中的**原顺序前两图 0001/0002**，不按本轮误差挑选，不自动扩大到八图。

## 固定范围与成本

| 输入/条件 | 图数 | 每图评分路线 | 计划评分数 |
|---|---:|---|---:|
| 两图 × CG/G × ±15° × 黑色/反射，固定双线性 | 16 | pre、predicted-H post、oracle | 48 |
| 两图 × CG/G，未攻击参考 | 4 | 原图 | 4 |
| 合计 | 20 | | 52 |

`pre` 是攻击后、校正前的 RGB 评分。`post` 与 oracle 均从该攻击图分别一次恢复后评分，
不串联 warp。预测 H 与 oracle 使用同一生产恢复算子；oracle truth 不进入内容评分器。
历史采样约定不变；oracle 使用历史 H 与恢复采样器配对，不额外叠加像素中心量测补偿。

先运行固定 `0001,+15°,black` 的 CG/G 两图，即六条评分路线 pilot。
看实际耗时、调用数、完整统计和失败后，手动继续剩余 46 条；不自动连续执行。
CG 的八份 H 全部复用已完成的 renderer-v1 原始结果；G 若以后获准执行则需八次独立几何检测。
52 是计划路线数：H 缺失时保留 post 失败行，实际 scorer 调用数和完整统计数另计。
当前 V1 每次完整统计使用注册 key 加 16 个 wrong keys，原路径通常为 34 次 VAE encode；
52 次上界为 1768 次 encode，不启用 V2 评分复用或搜索优化。
真实耗时未测，脚本只在未来 pilot 实测后给剩余 `elapsed × 46/6` 的粗略估计，明确排除初始化。

## 原图、密钥与评分语义

R0 原件 `4f0bf1560805672f786dc86dd50d793aec18aae7/r0-f1` 的 evaluation/.75 原行已直接读取。
CG=`CG_with_content_with_sync`；G=`G_no_content_with_sync`。
**G 含同步水印但不含内容水印**，是配对内容负对照，不是完全无水印，不能代表所有负分布或正式 FPR。

输入根沿用 `/home/richar/projects/CEG-WM/diagnostics/RotationRenderer-Diagnostic-V1/inputs/`。
`r0-result.json`、四个 CG/G 文件与 `content_source_mapping.json` 已核对原 arm/path，未重新生成。
原嵌入 key 的公开身份为
`805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77`；
运行时只接收原密钥，不保存密钥原文。原图身份用于诊断配对，评分接口仍只接收当前 RGB、key、公共资产。

内容调用固定为当前 `cegwm.runtime.blind_detection._score_current_rgb`：
`blind_weighted_scores` → 注册 key 与相同域的 16 wrong keys → `registered W - max(wrong W)`。
保存全部 17 个 W 与完整 m，不沿用旧 R0 的 CG−G 或 G−U 配对 gate 作为当前结论。
旧 R0 paired decision 还要求 candidate W 减 paired-null W 为正；该原图依赖判据不进入本包。
每条强制评分路线也是诊断路线，不冒充包含 pre 提前返回等逻辑的完整生产 `detect_watermark`。

公共资产沿用当前生产构造路径 `experiments.run_blind_detection_v1.build_production_runtime`：

- SD3.5-medium 的同一 VAE / image_processor，生产 LF 与 HF carrier。
- `configs/content_chain/assets/content_v4_clean_null_whitening_operator_v1.json`。
- `configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json`。
- `configs/content_chain/assets/content_v6_iss_gain_target_v1.json`（生产资产结构中的原 ISS 资产；本包不嵌入）。

脚本接收既有 typed production assets，检查实际计算资产语义一致；不自行创建模型。
现有完整 factory 会加载 SD3.5 pipeline 与 DINO，因此若以后需要新建运行时，初始化成本要单独计；
本包没有调用 factory，也不为此新增评分实现或环境门槛。

**主表沿用历史正式 tau = 1.2657276026437319，仅作描述参考**，与冻结 RotationFailure 诊断一致。
仓库工程 N_dev=256 的约 1.13 阈值不替换主参考；若传入资产携带工程阈值，只原样记录并明确分开。
不校准、不调阈值，核心证据为完整 m。不会因几何准确便宣称内容检出或正式协议成功。

## 使用边界与输出

直接 `python -m diagnostics.rotation_renderer.content_plan` 只打印方案，无模型执行。
未来执行须审阅本包后另行批准，再用既有生产资产构造：

```python
session = ContentSession(inputs, geometry_results, new_output, original_key, production_assets)
pilot = session.pilot()  # 六条，检查耗时/完整性
# 审查实际 pilot 后另一个手动步骤：
session.remaining()     # 剩余46条，不重复pilot
```

未来输出为新的 `content-renderer-v1` 目录，保留 `pilot.jsonl`、`remaining.jsonl`、
`paired.csv` 和 `summary.json`。预建脚本不覆盖 renderer-v1 的64行；不自动重试或补样。
每个失败占原计划位置，缺 H 仅使 post 失败，pre 与 oracle 仍独立记录。

黑色填充列为候选攻击协议，反射填充列为 stress 条件。若 CG oracle 两条件均有内容证据，
而 predicted-H 只在黑色下恢复，支持几何估计作为中介；若 oracle 同样差，保留内容/重采样影响解释。
G 的各路线只描述该内容负对照的分数变化。旧正式结果保留，新协议后续明确标识。
V2 暂停，`science_denominator=0`。是否实际运行由控制会话对具体包决定。
