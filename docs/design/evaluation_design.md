# Evaluation Design

HF-only 参考验证依次由 `hf_only_reference_protocol`、
`hf_only_reference_metrics` 和 `hf_only_threshold_fit_gpu_execution` 组成；整体科学门
统一称为 `hf_only_reference_validation`，三者完成度不得互相替代。

## Separation Of Purposes

CEG-WM 使用两个不同实验表面：

1. 内部设计验证：回答 LF 职责、LF/HF 组合、Q/K 几何和联合门控是否成立。
2. 外部方法比较：在项目方法冻结后，与登记 baseline 进行公平比较。

内部设计验证不应被强制伪装成外部 baseline 比较；外部比较也不能替代组件机制验证。

## Splits

至少固定以下互不重叠的 source-cluster manifests：

- development：实现调试，不进入正式结论；
- candidate-selection calibration：只选择 LF/HF、routing 与其他预登记候选；其内部预登记 selection 与 untouched confirmation partitions；
- content-threshold-fit calibration：只拟合冻结内容检测器的 `tau`；
- rescue-window-fit calibration：只拟合 `tau_rescue`；
- geometry-reliability-fit calibration：只拟合几何可靠性规则；
- end-to-end-calibration-check：只检查冻结联合检测器，不再拟合或选择；
- evaluation：冻结后只评估，不调参。

聚类单位由 Prompt、seed、生成图像 lineage 和注册 key family 共同定义；同一 source cluster 的所有攻击、回正、多 key 派生样本或近重复内容必须留在同一职责 manifest。候选选择、阈值、rescue、geometry reliability、end-to-end check 和 evaluation 之间不得泄漏。

## Development Preliminary Exploration

13 项方法职责的首轮科学探索只能读取 `development`。每项职责必须冻结
`responsibility_id`、独立 scientific question/case/candidate/paired-ablation
身份、可重算 candidate config digest、negative controls、metrics、record
fields、dependency stop rule、四项 module outcome 和独立的两项 candidate
recommendation。outcome 只描述 mechanism signal、implementation block 或
resource block；recommendation 不等于晋升，也不授权 candidate-selection。

先执行 2-cluster 环境、身份和吞吐 preflight；8-cluster wiring smoke 也不计入
科学覆盖。冻结顺序为 key schedule；HF carrier 与 HF detector；LF carrier 与 LF
detector；Q/K sync；64 个不计科学覆盖的 routing reference；router、embedder 与
content detector；estimator、reliability 与 rectifier；conditional recovery。
普通职责各覆盖 16 个 source clusters，Q/K sync 与 estimator 各覆盖 32，三个
detector 各覆盖 64，共 384 个核心科学 unit；只有 router、embedder 与 content
detector 各增加 16 个同 cluster paired-ablation unit，因此科学总数精确为 432。
content detector 的 64 个核心 unit 全部使用 routed LF/HF combined，作为完整
cross-fit 输入；source cluster 0--15 另各有一个 HF-only 配对 unit，HF 非退化只在
这 16 个冻结配对 cluster 上比较，不缩减 combined 的 64-cluster cross-fit。
每个核心 scientific unit 只绑定一个职责相关 case，不展开 content branch 与
geometry case 的笛卡尔积：非几何职责只循环本职责分支，几何职责只循环登记的八个
case，conditional recovery 将内容与几何 case 确定性 zip/cycle。连同 74 个操作
unit，冻结 roster 总数为 506；每 unit 最多三次 attempt，科学上限 1296、整体上限
1518。普通 unit 上限 900 秒，8 个 wiring unit 单独为 2100 秒；整体 roster 有
digest/硬上限，禁止按观测分数增删或重排。

routing 的 adaptive/uniform 两臂复用同一 HF-only public content operation；该门
不读取尚未拟合的 LF/HF combined null/CDF，也不替代后续 content combination。
routing reference 只在持久 cursor 到达其冻结局部 phase 时准备；在此之前 key、HF、
LF 与 Q/K unit 必须已经各自形成可验证的 `COMMITTED` 记录。跨会话恢复身份绑定 exact
revision、protocol、config、manifest、roster 与 run；package/bootstrap 的传输摘要不
属于 development 科学恢复身份，session receipt 仍可记录当前 package 的普通 SHA-256。

内容只含 clean、HF-only、LF-only、disabled-uniform LF/HF control 和 routed
LF/HF combination 五个分支。几何只含一个 identity、crop、scale、rotation 和
compound case，另冻结 ambiguous、boundary、extreme-crop 三个 negative-control
case；不展开无约束笛卡尔积。

development provisional threshold 的角色固定为 `development_exploratory`。
四折 cross-fit 只接收 manifest 中 case identity 精确匹配的 development
primary-null score payload；wrong-key 只作独立 attribution control，不参与拟合。
input manifest、detector identity、raw/rectified 共用 preprocessing identity、
primary-null 的 registered/detection public-key roster、冻结 maximum order-statistic
rule 和 fold 均由真实 payload 重算摘要并共同进入 detector config binding；产阈值
clusters 不得评分同折 recovery probes。阈值
在 candidate-selection 前强制作废，不是 formal `tau`，不能支持 fixed-FPR、候选
晋升或科学 claim。

其中 threshold detector authority 由 checked-in development protocol 唯一提供：
职责固定为 HF detector、模式固定为 HF-only、公开预处理固定为
`rgb8_public_image_float32_unit_interval`，并回绑登记的 HF detector candidate/config
和 `main_shared_key_schedule_identify_root_key` key-schedule identity。结果调用者不得
替换 base config 或预处理。每个 primary-null public-key mapping 以 authority、
registered/detection public digests 和 key-schedule config 计算 canonical key-family
digest，并必须逐 cluster 等于 split manifest 已冻结的 registered key-family digest；
不保存 raw secret。protocol、authority、manifest 或 public roster 任一变化都会形成
不同摘要，不能在 threshold 结果侧同源重绑后冒充原实验身份。

在产生任何 development scientific record 前，还必须创建
`FrozenDevelopmentExecutionIntentAuthority`。该 intent 以
`create_only_before_scientific_records` 角色绑定 run/seed namespace、完整 split
manifest 及全部 assignment identities、全 cluster public-key roster、protocol、
detector authority 和 key-schedule authority；raw secret 明确禁止进入。plan、fit
input、binding、threshold 和 recovery authorization 都必须接收调用边界已经 pinned
的 `expected_execution_intent_authority_digest`。完整重建 manifest/roster 可以形成另一个
尚未执行的新 intent，但即使重建全部 records 与下游摘要，也不能替换旧 run 的
expected digest。Batch1 只冻结该协议对象与验证；后续 runner 必须以 create-only
intent 持久化并传入 pinned digest，本批不实现存储。

prompt、source cluster、seed namespace、key family 和 image lineage 五维身份从
各 role 的真实 `FrozenSplitManifest` 与 seed namespace 绑定；任意两个已提供 role
在任一维度有交集即 fail closed。未来 role manifest 在获授权并真实冻结前明确标记
unavailable，不能用 role metadata 的摘要冒充 roster。本阶段只允许 development role；
`candidate_selection_selection` 唯一映射 candidate-selection，content candidate
confirmation 映射 untouched-confirmation/combined，HF-only reference confirmation
映射 untouched-confirmation/HF-only 且要求冻结 HF-only tau，三者本批均不执行。

## Content Validation Matrix

- HF-only；
- LF-only；
- LF/HF routing；
- LF/HF combined；
- route-disabled；
- LF-disabled；
- HF-disabled；
- correct key；
- wrong key；
- unwatermarked negative。

固定权重只能作为明确历史 baseline，不能作为默认项目方法。

## Geometry Validation Matrix

- identity；
- crop；
- scale；
- rotation；
- crop + scale；
- crop + rotation；
- scale + rotation；
- crop + scale + rotation；
- 几何与非几何失真组合。

每类同时报告参数估计误差、可靠率、拒绝率、回正质量、内容检测变化和错误密钥行为。

## Joint Decision Validation

至少比较：

- 原图内容检测；
- 无条件几何恢复；
- 近阈值条件恢复；
- 几何可靠但内容仍未达阈值；
- 几何不可靠；
- oracle transform 诊断上界；
- 回正后同检测器同阈值重判。

需要报告救援成功率、错误救援率、额外 FPR、触发比例和计算成本。

## Metrics

计划指标包括：

- 固定 FPR `0.001` 级别下的 TPR；
- ROC/AUC 作为补充，不替代固定 FPR；
- correct-key 与 wrong-key 分离；
- LF、HF 和组合分数分布；
- rotation、scale、crop/translation 估计误差；
- 几何可靠性校准；
- 图像质量和感知指标；
- 每样本延迟、显存和恢复触发率。

具体指标实现和聚合规则在实验协议阶段冻结。

## Fixed-FPR Target

primary null 定义为无水印生成图像与预登记检测密钥构成的独立检测试验。wrong-key on watermarked image 属于 attribution null，必须单独报告，不能与 primary null 混池。

完整联合检测器的假阳性包括 raw 内容分数越阈值和近阈值样本经几何回正后越过同一阈值。正式 calibration 和 evaluation 必须执行完整联合路径，不能只校准 HF direct score。

每个职责 manifest 必须单独预登记样本量和确定规则：

- candidate selection 由最小相关效应、候选数和预登记选择误差的 power 计算确定；
- content-threshold fit 由目标尾部概率、阈值估计容忍度和 raw 路径误差预算确定；
- rescue-window fit 由预登记增量 TPR/FPR 效应和触发率 power 确定；
- geometry-reliability fit 由变换/key 分层覆盖与可靠性校准精度确定；
- end-to-end check 与 formal evaluation 由联合检测器的单侧置信上界、核心攻击数量和 simultaneous confidence 方案确定。

这些规模必须在访问对应职责数据前冻结。预登记计算可基于 development/pilot 摘要上调样本量；不得根据该职责或 evaluation 结果缩减样本量或改变停止规则。任何单个固定总数都不能同时充当候选选择、阈值、rescue、geometry reliability 或 end-to-end check 的充分性证明。

对 `n` 个独立 primary negatives 中的 `k` 个假阳性，必须报告经验 FPR 和单侧 `95%` Clopper-Pearson 上界。若声明 `FPR <= 0.001`，两者都必须不超过 `0.001`。即使观察到零假阳性，单侧 `95%` 上界达到该要求也至少需要 `2995` 个独立负样本；零误报本身不是充分证据。

多个核心攻击条件同时声称相同 FPR 上界时，必须预登记 family-wise confidence 控制。详细误差预算、样本单位和停止规则见 [research_construction_roadmap.md](research_construction_roadmap.md)。

## Evidence Rules

- 所有阈值和选择规则必须绑定 calibration provenance。
- 失败、排除和资源不足必须显式记录。
- 日志、Notebook 输出、示例图和 harness 报告不能进入正式效果表。
- 当前文件是设计要求，不是实验结果。
