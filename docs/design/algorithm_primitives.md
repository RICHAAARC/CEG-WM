# CEG-WM Algorithm Primitives

runtime 原语按 `runtime_configuration_and_adapter`、`content_write_and_vae`、
`qk_observation`、`runtime_qualification_delivery` 登记；这些语义身份不改变公式、
数值或真实模型验证要求。

## Document Authority

本文档定义 CEG-WM 在进入方法实现前必须冻结或通过设计验证关闭的算法原语。它规定数学职责、输入输出语义、身份绑定、候选选择边界和失败规则，不表示任何原语已经实现、通过真实模型验证或能够支撑论文结论。

本项目的阳性证据只能来自内容检测统计。几何原语只能估计变换、判断可靠性和生成回正图像。

所有可实施候选的精确输入、输出、有限参数集合、历史边界和验收门见 [candidate_specifications.md](candidate_specifications.md)。本文给出原语不变量；候选规格文件关闭实现选择。

## Notation

- `I`：待检测 RGB 图像。
- `K`：检测密钥；正式 records 只保存不可逆密钥身份摘要，不保存原始密钥。
- `M`：冻结的方法配置身份，包含模型、预处理、载体、检测和几何配置。
- `E_M(I)`：由冻结公共模型和预处理定义的图像观测编码。
- `s_hf`：CEG-WM HF direct score。
- `s_lf`：LF 分支分数。
- `D_M(I, K)`：当前正式内容检测器。
- `tau`：由 calibration 数据冻结的内容阳性阈值。
- `tau_rescue`：由 calibration 数据冻结的近阈值负区间下界。
- `g`：几何估计结果。
- `R_M(I, g)`：按冻结规则生成的回正图像。

任何能够改变上述量统计含义的参数都必须进入 `M`。不同 `M` 产生的分数、阈值和几何可靠性不得混用。

## Key Semantics

### Frozen Candidate

所有 LF、HF、Q/K、错误 key、公开 Q/K noise 和 routing probe 都必须使用
`key_schedule_sha256_counter`。该候选已经在
[candidate_specifications.md](candidate_specifications.md) 唯一冻结：

- 非空 Unicode root key 的严格 UTF-8、无 Unicode normalization 语义；
- `stable_json_utf8 → SHA-256 domain digest → uint128 big-endian counter`；
- uniform 高 53-bit open-interval 映射；
- MSB-first 20-bit normal indices 与摘要固定的 `2^20` midpoint float32 quantile table；
- HF、LF、geometry 三个秘密职责域和两个 public-noise 职责域的精确字段集；
- 预登记 wrong-key 派生、仅公开 digest 持久化、golden vectors 与 fail-closed 条件。

实现不得用 PyTorch RNG、Python hash、平台 inverse-CDF、非规范 JSON、运行顺序或
隐式 string/bytes 转换补空。任何密钥协议变化都产生新候选身份，并使载体、几何、
阈值和既有 records 身份失效。

### Key Ownership

正确密钥归属需要同时满足：

- 正确密钥分数相对无水印负样本可校准；
- 正确密钥相对预登记错误密钥具有可观测分离；
- 分支分数和组合分数均保留，组合不得掩盖单分支的错误密钥失败；
- 几何密钥匹配不能替代内容密钥归属。

错误密钥试验用于评价 attribution，不得与无水印图像负样本混池来扩大固定 FPR 的主负样本量。

## HF Carrier And Direct-Score Primitive

### Keyed HF Template

给定冻结形状和 HF 职责域密钥，构造可复现的零均值伪随机张量 `G_hf`。使用冻结低通算子 `P_M` 得到高频剩余：

```text
H_hf = G_hf - P_M(G_hf)
```

本项目的参数级历史迁移候选严格保持历史顺序：先按冻结 tail 规则保留幅值最大的坐标集合，tail 外坐标保持精确零，再直接执行单位二范数归一化：

```text
S_hf = tail_M(H_hf)
T_hf = S_hf / norm(S_hf)
```

模板构造阶段不得对 `S_hf` 中心化，否则 tail 外的零坐标会变为非零并改变稀疏支持。中心化只发生在后述 normalized-correlation 评分阶段。低通核、边界处理、tail 比例、排序平局规则、dtype、形状和归一化顺序都属于方法身份。

若未来要比较 `normalize(center(tail_M(H_hf)))` 或其他顺序，必须登记为新的 CEG-WM HF 候选身份，使用新的配置摘要、校准与证据；它不得继承 historical DirectHF 来源候选的参数级复现或有效性证据。

### HF Carrier Direction

`hf_carrier` 只把 router 的 `mask_hf` 施加到稀疏模板，并输出单位写入方向：

```text
u_hf = normalize(mask_hf * T_hf)
```

HF-only 对照令 `mask_hf = 1`。模板、稀疏支持、mask identity 和单位方向属于
`hf_carrier`；它不产生 `delta_content`，不物化 latent，也不拥有实际写入能量。

### Content Embedder Write

`content_embedder` 独占配置中的冻结 `a`、LF/HF 方向组合、名义相对能量
`content_relative_l2_nominal=3/250` 和 actual-dtype combined content hard limit
`content_relative_l2_limit=3/250`。HF-only、LF-only 和 combined 都由它产生统一的
`delta_content_nominal`；combined 情形为：

```text
gamma_lh = dot(u_lf, u_hf)
v_content(a) = a * u_lf + (1-a) * u_hf
c(a) = ||v_content(a)||_2
u_content(a) = v_content(a) / c(a)
delta_content_nominal =
    content_relative_l2_nominal * norm32(fp32(z0)) * u_content(a)
```

`a` 只能取 `lf_low_pass` 登记的有限集合；`a` 与 `1-a` 是 mixing coefficients，
不是可加的方向份额。HF-only/LF-only 直接使用相应单位方向，但仍
受同一 nominal 与 actual hard limit 约束。任一 active direction 为零、`c(a)`
为零/非有限、nominal formula replay 不符或 `a` 非法都由 `content_embedder`
fail closed。方向、交叉项和 target components 都是 nominal formula witnesses；
它们不定义 actual branch decomposition。

### Runtime Materialization

令 `z0` 为 callback 18 已按登记 binary16 dtype 物化的 baseline。runtime 只在冻结
callback/model/dtype 边界，按 `content_embedder` 请求的正 binary32 scale `s<=1`
逐项物化：

```text
d_s[i]            = f32(f32(delta_content_nominal[i]) * f32(s))
precast[i]        = f32(f32(z0[i]) + d_s[i])
z_s[i]            = binary16_RNE(precast[i])
delta_actual_s[i] = f32(z_s[i]) - f32(z0[i])
```

binary16 转换冻结为 round-to-nearest-ties-to-even；subnormal 按 binary16 规则保留
或舍入为零，overflow、非有限或非法 baseline fail closed。runtime 对 `z_s` 做独立
逐 bit replay，只返回实际张量、`delta_actual_s`、replay identity、完整性状态与
realized 测量，不拥有 budget acceptance、retry 或 scale 选择。

所有 norm 都按 row-major binary32 协议执行：

```text
S_0 = f32(0)
q_i = f32(x_i*x_i)
S_{i+1} = f32(S_i+q_i)
norm32(x) = f32(sqrt(S_n))
L = f32(norm32(fp32(z0)) * f32(3/250))
A = norm32(delta_actual_s)
accept iff A <= L
```

权威 gate 是 `A<=L` 的直接比较。realized relative L2 和
`budget_utilization=A/L` 只作诊断；不得引入比值门、`q_budget`、
`tau_actual_budget`、经验 tolerance 或 actual 强度下限。

`content_embedder` 先请求 `s=1`。若超限，则在 binary32 `[0,1]` 上使用
`f32(f32(f32(lower)+f32(upper))*f32(0.5))` midpoint 二分；actual delta 为零的
点是不可接受 zero plateau，只推进 lower。midpoint 与任一边界 bitwise 相同时终止，
返回最大非零可行 observation；若不存在则 fail closed。不得使用幂次粗回退。

runtime 不得改变 `a`、方向、nominal/limit 或重新分配能量，也不得声称可观测载体级
实际写入分解。需要分支贡献诊断时，只记录 `a`、`gamma_lh`、`c(a)` 与可重建的
nominal target component vectors/norms；它们因交叉项存在而不可加为 total。actual
hard limit 只约束最终 combined content delta；geometry delta 与其独立。低
utilization 不得成为未来实验的结果后排除规则。
注入位置、调度器、剩余生成区间、latent 变换和模型 revision 必须在 runtime 验证前
冻结。

SD3.5、二十步 FlowMatch、callback index 18 与上述 `3/250` nominal/limit 已登记为
当前项目候选身份；该精确 candidate 的真实模型执行已通过独立 runtime/GPU
qualification。任何模型、revision、配置或方法身份变化仍须重新 qualification，
不能由 CPU/mock 或历史结果替代。

### Direct Score

最终图像经过冻结预处理和图像观测编码得到 `Y = E_M(I)`。HF direct score 使用冻结的中心化归一相关形式：

```text
s_hf = dot(center(Y), center(T_hf))
       / (norm(center(Y)) * norm(center(T_hf)))
```

正式检测不得访问生成 latent、注入时模板实例、原始参考图或 embed record。模板只能由检测密钥和公共方法身份重新构造。

HF direct score 在 LF/HF 组合通过晋升门之前是 `D_M` 的唯一原始内容检测统计。

## LF Candidate Primitive

### Candidate Responsibility

LF 的目标职责是：在预登记的 HF 证据易受损条件下提供仍保持密钥特异性的互补内容证据。LF 不得仅检测通用低频偏移、图像内容类别、压缩痕迹或模型身份。

### Keyed LF Template Family

候选 LF 模板从独立职责域的伪随机张量 `G_lf` 构造：

```text
L_lf = P_M(G_lf)
T_lf = normalize(center(L_lf))
```

首个 LF 候选的滤波器、边界、写入位置、同预算组合和分数已在 `lf_low_pass` 中明确；
实验只在其中登记的有限 mixing-coefficient 候选集合内选择。LF 分数使用与其模板和
观测域匹配的冻结归一相关统计：

```text
s_lf = score_lf_M(E_M(I), T_lf)
```

`score_lf_M` 只有在正确密钥、错误密钥、无水印负样本和图像质量证据共同通过后才能冻结。

### LF Rejection Conditions

出现任一情况时，LF 候选不得进入正式内容检测器：

- LF-only 不具备正确密钥与错误密钥分离；
- LF 的无水印 FPR 不能独立校准；
- 增加 LF 后 HF-only 的关键攻击 TPR 或 key attribution 明显下降；
- LF 的增益只能由攻击特定阈值、样本选择或结果后调权获得；
- LF 需要参考图、embed record 或私有嵌入状态；
- 图像质量或可见性代价超过预登记边界。

## Content-Adaptive Routing Primitive

内容路由只输出生成时内容观测结果、空间 mask 和身份，不拥有检测阳性语义、
`a` 或任何能量预算。

候选路由函数记为：

```text
(A, mask_lf, mask_hf, route_identity, observation_digests)
    = route_M(generation_content_observations)
```

路由必须满足：

- 输入观测和计算顺序可复现；
- 嵌入端使用的观测在正式访问模型内合法；
- 检测器若需要重建路由，只能从待检图像、密钥和公共资产获得；
- 若检测器不重建路由，则评分必须证明对嵌入路由具有已验证的统计解释；
- `mask_lf + mask_hf == A` 且 `A` 与两 mask 都在 `[0,1]`；
- route-disabled uniform control 固定 `A=mask_lf=mask_hf=1`，且不读取内容观测；
- LF-only、HF-only、route-disabled 和 combined 都能由 `content_embedder` 在相同
  总预算下消融；
- router 记录 mask 覆盖率和 identity/digests；`content_embedder` 核验 mixing
  coefficients、方向/支持身份、nominal/limit，并在 runtime handshake 核验
  integrity、scale/attempt/budget status 与 realized combined total
  norm/relative L2；
- 路由不得根据攻击标签、evaluation 分数或错误密钥结果改变。

router 不返回 `budget_lf`、`budget_hf` 或其他标量预算。首轮必须同时评估无路由
基线、公开图像观测路由和被允许的共享确定性路由候选。任何依赖私有嵌入状态且
检测端无法复现或边缘化的候选直接失败。

## LF/HF Combination Primitive

### Current Authority

原语责任分为两部分：`content_embedder` 负责 `u_content(a)` 和共同总预算写入；
`content_detector` 负责 `s_lf`、`s_hf` 的冻结标准化与组合。独立 `lf_detector`
必须直接实现盲 `s_lf`，不得把 LF score 隐藏在 carrier 或组合器中。

当前正式 content detector 保持 HF-only。LF/HF 组合是待晋升候选，不是已经冻结的方法。

### Calibrated Candidate Family

每个分支按 `content_combination_calibrated` 的有限样本 mid-rank empirical CDF、
`1/(2n)` 双尾 clipping 和冻结 `2^20` midpoint float32 normal-quantile table
转换为方向一致的 `z_hf` 与
`z_lf`。候选组合限制为该规格登记的 `hf_only_standardized_score/weighted_hf_lf_standardized_score/maximum_hf_lf_standardized_score`，不再允许实现者添加其他
“单调概率或尾部证据组合”：

```text
s_content = combine_M(z_hf, z_lf)
```

candidate-selection 内的 CDF 与 `tau_provisional` 只服务 selection/untouched
confirmation，随后丢弃；正式 branch CDF 与 `tau` 只能由独立
content-threshold-fit 重新拟合。两者不得跨五类 calibration 职责复用。

不得把历史固定权重作为默认候选，不得按攻击类别切换组合函数，不得为回正图拟合第二套权重。

### Promotion Gate

LF/HF 组合只有同时满足以下条件才可以替换 HF-only content detector 成为正式 `D_M`：

- LF 单分支通过 key attribution 和无水印校准门；
- 组合参数只在 calibration 内确定；
- candidate-selection manifest 内预登记且未参与拟合的 confirmation partition 中，至少一个预登记核心攻击族出现统计稳定的增量 TPR；
- identity 和 HF-only 已擅长条件下的退化不超过预登记容忍度；
- end-to-end FPR、错误密钥误归属和图像质量全部满足边界；
- LF、HF 和组合分数在 records 中持续独立保存；
- 原图和回正图同步切换到同一个组合检测器、同一个配置身份和同一个阈值。

若晋升门不通过，LF/路由研究问题可以以 `research_question_closed_negative` 诚实闭合并作为失败或诊断消融报告，但这不构成完整 CEG-WM 方法成功。HF-only 加几何只能在重新命名、重新限定贡献范围并单独取得研究定义授权后成为 reduced-scope 候选；不得继续使用包含内容自适应路由、LF、HF、Q/K 与联合判定的完整 CEG-WM 身份或完成门。

## Q/K Geometry Synchronization Primitive

### Public Observation

几何观测由冻结公共模型在登记层、头、时间条件和预处理下，从普通图像经
image-only empty-condition forward 提取 Q/K：

```text
Q, K_obs = qk_extract_M(I)
```

这里的 `K_obs` 表示 key tensor，不是检测密钥。正式实现必须避免名称歧义，并在接口中使用清晰的 attention key tensor 语义。

### Keyed Relation

从 Q/K 构造对平移、尺度和旋转变化具有可评价响应的关系描述 `A_obs`，再使用几何职责域密钥生成投影或同步目标 `A_key`。关系统计必须绑定真实 Q/K 内容，不得由摘要、随机占位或 embed 缓存替代。

嵌入端目标和盲检端都必须使用同一 VAE posterior mode、公开噪声、schedule index
7、三路空文本、无 CFG 的 image-only forward；生成 conditional Q/K 不属于候选。
四通道、逐 row correlation、两层聚合、geometry-key 对称零对角投影、非正交
content-subspace 投影和 actual-dtype line search 的唯一公式以
`qk_relation_similarity` 为准。几何同步不能独立支撑水印阳性，也不能改变内容分数
定义。

## Transform Estimation Primitive

首个正式变换族是有界 similarity 与 crop 支持：

```text
g = (rotation, scale, translation_x, translation_y, crop_support)
```

候选估计器在冻结搜索域内最大化正确密钥关系目标：

```text
g_hat = argmax_g geometry_objective_M(A_obs, A_key, g)
```

搜索范围、粗搜索、细化、插值、坐标约定和边界处理都属于方法身份。真实攻击参数只能用于 evaluation 误差计算，不能作为估计器输入。

crop 删除的信息不可恢复。回正原语只能恢复可观测坐标，并按冻结规则处理缺失区域；不得使用生成式补全掩盖不可恢复内容。

## Geometry Reliability Primitive

该原语由独立 `geometry_reliability` 组件实现。transform estimator 只能提供原始
估计与诊断指标，不得自行吞并可靠性结果。

可靠性判定是冻结的多条件门，至少消费：

- 最优候选与次优候选的目标间隔；
- 有效覆盖和唯一覆盖；
- anchor 或 pair inlier 比例；
- 残差；
- 估计是否位于支持域边界；
- 正确密钥与错误密钥几何行为；
- 回正图像是否有效。

输出只能是“可靠并允许回正”或带明确原因的 fail-closed 状态。可靠性不能加入内容分数，也不能把内容负样本改为阳性。

## Rectification Primitive

可靠时，使用 `g_hat` 的冻结逆变换和插值规则生成：

```text
I_rectified = R_M(I, g_hat)
```

输出尺寸、颜色空间、量化、插值、padding、crop 缺失区域处理和有效支持掩码必须冻结。回正图像必须重新走完整的 `D_M` 图像输入路径；不得复用原图 latent、原图分支分数或几何内部特征。

## Joint Decision Primitive

联合判定唯一允许的流程为：

```text
s_raw = D_M(I, K)

if s_raw >= tau:
    positive by raw content evidence
elif s_raw < tau_rescue:
    negative without geometry
else:
    estimate geometry
    if geometry is unreliable:
        negative with explicit geometry failure
    else:
        I_rectified = R_M(I, g_hat)
        s_rectified = D_M(I_rectified, K)
        positive only if s_rectified >= tau
```

`tau_rescue` 只决定是否允许付出几何计算，不降低阳性阈值。原图和回正图必须共享完全相同的 `D_M`、密钥语义、预处理和 `tau`。

## Threshold Calibration Primitive

`tau`、`tau_rescue`、几何可靠性门和候选 LF/HF 组合参数必须只从 calibration 范围获得。正式 calibration 必须用互不重叠的 source-cluster manifests 分离：

- LF/HF 或路由候选选择；
- 内容阈值拟合；
- rescue 区间拟合；
- 几何可靠性拟合；
- 完整联合检测器的 calibration 检查。

候选选择不得读取阈值拟合、rescue、几何可靠性或 end-to-end check 的样本；四类拟合/check 也不得反向选择候选。聚类单位是同一 Prompt、seed、生成图像 lineage 与注册 key family 形成的 source cluster，其所有攻击、回正和多 key 派生样本必须留在同一职责 manifest。每类样本量由预登记的 power/sample-size 或尾部置信计算确定，禁止以一个混合固定总数替代独立职责规模。最终 evaluation 不得再选择参数。任何检测器身份变化都使旧阈值失效，必须重新校准。

## Failure Semantics

实现和协议必须显式区分：

- 内容编码失败；
- 分支模板或分数无效；
- 路由不可重建或覆盖退化；
- 几何未触发；
- 未发现同步；
- 变换不可辨识或多解；
- 估计超出支持域；
- 回正无效；
- 回正完成但内容仍为负；
- 运行或资源失败。

失败不得从分母中静默移除，也不得通过 fallback、跳过、降阈值或替换检测器制造通过。

## Closed Candidate Choices And Open Evidence

以下实现选择已经关闭为具名候选，实施时不得自行补空：

- `key_schedule_sha256_counter` 固定 root key、serialization、KDF/PRG、职责域、
  wrong-key/public-noise 与 golden vector；
- `runtime_sd35_flowmatch` 固定 backbone、model revision、scheduler、steps、dtype、VAE 和 runtime contract；
- `hf_sparse_tail` 固定 CEG-WM 首个 HF 模板/方向顺序、embedder 使用和 runtime
  物化边界；其 sparse-tail 顺序具有 historical DirectHF 来源，但候选身份和证据
  不继承历史名称；
- `lf_low_pass` 固定 LF template/direction、embedder 使用、score 与有限 `a` 集合；
- `routing_stqr` 固定 S/T/R/Q observations 和路由公式；
- `content_combination_calibrated` 固定有限组合函数集合；
- `qk_relation_similarity` 固定 Q/K 层、直接 relation、keyed objective 和同步写入；
- `rectification_similarity` 固定搜索域、目标、可靠性指标和回正规则；
- `joint_conditional_recovery` 固定联合判定。

registry 共 10 个 ID：9 个具名候选和 1 个强制同预算禁用对照
`routing_uniform_control`。CPU/synthetic 实现不等于实验晋升；该计数不等于固定的
13 项实现职责，也不把对照视为方法候选。

CPU/synthetic 实现与方法行为门已经闭合。仍开放的是候选能否通过真实 runtime、
candidate-selection、calibration 和 formal evaluation 门，以及由互斥 calibration
职责拟合的阈值数值。这些是证据结果，不是实现空白。

13 项职责、27 个 CPU/synthetic 行为节点和唯一 readiness 已完成并经独立语义
审计。实际 stage/status 已由独立 revisions 同步为
`experiment_ready / implemented`。冻结 SD3.5 runtime 已通过真实 GPU
qualification；正式 detector 保持 HF-only，LF/routing 尚未实验晋升，
`full_ceg_wm_eligible=false`。实验准备基础设施闭环不提供 `tau`、confirmation
结果、Calibration Locked、正式 evaluation、正式 FPR 或效果证据，也不晋升
LF/routing/组合/geometry。
