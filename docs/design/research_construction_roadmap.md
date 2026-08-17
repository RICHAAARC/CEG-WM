# CEG-WM Research Construction Roadmap

## Purpose

本路线图定义方法从规格、实现、机制验证、校准到论文证据的依赖顺序。

## Method Success Conjunction

完整 CEG-WM 成功是下列硬门的合取：

```text
G_full = G_route
       and G_lf
       and G_hf
       and G_content
       and G_qk
       and G_geometry
       and G_joint
       and G_quality_fpr
```

不得以“通过模块数 / 模块总数”计算完整方法成功率。任何必需门失败都关闭完整方法
成功身份；缩减方法必须重新命名、重新限定主张并独立校准/评估。

## Gate 1: Research And Candidate Specification

### Required Work

- 冻结内容证据唯一阳性权；
- 冻结盲检测边界和同 detector/key/preprocessing/threshold 回正重判；
- 冻结 root key、KDF/PRG、LF/HF carrier、语义—纹理软路由、组合统计、Q/K relation、
  变换支持域、可靠性和回正；
- 规定所有输入、输出、身份、失败语义和 falsification gate；
- 把历史项目限定为 provenance，不继承固定权重、私有状态或效果证据。

### Pass Condition

所有候选有限、可实施、可证伪，没有实现时补空、reference image、embed record、
private latent、geometry direct positive 或 attack-conditioned detector。

## Gate 2: Implementation And API Readiness

### Required Components

```text
shared:    key_schedule
content:   content_router, lf_carrier, hf_carrier, content_embedder,
           lf_detector, hf_detector, content_detector
geometry:  qk_geometry_sync, geometric_transform_estimator,
           geometry_reliability, image_rectifier
joint:     conditional_recovery_decision
```

### Required Checks

- 13 项职责具有不同真实 symbol，不能折叠为代理；
- key/domain/golden、carrier、route、combined budget、blind scores、Q/K relation、
  transform、reliability、rectification 和 same-detector joint flow 都有数据依赖检查；
- research code 不依赖 governance，内容链和几何链不互相导入；
- CPU/synthetic 检查只证明结构和数学行为，不冒充 runtime 或科学效果。

## Gate 3: Runtime Qualification

### Required Work

- 记录 model/repository/name/revision/checkpoint/source 与 Python/CUDA/GPU/dependency
  版本 locator/observed metadata，并锁定 behavior-changing pipeline/scheduler
  capability、steps、callback、VAE、dtype、依赖/API/resource capability、科学资产和
  implementation revision；
- 验证 clean/watermarked paired trajectory、actual-dtype content materialization、
  binary32 `3/250` hard budget 和 image decode/encode；
- 捕获登记层真实 `to_q/to_k`；
- 验证 registered/wrong/public key-domain determinism；
- 明确资源、身份、完整性、非有限量和恢复失败。

### Pass Condition

冻结 runtime 身份可执行且失败可审计。该门不证明路由、LF/HF 检测、几何恢复或固定
FPR 的科学有效性。

## Gate 4: Content Route And Carrier Mechanism

### Step 4.1: Public Route Reconstruction

- embed、raw detect、rectified detect 分别从当前 RGB8 重建 `M/T`；
- `M` 使用冻结 InSPyReNet soft probability；
- `T` 使用冻结 grayscale Sobel/area/P95 映射；
- 验证 `m_hf=(1+M*T)/(2+M)`、`m_lf=(1+M*(1-T))/(2+M)` 的确定性、正值与和为一；
- route-disabled control 固定 `m_hf=m_lf=0.5` 且不读取 `M/T`。

### Step 4.2: Same-Budget Causal Write

- clean、HF-only、LF-only、soft-routed LF/HF、route-disabled 和失败保留固定分母；
- 每个写入都使用相同 key/Prompt/seed/callback 和 combined `3/250` hard limit；
- 验证 active directions 非零、combined formula、actual-dtype integrity 和质量；
- 不把 route map 或 nominal components解释为 actual branch energy。

### Step 4.3: Blind Branch Attribution

- HF 与 LF 检测端从普通 RGB8 和 key 独立重建 route/template；
- soft-routed LF 使用专属 32-clean-null whitening fit；
- registered、wrong-key、unwatermarked primary null 共用同一 detector identity；
- 两分支分别通过 key attribution、primary-null calibration 和质量门。

### Step 4.4: Fixed Content Statistic

- 分别标准化为 `z_hf_soft`、`z_lf_soft`；
- 组合只计算 `max(z_hf_soft,z_lf_soft)`；
- 不搜索权重、函数或 attack-conditioned switch；
- 在 untouched confirmation 中验证增量 TPR、HF 非退化、wrong-key、FPR 和质量。

### Content Gate Outcomes

- 全部通过：冻结内容机制候选并进入 Gate 5；此时不得拟合 formal `tau`、
  `tau_rescue` 或 formal geometry reliability；
- 任一必需机制失败：保存固定分母负结果，停止完整方法晋升；
- 证据不足：增加样本只允许按预登记规则，不得调参后沿用原确认集。

## Gate 5: Q/K Geometry Mechanism

### Step 5.1: Synchronization Write

- 从真实 Q/K 构造冻结 relation；
- geometry key projection 与 content key domain 分离；
- 写入改善 correct-key relation，不改善 wrong-key relation；
- actual-dtype geometry/total budget、content-subspace projection 和质量同时通过。

### Step 5.2: Bounded Transform Estimation

覆盖 identity、crop、scale、rotation 及其组合。estimator 只输出 best/second、
coverage、uniqueness、gap、key margin、inlier、residual、boundary 和 identity margin，
不输出阳性。

### Step 5.3: Reliability And Rectification

- reliability 独立执行冻结合取门；
- wrong-key、多峰、低 coverage、高 residual、边界和非有限量 fail closed；
- 可靠时按冻结 inverse warp 回正；
- crop 删除信息不做生成式补全；
- 回正价值只能由同一个内容检测器的变化衡量。

### Geometry Gate Outcomes

- 全部通过：冻结 Q/K、transform、reliability procedure 与 rectification 候选，进入
  Gate 6；此时只允许保留机制 selection/confirmation 参数，不得把它们重签为
  formal calibration artifact；
- Q/K、estimator、reliability 或 rectification 任一必需机制失败：保存固定分母负
  结果，停止依赖该机制的完整方法路径；
- 实现或 runtime 身份失败：返回 Gate 2 或 Gate 3，不得解释为科学负结果。

## Gate 6: Joint Detector

冻结唯一流程：

其中当前 `D_M` 唯一指现有 HF-only detector/config identity 与既有 threshold，不指向
尚未校准的 semantic-texture soft max。

```text
s_raw = D_M(image, key)

if s_raw >= tau:
    positive_by_raw_content
elif s_raw < tau_rescue:
    negative_without_geometry
else:
    estimate_and_check_geometry
    if unreliable:
        negative_with_geometry_failure
    else:
        rectified = rectify(image)
        s_rectified = D_M(rectified, key)
        positive iff s_rectified >= tau
```

### Pass Condition

- raw/rectified 使用现有同一 HF-only detector/config identity、key semantics、
  preprocessing 和既有 `tau`；
- 远负样本不调用几何；
- 几何失败保留内容负判定；
- geometry confidence 不进入阳性统计；
- conditional recovery 提供预登记 TPR 增益且完整 raw+rescue FPR 不超预算。

Gate 6 冻结的是联合控制流、raw/rectified detector 同一性、拟合职责和 artifact
schema；当前 formal/default/joint `D_M` 继续使用既有 HF-only `tau`。未来 soft-max
候选自己的 `tau`、`tau_rescue` 与 formal reliability 数值仍保持未拟合。只有 Gate 4、
Gate 5 和 Gate 6 的算法身份均冻结后，才允许进入 Gate 7 的职责分离校准。

## Gate 7: Calibration Separation

Gate 4 的 route/content candidate selection、soft-LF 专属 whitening `W` fit、
provisional branch CDF 和 Gate 5 的 provisional reliability selection 只服务各自机制
确认；它们不得被重签为 formal calibration artifact。Gate 6 冻结联合算法身份后，
互不重叠的正式职责依次为：

1. `content_threshold_fit`：只读消费 Gate 4 已冻结的 `W`，拟合 formal branch CDF
   和 content `tau`，不得重新拟合 whitening；
2. `rescue_window_fit`：拟合 `tau_rescue`；
3. `geometry_reliability_fit`：拟合 formal geometry reliability；
4. `end_to_end_calibration_check`：只检查冻结联合路径，不再拟合；
5. `formal evaluation`：只评估，不调参。

当前 formal/default/joint `D_M` 仍为现有 HF-only detector/config identity 并使用既有
threshold。语义—纹理
`max(z_hf_soft,z_lf_soft)` 只是未校准、未科学验证、未晋升的 diagnostic；它没有
formal decision。未来晋升必须重新完成独立分支 calibration、max threshold fit、固定
FPR/科学确认和显式 promotion，并以未来独立新拟合的 W/CDF/`tau` 为 raw/rectified
共享新的 detector/config identity 与新阈值；不得继承旧 W/CDF、`tau` 或 HF-only
threshold。完成这些门前 soft max 不得替换现有正式/default/joint `D_M`。

同一 source cluster 的所有攻击、回正和多 key 派生样本留在同一职责。任何 detector
identity 变化都使旧 threshold 失效。

## Gate 8: Fixed-FPR Statistical Design

primary null 为独立的：

```text
unwatermarked generated image + preregistered detection key
```

wrong-key null 单独报告。完整联合 FPR 同时计入 raw 阳性和 rescue 后阳性。

对 `n` 个独立 primary negatives 的 `k` 个假阳性，报告经验 FPR 和单侧 `95%`
Clopper-Pearson 上界。声明 `FPR <= 0.001` 时二者都必须不超过 `0.001`。零误报、单
条件、无聚类情况下的数学下限为：

```text
n >= ceil(log(0.05) / log(0.999)) = 2995
```

多攻击条件需要 simultaneous confidence；样本规模还需由聚类、tail tolerance、
最小相关效应和 power 预登记。

## Gate 9: Formal Evaluation

### Core Matrix

- identity positives 与 primary negatives；
- correct-key/wrong-key attribution；
- fixed-FPR TPR；
- route、LF、HF、combined 消融；
- raw-only、geometry-always diagnostic、conditional recovery 和 oracle upper bound；
- image quality、latency、GPU memory、trigger/failure rate。

### Attack Matrix

- JPEG/compression、blur、resize、noise、color/brightness/contrast；
- crop、scale、rotation 及其组合；
- 几何与非几何组合；
- 独立预登记的生成式攻击与自适应攻击。

evaluation 全程不得修改方法、阈值、attack 或排除规则。所有 method/runtime/resource
失败保留在分母。

## Gate 10: Paper Evidence Closure

- records 是事实来源；
- tables/figures/reports 从 frozen records 和 manifests 重建；
- 每项 supported claim 映射到 exact artifact、sample scope、threshold、metric、
  uncertainty 和 revision；
- 独立复算 FPR、TPR、置信区间和主要消融；
- unsupported、insufficient-evidence、negative 和 reduced-scope 结论明确区分；
- release 不包含原始密钥、private path、cache 或非权威历史输出。

## Stop And Return Rules

以下情况返回对应上游门并建立新的方法/协议身份：

- content route 不能从普通 RGB8 盲重建；
- LF 或 HF 不具备独立 key attribution；
- max statistic 掩盖 wrong-key 或降低 HF-only；
- Q/K 不可稳定观察或 synchronization write 无 correct-key-specific gain；
- geometry reliability 不能 fail closed；
- rectification 需要 reference/private embed state；
- raw/rectified detector 或 threshold 不同；
- end-to-end FPR 超预算；
- calibration/evaluation 泄漏；
- 正式运行中修改方法或协议；
- 样本量不足；
- artifacts 不能从 frozen records 重建。

返回不得覆盖旧记录，也不得通过降阈值、放宽预算、删失败样本或后续复杂度掩盖根因。
