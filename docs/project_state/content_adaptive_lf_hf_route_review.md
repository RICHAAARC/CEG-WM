# Content-Adaptive LF/HF Route Review

## Status And Authority Boundary

本文件记录 2026-08-20 的研究路线判断，不是 `docs/design` 方法权威，不修改
既有 mechanism-validation 协议、不授权实现或 GPU 运行。该路线已被收敛为
`docs/design/candidate_specifications.md` 中的有限登记设计；本文件只维持来源、负结果
边界与实施授权隔离。后续仍必须另行授权，并在任何数据或调参前绑定新的
revision、资产、split 和 detector identity。

当前 exact semantic-texture soft-route mechanism-validation candidate selection 已形成
`authenticated_development_negative`。权威 run、固定分母和证据边界见
[method_route_registry.md](method_route_registry.md)。该结果关闭的是当前五候选的
合取，不是“内容自适应 LF/HF”研究主题本身。

## What The Negative Result Does And Does Not Close

已被否定的 exact 假设：

- 当前仅依靠 `M/T` 的空间调制、逐分支归一化、共同总方向归一化和固定 soft-max
  统计，不能按已冻结 mechanism-validation 门形成可晋升内容候选；
- 当前 soft-routed combined max 检测身份没有满足冻结的 identity wrong-key 归因门；
- 当前候选也没有满足预登记 raw-crop 比较门，不能以未来几何链作事后豁免；
- 因而当前 `W`、CDF、provisional `tau`、selection roster 和效果观测不得迁入新候选。

仍开放的研究假设：

- 语义、纹理和边缘可以作为公开可重建的内容先验，但必须与直接反映局部生成路径
  保持度或扰动响应的稳定性信号共同验证；
- LF 与 HF 可能在攻击族上互补，但主辅关系必须由正确密钥、错误密钥、primary null、
  攻击负样本和固定 FPR 证据决定；
- 内容自适应可以同时控制空间支持和受约束分支分配，而不必停留在只改变方向/落点的
  调制；
- oracle rectification 可作为内容证据仍存时的诊断上界，actual Q/K geometry 则必须
  在独立阶段验证能否接近该上界；两者都不能修复已删除内容或内容链 key attribution
  失败。

## Literature Transfer Assessment

这些工作只作为研究启发，不提供 CEG-WM 方法、资产、阈值或效果证据。

| Work | 可借鉴机制 | 对 CEG-WM 的合理迁移 | 不能直接迁移的部分 |
| --- | --- | --- | --- |
| ARIW-Framework | 以图像梯度形成逐像素强度图，并行攻击残差优化 | 将纹理/梯度从“落点先验”提升为受硬总预算约束的局部强度控制输入；保留 route-disabled 因果对照 | 训练式图像域 encoder/decoder、攻击条件残差权重和 bit extraction 不能替代盲 key attribution |
| FARW | 多尺度频域注意力同时形成空间 mask 与通道/频率权重 | 设计真正可观测、可约束的 LF/HF 分支分配，而不是把 `m_lf/m_hf` 误写成实际能量 | 端到端学习的 mask、noise-layer 效果和 bit accuracy 不能直接证明潜空间载体或固定 FPR |
| ILFA | smooth/texture/edge 区域使用不同强度；局部冗余与独立同步处理不规则裁剪 | 引入 edge/region survival 支持和分散式重复证据；把 raw crop、oracle 回正和 actual geometry 分开诊断 | hard 区域阈值、图像域对称模板、私有/传输 pattern 及 SLCA 水印定位不能进入当前盲检测边界 |
| Adaptive weighted U-Net | 按内容学习跨层融合权重，以 host prior 恢复细节 | 可启发一个受界、冻结、攻击无关的分支控制器及质量约束 | U-Net skip 融合权重不是 LF/HF 水印能量，也不天然提供 key specificity 或公开重建性 |
| UniMark | key-dependent、位置相关的语义近邻替换；块级冗余和显式统计检测 | 强调“语义自适应必须与密钥归属同时设计”，并启发 crop 下的分散证据/纠错布局 | 自回归 codebook token 替换、multi-bit payload 和其解析 FPR 不适用于 SD3.5 潜空间零比特 detector |

来源身份：

- Wu, Zeng, Lu, “ARIW-Framework: Adaptive Robust Iterative Watermarking Framework,”
  AAAI-26, DOI `10.1609/aaai.v40i42.40901`；
- Zhang et al., “Frequency-domain attention-guided adaptive robust watermarking model,”
  Journal of the Franklin Institute 362 (2025) 107511, DOI
  `10.1016/j.jfranklin.2025.107511`；
- Yang et al., “Robust image watermarking towards iPhone intelligent matting and social
  platform sharing,” Knowledge-Based Systems 326 (2025) 114029, DOI
  `10.1016/j.knosys.2025.114029`；
- Zou et al., “An attack-resilient Unet watermarking framework for copyright protection
  via adaptive weighting and resolution recovery,” Signal Processing 246 (2026) 110609,
  DOI `10.1016/j.sigpro.2026.110609`；
- Yilmaz et al., “UniMark: Unified Adaptive Multi-bit Watermarking for Autoregressive
  Image Generators,” arXiv:`2604.11843v1`。UniMark 是预印本，其实验结论不按同行评审
  论文处理。

## Adopted Unimplemented Candidate Route

已登记路线不是直接加入 geometry，也不是回退为 HF-only。有限候选依赖分为：

1. 先修复内容归因底座。冻结 HF 作为高特异性 attribution anchor；为 LF 建立新的
   carrier/detector 或 target-vs-impostor key-contrastive score。此级暂时禁用 adaptive
   route，只验证 identity 与非几何攻击下的 correct-key/wrong-key、primary null、攻击
   负样本和质量。不得搜索固定 LF/HF 混合权重。
2. 归因底座通过后，再加入公开可重建的语义 `M`、纹理/边缘 `T/E` 和局部分支稳定性
   `S_lf/S_hf`。控制对象必须同时包括空间支持与硬总预算内的受界 LF/HF 分配；检测端
   必须能从普通 RGB8 重建同一控制规则。route-disabled 使用相同 carrier、detector、
   总预算和样本形成因果对照。

`S_lf/S_hf`、spatial `q`、nominal `rho`、orthogonal write、combined detector、
fresh split 和唯一 conditional controller 均已在
[candidate_specifications.md](../design/candidate_specifications.md) 冻结。本文件不复制
公式，也不把这些 `adopted_design_unimplemented / not_yet_tested` 身份改写为
implemented 或 approved-for-run。公式仍禁止读取 attack label、embed record、reference、
private latent 或测试结果进行 runtime switching。

## Stage Order, Success And Stop Conditions

### Stage A: Branch Attribution Foundation

验证内容：HF-only 与新 LF-only 在 identity、JPEG/blur/noise 等非几何攻击下的
correct-key/wrong-key、primary null、攻击负样本、质量和预算。

- 成功：两分支均满足预登记 key-attribution/FPR 门，LF 至少在一个预登记非几何攻击族
  对 HF 提供互补 margin，进入 Stage B。
- 失败：若 LF 归因失败，停止该 LF identity，修改 carrier/detector 后建立新候选；
  不进入路由、组合或 geometry。若 HF anchor 失败，先修复 HF。

### Stage B: Adaptive LF/HF Causal Mechanism

验证内容：固定总预算下比较 adaptive route、route-disabled、HF-only、LF-only；分别
记录 LF/HF 正确密钥裕量、impostor 裕量、null/FPR 和质量。identity 与非几何攻击是
内容链硬门。

- 成功：adaptive route 相对 route-disabled 产生预登记因果增益，不退化 HF anchor，
  且不恶化 primary-null、wrong-key 和攻击负样本门，进入 Stage C。
- 失败：停止该 controller identity。若分支本身仍合格，只修改稳定性输入或受界分配
  机制并建立新候选；不得回退固定权重后继续使用同一身份。

### Stage C: Crop Fault Decomposition

验证内容：同一内容 detector 依次测 raw crop 与已知变换的 oracle rectification。
raw crop 记录删除、失同步和重采样的总损失；oracle 只给出坐标恢复后的可达上界。

- 成功：oracle 相对 raw 显著恢复且内容归因/FPR 仍合格，说明 geometry 有可达收益，
  进入 Stage D。
- 失败：若 oracle 仍失败，返回 carrier、冗余布局或 detector；geometry 不能救回。

### Stage D: Actual Q/K Geometry

验证内容：identity、crop、scale、rotation 与组合攻击下的估计误差、coverage、reliability、
wrong-key、攻击负样本、回正质量，以及相对 oracle 的恢复差距。

- 成功：可靠回正接近预登记 oracle 上界且 fail-closed，不增加内容阳性权，进入 Stage E。
- 失败：停止该 Q/K/estimator/reliability identity，返回几何链；不修改内容阈值补偿。

### Stage E: Joint Detector And Formal Evidence

冻结 raw 与 rectified 共用的同一内容 detector、key semantics、preprocessing 和 `tau`；
只有近阈值负样本且 geometry 可靠时回正，随后同阈值重判。之后再按互斥职责拟合 formal
content threshold、rescue window 和 geometry reliability，并完成完整 raw+rescue
fixed-FPR evaluation。

- 成功：内容自适应因果门、LF/HF attribution、geometry、联合增益、质量与完整 FPR
  全部通过，才可形成完整 CEG-WM 论文贡献。
- 失败：按首个失败门回到对应新候选；不得把 HF-only + geometry 写成完整
  “内容自适应 LF/HF”成功。

## Existing Mechanism-Validation Raw-Crop Decision

当前已完成的 mechanism-validation 不修改，也不豁免，其 raw-crop 失败仍属于当前 exact 候选负结果。

已登记设计把 identity/非几何 attribution、null/wrong、route-disabled 因果证据、
raw/oracle crop 分解和 actual Q/K 的后续依赖隔离。实施协议仍必须在任何新数据、调参
和结果观察前以独立授权冻结，使用与旧路线零交叉的 split/manifest，且不继承
当前 selection 资产。这是前置设计闭合，不是对当前失败的事后改门。

## Immediate Decision

本次只完成登记设计闭合。下一项只能是独立授权的实施准入，不得直接做
Q/K geometry、formal calibration、soft-max promotion、GPU 运行或重跑当前
mechanism-validation。本状态文件不授权 implementation、config、manifest、
Notebook、package、GPU 或 Colab。
