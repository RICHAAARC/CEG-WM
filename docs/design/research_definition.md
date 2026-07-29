# CEG-WM Research Definition

## Research Objective

CEG-WM 研究生成式图像中的密钥归属水印。目标是在常规内容失真和 crop、scale、rotation 等几何变换后，以受校准的内容检测统计判断水印是否属于给定密钥，并在必要时通过独立几何链恢复检测坐标。

受保护属性是“给定图像是否携带给定密钥对应的项目水印”，不是图像来源平台、用户身份、payload 完整性或事件级 attestation。

## Generated Medium And Model Assumptions

- 研究介质是生成式图像。
- 生成与检测所用模型族必须能够提供 CEG-WM HF carrier/direct score 所需能力以及可复现的 Q/K 观测。
- 首个 backbone/runtime 候选已限定为 `candidate_specifications.md` 的
  `runtime_sd35_flowmatch`，包含具体 model revision、调度器、精度、尺寸、steps、
  VAE 与依赖锁。该候选的真实 SD3.5 runtime 边界已由独立审核的 qualification
  结果复现；这不把模型可执行性扩展为方法效果。未来候选失败或身份变化时必须先
  修订规格并重新 qualification，不能在实现中静默换模型。
- 项目方法 API 不得依赖具体设备、远程服务或 Notebook 状态。

## Access Model

嵌入和特征构建允许访问受控生成模型内部组件。正式检测输入限定为：

- 待检测图像；
- 检测密钥；
- 冻结且可公开复现的方法与模型资产；
- 预先校准且与协议绑定的阈值。

正式检测不得读取原始参考图、嵌入端 record、嵌入 latent、嵌入端 Q/K 缓存或其他私有生成状态。

root key 的 UTF-8 语义、职责域派生、wrong-key 和 public-noise 由
`key_schedule_sha256_counter` 统一定义；正式 records 只保存不可逆 public digest。

## Attacker Capabilities

设计验证至少覆盖：

- crop、scale、rotation 及其组合；
- 压缩、噪声、模糊、颜色和亮度等非几何失真；
- 几何与非几何失真的组合；
- 错误密钥检测；
- 自适应攻击在独立协议中登记后执行。

攻击参数、顺序、随机性、失败和排除规则必须在运行前固定，不能按结果选择。

## Detector Output

正式检测输出至少应区分：

- 内容原始分数、校准阈值和 margin；
- LF/HF 分支可观测统计与内容路由；
- 是否满足近阈值救援资格；
- 几何估计、可靠性和失败原因；
- 是否执行回正；
- 回正后由同一检测器产生的分数；
- 最终内容判定。

当前 CPU/synthetic 方法结果已覆盖其中的方法级分支统计、门控、几何、回正和判定
身份；正式 calibration 阈值与持久化 records 字段仍是未来协议要求。

当前方法实现按 13 项正式职责分层：共享 key schedule；内容链的 router、LF/HF
carrier、content embedder、LF/HF detector、content detector；几何链的 Q/K sync、
transform estimator、独立 geometry reliability、rectifier；以及 conditional recovery
decision。27 个 CPU/synthetic 行为节点和唯一 readiness 已完成并审计；实际
stage/status 已由独立 revisions 同步为 `runtime_verified / implemented`；真实
qualification 只验证冻结 runtime 边界，不改变下述科学成功条件。
候选 registry 的 10 个 ID 是算法身份计数，不是组件计数。

## Success Conditions

成功条件必须由独立 calibration/evaluation 切分上的固定 FPR 检测能力、错误密钥区分、攻击鲁棒性、几何估计误差、图像质量和资源成本共同定义。单一平均分数或未经校准的准确率不能独立定义成功。

完整 CEG-WM 成功还要求内容自适应路由、LF、HF、Q/K 几何、回正与联合判定均完成机制验证。LF 或路由未晋升可以闭合为 `research_question_closed_negative` 并形成论文负结果，但不能与完整方法成功共用终态；HF-only + geometry 若继续，必须作为重新命名、重新定义贡献并独立授权的 reduced-scope research identity。

完整方法的论文目标 operating point 为 `FPR = 0.001` 级别。该目标针对无水印图像与预登记检测密钥构成的 primary null，并包含 raw 阳性与几何救援后阳性。wrong-key null 单独报告，不得与 primary null 混池增加样本量。

若论文表述为 `FPR <= 0.001`，独立 evaluation 的经验 FPR 和预登记单侧置信上界都必须不超过 `0.001`。样本量不足、仅观察到零误报或只校准 raw detector 均不能独立支持该表述。统计设计、样本量和停止规则见 [research_construction_roadmap.md](research_construction_roadmap.md)。

## Non-Goals

- 不把几何可靠性直接解释为水印存在。
- 不使用原始参考图做正式注册。
- 不把 payload 恢复或系统级 attestation 作为当前主方法链。
- 不继承历史项目的固定 LF/HF 权重。
- 不以治理通过、代码数量或示例输出支持科研结论。
- 不把 CPU/synthetic 实现与行为通过误写成 LF/routing 实验晋升、runtime/GPU、
  正式 FPR 或科学效果验证。

## Expected Failure Modes

- 内容分数在严重攻击后远离可恢复区间；
- Q/K 特征不足以可靠估计几何变换；
- crop 删除了关键同步或内容证据；
- LF/HF 组合掩盖错误密钥或造成 calibration 漂移；
- 几何恢复引入额外内容失真；
- 模型 revision、预处理或图像尺寸变化破坏统计同一性。

这些失败必须在 records 中显式保存，不能被静默排除。

## Evidence Boundary

本文件和其他登记设计文档只支持“项目已定义”的阶段判断。算法原语见 [algorithm_primitives.md](algorithm_primitives.md)，端到端机制见 [method_mechanism.md](method_mechanism.md)，实施和证据门见 [research_construction_roadmap.md](research_construction_roadmap.md)。方法有效性、鲁棒性、比较优势和论文结论必须等待实质实现、冻结协议、真实 records 和可重建 artifacts。
