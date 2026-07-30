# Field Registry

## 文档定位

本文档登记研究配置、跨边界协议、records、manifests、artifact provenance 和模板中实际使用或预留的字段。构建工具的内部字段不在此重复登记。

本文档自包含稳定字段的 category、后缀和 claim 使用边界，删除外层治理目录后仍可解释 records 与 artifacts。证据链的整体语义见 `docs/reference/artifact_evidence.md`。

| category | required_suffix | lifecycle_and_claim_rule |
| --- | --- | --- |
| `placeholder` | `_placeholder` | 必须声明替换条件，不得支撑 claim 或完成状态。 |
| `random` | `_random` 或 `_digest_random` | 必须能够追溯 seed、digest 或等价复现信息。 |
| `intermediate` | `_intermediate` | 跨步骤保存但尚未成为稳定协议，不得支撑 claim。 |
| `temporary` | `_temporary` | 可清理且不得支撑 claim。 |
| `cache` | `_cache` | 必须可由输入、配置和代码重建，不得支撑 claim。 |
| 稳定语义 category | 无统一后缀 | 使用实际语义名称，并按字段等级决定是否可以进入 records 或 claims。 |

## 何时需要登记

新增字段只要进入下列任一稳定或跨边界位置，就应先登记到本表：

```text
研究与实验配置文件
records
manifests
tables
reports
稳定序列化 schema 或跨边界 Python mapping
正式 schema 的测试 fixture
定义正式 schema 的 Markdown 示例
Notebook 与 repository module 的跨边界数据
```

函数内部一次性局部变量、只服务局部计算的 dict key、普通 Markdown 表格列和不代表正式 schema 的测试数据不需要登记。跨文件、跨进程、跨 Notebook 或稳定模块接口保存的字段需要登记。

## 治理等级

| governance_level | scope |
| --- | --- |
| `internal_state` | 可重建、不可支撑 claim 的内部状态。 |
| `cross_boundary` | 跨模块、跨进程或跨 Notebook 边界传递的字段。 |
| `persisted_protocol` | 进入配置、records、manifests 或稳定序列化接口的字段。 |
| `evidence_bearing` | 可以直接参与 governed artifact 或 supported claim 证据链的字段。 |

等级使用语义名称，不使用 `level_1`、`p1` 或数字阶段名。每个登记字段必须有非空说明。

## 字段登记表

| field_name | governance_level | category | required_suffix | allowed_in_records | allowed_in_claims | replacement_required | description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run_id | persisted_protocol | protocol | none | true | false | false | 一次运行的稳定标识。 |
| record_id | persisted_protocol | protocol | none | true | false | false | 单条记录的稳定标识。 |
| split | persisted_protocol | protocol | none | true | false | false | 数据或事件划分。 |
| method_name | persisted_protocol | protocol | none | true | false | false | 实验记录中的方法名称。 |
| comparison_group_name | persisted_protocol | protocol | none | true | false | false | 一组受同一公平对比协议约束的方法集合名称。 |
| comparison_protocol_digest | persisted_protocol | protocol | none | true | false | false | 当前 record 所属公平对比协议的内容摘要。 |
| protocol_digest | cross_boundary | protocol | none | true | false | false | Preflight approval 或内部 record provenance 携带的协议内容摘要。 |
| sample_manifest_digest | persisted_protocol | protocol | none | true | false | false | 当前运行所用样本 manifest 的内容摘要。 |
| split_manifest_digest | persisted_protocol | protocol | none | true | false | false | calibration 与 evaluation 切分 manifest 的内容摘要。 |
| generation_conditions_digest | persisted_protocol | protocol | none | false | false | false | 各方法共享生成条件的内容摘要。 |
| seed_policy_digest | persisted_protocol | protocol | none | false | false | false | 随机种子分配规则的内容摘要。 |
| output_specification_digest | persisted_protocol | protocol | none | false | false | false | 可比输出规格的内容摘要。 |
| attack_matrix_digest | persisted_protocol | protocol | none | false | false | false | 对比使用攻击矩阵的内容摘要。 |
| metric_set_digest | persisted_protocol | protocol | none | true | false | false | 对比或内部逐样本 record 绑定的指标集合内容摘要。 |
| calibration_split | persisted_protocol | protocol | none | false | false | false | 仅用于调参与阈值校准的切分名称。 |
| evaluation_split | persisted_protocol | protocol | none | false | false | false | 仅用于报告比较结果的独立切分名称。 |
| tuning_budget_policy_digest | persisted_protocol | protocol | none | false | false | false | 各方法调参预算规则的内容摘要。 |
| compute_budget_policy_digest | persisted_protocol | protocol | none | false | false | false | 各方法计算预算规则的内容摘要。 |
| failure_policy_digest | persisted_protocol | protocol | none | false | false | false | 运行失败处理规则的内容摘要。 |
| exclusion_policy_digest | persisted_protocol | protocol | none | false | false | false | 样本或结果排除规则的内容摘要。 |
| method_role | persisted_protocol | protocol | none | true | false | false | 方法在对比中的项目方法或外部 baseline 角色。 |
| implementation_revision | persisted_protocol | protocol | none | false | false | false | 协议固定的方法实现 revision。 |
| method_config_digest | persisted_protocol | protocol | none | true | false | false | 当前 record 所用方法配置的内容摘要。 |
| candidate_id | cross_boundary | method_identity | none | false | false | false | 方法组件实际消费的冻结候选身份。 |
| candidate_ids | cross_boundary | method_identity | none | false | false | false | 跨职责组件实际绑定的完整有序候选身份集合。 |
| mode | cross_boundary | method_identity | none | false | false | false | router 或 content embedder 当前执行的冻结候选模式。 |
| keyed_prg_version | cross_boundary | method_identity | none | false | false | false | 密钥流使用的 KDF/PRG 算法身份。 |
| normal_quantile_table_sha256 | cross_boundary | method_identity | none | false | false | false | 共享冻结 `2^20` midpoint float32 normal quantile table 的字节摘要。 |
| root_key_encoding | cross_boundary | method_identity | none | false | false | false | root key 严格 UTF-8 且不执行 Unicode normalization 的编码协议身份。 |
| stable_serialization | cross_boundary | method_identity | none | false | false | false | key schedule 规范 JSON 序列化协议身份。 |
| uniform_protocol | cross_boundary | method_identity | none | false | false | false | uniform stream 的 high-53-bit open-interval binary32 物化协议身份。 |
| gaussian_protocol | cross_boundary | method_identity | none | false | false | false | Gaussian stream 的 MSB-first 20-bit midpoint-table 协议身份。 |
| root_key_public_digest | persisted_protocol | provenance | none | true | false | false | root key 的不可逆公开身份；原始 root key 和派生材料不得持久化。 |
| registered_root_key_public_digest | persisted_protocol | provenance | none | true | false | false | 派生 wrong-key roster 所绑定注册 root key 的不可逆公共身份。 |
| domain_digest | cross_boundary | provenance | none | false | false | false | 规范 key material、职责字段与 shape 共同形成的密钥流域摘要。 |
| distribution | cross_boundary | method_identity | none | false | false | false | key schedule 当前物化的 `uniform` 或 `gaussian` 冻结分布身份。 |
| quantile_indices_random | cross_boundary | random | `_random` | false | false | false | Gaussian PRG bitstream 查表产生的 20-bit index 随机轨迹；只用于 CPU golden 与内存诊断，不得进入 records。 |
| values_float32_be_sha256 | cross_boundary | provenance | none | false | false | false | row-major CPU float32 流按 IEEE-754 big-endian 拼接后的输出摘要。 |
| values | cross_boundary | method_state | none | false | false | false | 已由 runtime 或共享数值责任物化、供方法组件消费的有限数值序列。 |
| shape | cross_boundary | method_identity | none | false | false | false | 方法张量的显式 row-major 维度身份。 |
| spatial_shape | cross_boundary | method_identity | none | false | false | false | 单通道 routing observation 的 `[H,W]` 空间维度身份。 |
| latent_shape | cross_boundary | method_identity | none | false | false | false | content router 输出按 `[1,C,H,W]` 广播时绑定的 latent 网格身份。 |
| source_identity_digest | cross_boundary | provenance | none | false | false | false | runtime 提供 routing observation 时所绑定的公共来源配置摘要。 |
| values_digest | cross_boundary | provenance | none | false | false | false | routing observation 数值按 row-major float32 字节计算的摘要。 |
| template | cross_boundary | method_state | none | false | false | false | carrier 输出、可由 key 与公共身份重建的单位模板。 |
| direction | cross_boundary | method_state | none | false | false | false | carrier 经 mask 后交给 content embedder 的单位写入方向。 |
| support_indices | cross_boundary | method_state | none | false | false | false | HF sparse-tail 模板的 row-major 非零支持坐标。 |
| template_digest | cross_boundary | provenance | none | false | false | false | LF 或 HF 未遮罩模板的 float32 字节摘要。 |
| direction_digest | cross_boundary | provenance | none | false | false | false | mask 后 LF 或 HF 单位方向的 float32 字节摘要。 |
| mask_digest | cross_boundary | provenance | none | false | false | false | carrier 实际消费的空间 mask float32 字节摘要。 |
| key_role | cross_boundary | method_identity | none | true | false | false | 当前检测 key 是 registered 还是预登记 wrong-key。 |
| wrong_key_index | persisted_protocol | provenance | none | true | false | false | wrong-key roster 的预登记非负索引；registered key 时为空。 |
| key_domain_digest | cross_boundary | provenance | none | false | false | false | carrier 实际消费的 key schedule 职责域摘要。 |
| carrier_config_digest | cross_boundary | method_identity | none | false | false | false | carrier 算法、shape、mask、可选权威 route 绑定与 key schedule 配置身份摘要。 |
| delta_content | cross_boundary | method_state | none | false | false | false | content embedder 产生、尚未由 runtime 缩放/物化的 binary32 nominal combined content delta。 |
| delta_content_digest | cross_boundary | provenance | none | false | false | false | nominal `delta_content` 的 row-major binary32 字节摘要。 |
| latent_norm | cross_boundary | method_state | none | false | false | false | content embedder 按冻结 row-major binary32 协议重算 callback actual baseline 的 L2 norm。 |
| target_total_norm | cross_boundary | method_state | none | false | false | false | legacy-named nominal total L2 formula witness；不承诺 actual realized 值接近它。 |
| target_relative_l2 | cross_boundary | method_state | none | false | false | false | legacy-named nominal relative-L2 formula witness，固定为 `3/250`；不是 actual ratio 接近门。 |
| content_direction | cross_boundary | method_state | none | false | false | false | LF/HF/routing 最终 combined nominal unit direction 的公式重放 witness；不是 actual branch decomposition。 |
| active_lf_direction | cross_boundary | method_state | none | false | false | false | nominal 组合公式实际启用的 LF masked unit direction witness；不表示 LF actual-dtype delta。 |
| active_hf_direction | cross_boundary | method_state | none | false | false | false | nominal 组合公式实际启用的 HF masked unit direction witness；不表示 HF actual-dtype delta。 |
| embedding_result_identity | cross_boundary | provenance | none | false | false | false | nominal embedding 全字段、方向 witness 与配置摘要的不可变身份。 |
| mixing_coefficient | cross_boundary | method_identity | none | true | false | false | combined content embedder 使用的冻结 `a`；它是方向混合系数而非可加能量份额。 |
| gamma_lh | cross_boundary | method_statistic | none | true | false | false | LF/HF 两条 masked unit direction 的内积交叉项。 |
| combined_pre_normalization_norm | cross_boundary | method_statistic | none | true | false | false | 含 `gamma_lh` 交叉项的 combined direction 归一化因子 `c(a)`。 |
| target_component_lf | cross_boundary | method_state | none | true | false | false | 可重建的理论 LF target component vector；不得作为实际 dtype 分支能量。 |
| target_component_hf | cross_boundary | method_state | none | true | false | false | 可重建的理论 HF target component vector；不得作为实际 dtype 分支能量。 |
| target_component_lf_norm | cross_boundary | method_statistic | none | true | false | false | 理论 LF target component 的 L2 norm；不得与 HF component norm 相加冒充 total。 |
| target_component_hf_norm | cross_boundary | method_statistic | none | true | false | false | 理论 HF target component 的 L2 norm；不得与 LF component norm 相加冒充 total。 |
| lf_carrier_config_digest | cross_boundary | method_identity | none | false | false | false | content embedder 实际消费的 LF carrier 配置摘要；LF 未启用时为空。 |
| hf_carrier_config_digest | cross_boundary | method_identity | none | false | false | false | content embedder 实际消费的 HF carrier 配置摘要；HF 未启用时为空。 |
| embedder_config_digest | cross_boundary | method_identity | none | false | false | false | content embedder 候选、模式和共同总预算身份摘要。 |
| paired_base_latent_digest | cross_boundary | provenance | none | false | false | false | clean/watermarked 两条生成路径共享且各自 clone 的同一基础 float16 latent 身份摘要。 |
| clean_callback_indices | cross_boundary | runtime_identity | none | false | false | false | clean 生成路径实际触发 callback 的完整有序 index 序列。 |
| watermarked_callback_indices | cross_boundary | runtime_identity | none | false | false | false | watermarked 生成路径实际触发 callback 的完整有序 index 序列。 |
| content_materialization | cross_boundary | runtime_state | none | false | false | false | `content_embedder` 最终选中 scale 对应的 callback 18 actual-dtype 张量、独立重放身份和 realized 测量；runtime 对象本身不拥有预算判定。 |
| content_materialization_result | cross_boundary | method_state | none | false | false | false | `main.content_embedder` 返回的最终 nominal/limit、选中 observation、scale、attempt、integrity 与 budget 结果。 |
| content_materialization_attempts | cross_boundary | runtime_state | none | false | false | false | runtime 按 `content_embedder` 请求顺序保存的不含中间 tensor 的全部物化尝试身份与 realized 测量。 |
| embedding_result | cross_boundary | method_state | none | false | false | false | materialization 结果回绑且通过 nominal formula replay 的不可变 `ContentEmbeddingResult`。 |
| observation | cross_boundary | method_state | none | false | false | false | `content_embedder` 最终接受的唯一 actual-dtype materialization observation。 |
| baseline_latent_actual | cross_boundary | runtime_state | none | false | false | false | callback 18 写入前、按注册 float16 dtype 物化的实际 latent 防共享副本。 |
| written_latent_actual | cross_boundary | runtime_state | none | false | false | false | callback 18 将选中 scale 的 nominal content delta 加入 baseline 后按注册 float16 RNE 物化的实际 latent。 |
| delta_content_actual | cross_boundary | runtime_state | none | false | false | false | `float32(written_latent_actual)-float32(baseline_latent_actual)` 得到的 combined actual-dtype 内容更新。 |
| content_relative_l2_nominal | cross_boundary | method_identity | none | true | false | false | 当前内容候选的 nominal relative-L2，固定为 `3/250`；不要求 actual ratio 接近它。 |
| content_relative_l2_limit | cross_boundary | method_identity | none | true | false | false | combined actual content delta 的唯一 hard relative-L2 上限，固定为 `3/250`；geometry 不并入。 |
| materialization_scale | cross_boundary | method_state | none | true | false | false | 在 `ContentMaterializationMeasurement`/`ContentMaterializationAttempt` 中是当前单次请求/尝试的 binary32 scale，可能超限或产生 `write_disappeared`；仅在 `ContentMaterializationResult` 中表示 `main.content_embedder` 最终选中的 greatest nonzero feasible scale，`s=1` 直接可行时也属于最终选中。 |
| baseline_norm | cross_boundary | method_statistic | none | false | false | false | runtime observation 对 callback actual baseline 按 row-major binary32 协议计算的 L2，必须与 embedder 输入一致。 |
| scaled_nominal_delta_digest | cross_boundary | provenance | none | false | false | false | 每次尝试按 `f32(delta_content*s)` 得到的 row-major binary32 nominal delta 摘要。 |
| baseline_latent_digest | cross_boundary | provenance | none | false | false | false | actual baseline 的 dtype、shape 和逐位张量内容摘要。 |
| written_latent_digest | cross_boundary | provenance | none | false | false | false | actual written latent 的 dtype、shape 和逐位张量内容摘要。 |
| delta_content_actual_digest | cross_boundary | provenance | none | false | false | false | `delta_content_actual` row-major float32 big-endian 字节摘要。 |
| tensor_replay_identity | cross_boundary | runtime_identity | none | false | false | false | attempt index、callback、scale、binary16 actual tensor bits、delta digest 与 realized 测量的 runtime bitwise replay 身份。 |
| materialization_replay_identity | cross_boundary | runtime_identity | none | false | false | false | scale、scaled nominal delta、callback baseline、combined actual delta、integrity 和 row-major realized L2 的方法/runtime 联合重放身份。 |
| deterministic_binary16_replay_passed | cross_boundary | runtime_state | none | false | false | false | runtime 对登记 binary16 RNE 写入逐 bit 独立重放是否相同；false 必须 fail closed。 |
| realized_total_l2 | cross_boundary | method_statistic | none | true | false | false | runtime 对 combined `delta_content_actual` 按固定 row-major binary32 累加协议重算的实际总 L2；hard gate 直接比较此值与 limit norm。 |
| realized_relative_l2 | cross_boundary | method_statistic | none | false | false | false | `realized_total_l2` 除以 actual callback baseline L2 的实际相对量；不单独表示预算合格。 |
| budget_utilization | cross_boundary | method_statistic | none | true | false | false | `realized_total_l2/limit_norm` 的 binary32 诊断量；不是权威 gate，低值不得事后筛除。 |
| integrity_status | cross_boundary | method_state | none | true | false | false | 单次物化的 `passed` 或 `write_disappeared`；最终接受结果只能为 `passed`。 |
| budget_status | cross_boundary | method_state | none | true | false | false | `main.content_embedder` 成功返回时固定为 `accepted`；超限且无非零可行写入时 fail closed，不伪造 rejected result。 |
| attempt_index | cross_boundary | runtime_state | none | false | false | false | runtime 按 `content_embedder` 请求顺序从 1 起记录的单次 materialization 尝试序号。 |
| attempt_count | cross_boundary | method_statistic | none | true | false | false | `content_embedder` 为 full-scale 检查与 binary32 最大可行搜索实际请求的总物化次数。 |
| clean_generation_terminal_latent | cross_boundary | runtime_state | none | false | false | false | clean 路径完成冻结 scheduler suffix 后、进入 generation VAE decode 的实际 latent。 |
| watermarked_generation_terminal_latent | cross_boundary | runtime_state | none | false | false | false | watermarked 路径完成冻结 scheduler suffix 后、进入 generation VAE decode 的实际 latent。 |
| vae_scaling_factor_actual | cross_boundary | runtime_identity | none | false | false | false | prepared backend 从登记 VAE config 来源读取的有限正 scaling factor 实际值。 |
| vae_shift_factor_actual | cross_boundary | runtime_identity | none | false | false | false | prepared backend 从登记 VAE config 来源读取的有限 shift factor 实际值。 |
| clean_image | cross_boundary | runtime_state | none | false | false | false | clean final latent 严格按冻结 scaling/shift decode 协议产生的普通图像张量。 |
| watermarked_image | cross_boundary | runtime_state | none | false | false | false | watermarked final latent 严格按冻结 scaling/shift decode 协议产生的普通图像张量。 |
| clean_detection_latent | cross_boundary | runtime_state | none | false | false | false | clean image 经 VAE posterior mode 和冻结 shift/scaling encode 协议得到的检测 latent。 |
| watermarked_detection_latent | cross_boundary | runtime_state | none | false | false | false | watermarked image 经 VAE posterior mode 和冻结 shift/scaling encode 协议得到的检测 latent。 |
| observation_protocol | cross_boundary | method_identity | none | false | false | false | 普通检测图像进入 LF/HF detector 的公共编码协议身份。 |
| observation_digest | cross_boundary | provenance | none | false | false | false | 普通检测图像侧 LF/HF 编码观测的 float32 字节摘要；两分支组合时必须相同。 |
| hf_score | cross_boundary | method_statistic | none | true | false | false | HF detector 独立产生的 blind direct score。 |
| lf_score | cross_boundary | method_statistic | none | true | false | false | LF detector 独立产生的 blind low-pass score。 |
| combined_score | cross_boundary | method_statistic | none | true | false | false | 由冻结 C0/C1/C2 公式产生且当前仅用于未晋升候选诊断的组合统计。 |
| detector_identity | cross_boundary | method_identity | none | true | false | false | 分支或 content detector 的完整算法身份摘要。 |
| detector_config_digest | cross_boundary | method_identity | none | false | false | false | LF 或 HF 分支 detector 的配置身份摘要。 |
| content_score | cross_boundary | method_statistic | none | true | false | false | 当前正式 content detector 输出的 `D_M` 分数。 |
| content_config_digest | cross_boundary | method_identity | none | false | false | false | 当前 content detector 的分支与组合状态身份摘要。 |
| hf_result | cross_boundary | method_state | none | false | false | false | content detector 原样保留的独立 HF 分支结果。 |
| lf_result | cross_boundary | method_state | none | false | false | false | content detector 原样保留的独立 LF 分支结果；未执行 LF 时为空。 |
| formal_mode | cross_boundary | method_identity | none | true | false | false | 当前拥有正式 `D_M` 解释权的 content detector 模式；批次 3 保持 `hf_only`。 |
| diagnostic_combination | cross_boundary | method_state | none | false | false | false | 未晋升 C0/C1/C2 的完整标准化与组合诊断；不存在时为空。 |
| diagnostic_identity | cross_boundary | method_identity | none | false | false | false | 连接正式 HF-only detector 与未晋升组合诊断的不可变摘要。 |
| routing_observations | cross_boundary | method_state | none | false | false | false | routed result 保留的实际不可变 S/T/R/Q_sens 观测集合，供公式重演验证；uniform control 时为空。 |
| routing_map | cross_boundary | method_state | none | false | false | false | `routing_stqr` 输出并按 latent channels 广播的空间权威图 `A`。 |
| mask_lf | cross_boundary | method_state | none | false | false | false | content router 输出、供 LF carrier 消费的空间 mask。 |
| mask_hf | cross_boundary | method_state | none | false | false | false | content router 输出、供 HF carrier 消费的空间 mask。 |
| routing_map_digest | cross_boundary | provenance | none | false | false | false | router `A` 的 row-major float32 字节摘要。 |
| mask_lf_digest | cross_boundary | provenance | none | false | false | false | router LF mask 的 row-major float32 字节摘要。 |
| mask_hf_digest | cross_boundary | provenance | none | false | false | false | router HF mask 的 row-major float32 字节摘要。 |
| observation_digests | cross_boundary | provenance | none | false | false | false | 从实际 `routing_observations` 重算的 S/T/R/Q_sens 数值摘要与各自 runtime 来源身份有序集合。 |
| semantic | cross_boundary | method_state | none | false | false | false | content router 消费的 runtime-provided 语义观测 `S`。 |
| texture | cross_boundary | method_state | none | false | false | false | content router 消费的 runtime-provided 纹理观测 `T`。 |
| response | cross_boundary | method_state | none | false | false | false | content router 消费的 callback latent 响应观测 `R`。 |
| sensitivity | cross_boundary | method_state | none | false | false | false | content router 消费的 public-probe 局部敏感性观测 `Q_sens`。 |
| mean_routing_map | cross_boundary | method_statistic | none | true | false | false | router `A` 在完整 latent CHW 广播后的均值。 |
| mean_mask_lf | cross_boundary | method_statistic | none | true | false | false | LF routing mask 在完整 latent CHW 广播后的均值。 |
| mean_mask_hf | cross_boundary | method_statistic | none | true | false | false | HF routing mask 在完整 latent CHW 广播后的均值。 |
| route_config_digest | cross_boundary | method_identity | none | false | false | false | 路由公式、插值、候选和 latent shape 的配置摘要；route、carrier 与 combined embedder 必须一致。 |
| route_identity | cross_boundary | method_identity | none | true | false | false | 路由配置、observation 与 mask 摘要共同形成并由两 carrier、combined embedder 共同绑定的路由身份。 |
| score | persisted_protocol | method_statistic | none | true | false | false | empirical CDF null record 保存的分支 float64 分数。 |
| source_cluster_id | persisted_protocol | provenance | none | true | false | false | calibration null record 所属 Prompt、seed、lineage 与 key family 聚类身份。 |
| sample_id | persisted_protocol | provenance | none | true | false | false | calibration null record 在 source cluster 内的稳定样本身份。 |
| branch | cross_boundary | method_identity | none | false | false | false | empirical CDF 与标准化结果所属的 `hf` 或 `lf` 分支。 |
| partition_identity | persisted_protocol | provenance | none | true | false | false | empirical CDF 所属互斥 calibration 职责 partition 身份。 |
| records | cross_boundary | method_state | none | true | false | false | 分支 empirical CDF 的稳定排序输入或一个 run/case governed record collection 的逐尝试集合。 |
| calibration_identity | cross_boundary | method_identity | none | true | false | false | 分支 detector、partition、null records 与 quantile table 的绑定摘要。 |
| raw_score | cross_boundary | method_statistic | none | true | false | false | empirical CDF 标准化前独立保存的分支查询分数。 |
| less_count | cross_boundary | method_statistic | none | true | false | false | null multiset 中严格小于查询分数的记录数。 |
| equal_count | cross_boundary | method_statistic | none | true | false | false | null multiset 中与查询 float64 分数精确相等的记录数。 |
| null_count | cross_boundary | method_statistic | none | true | false | false | 当前分支 empirical CDF 的 primary-null 记录总数。 |
| u_raw | cross_boundary | method_statistic | none | true | false | false | ties 使用 mid-rank 后、tail clipping 前的 empirical CDF 值。 |
| epsilon_n | cross_boundary | method_statistic | none | true | false | false | 当前 null 样本量确定的双尾 clipping 下界 `1/(2n)`。 |
| u_clipped | cross_boundary | method_statistic | none | true | false | false | 限制到 `[epsilon_n,1-epsilon_n]` 的 empirical CDF 值。 |
| quantile_index | cross_boundary | method_state | none | true | false | false | `u_clipped` 映射到共享冻结 `2^20` normal table 的索引。 |
| z_score | cross_boundary | method_statistic | none | true | false | false | 从共享冻结 midpoint float32 normal table 恢复并提升为 float64 的分支统计。 |
| function_id | cross_boundary | method_identity | none | true | false | false | 未晋升诊断使用的冻结 C0、C1 weight 或 C2 公式身份。 |
| weight | cross_boundary | method_identity | none | true | false | false | C1 诊断公式的冻结有限权重；C0/C2 时为空。 |
| hf_standardization | cross_boundary | method_state | none | false | false | false | 组合诊断保留的完整 HF empirical-CDF 标准化结果。 |
| lf_standardization | cross_boundary | method_state | none | false | false | false | C1/C2 组合诊断保留的完整 LF 标准化结果；C0 时为空。 |
| formula_identity | cross_boundary | method_identity | none | true | false | false | C0/C1/C2 公式、weight 与共享 quantile-table 摘要形成的身份。 |
| combination_identity | cross_boundary | method_identity | none | true | false | false | 公式与两分支 calibration identity 共同形成的未晋升组合身份。 |
| diagnostic_only | cross_boundary | method_state | none | true | false | false | 明确组合输出只具诊断语义、不能替代当前正式 `D_M`。 |
| promoted | cross_boundary | method_state | none | true | false | false | 组合候选是否已通过独立晋升门；批次 3 固定为 false 且不提供晋升权。 |
| layers | cross_boundary | method_state | none | false | false | false | Q/K 几何结果中按冻结顺序保存的两个登记层 relation 结果。 |
| layer_name | cross_boundary | method_identity | none | false | false | false | Q/K observation 与 projection 绑定的登记 attention 层名。 |
| query | cross_boundary | method_state | none | false | false | false | runtime 登记层提供的真实 attention query tensor；必须消费数值本身，摘要不能替代 tensor。 |
| attention_key | cross_boundary | method_state | none | false | false | false | runtime 登记层提供的真实 attention key tensor；与检测密钥语义分离且摘要不能替代 tensor。 |
| head_count | cross_boundary | method_identity | none | false | false | false | 登记层 Q/K observation 的实际 attention head 数。 |
| head_width | cross_boundary | method_identity | none | false | false | false | 登记层 Q/K observation 的实际单 head 宽度。 |
| original_grid_side | cross_boundary | method_identity | none | false | false | false | 等距 token 采样前的原始方形图像 token 网格边长。 |
| token_indices | cross_boundary | method_identity | none | false | false | false | 从原始图像 token 网格按冻结等距规则选择的 row-major 索引。 |
| token_count | cross_boundary | method_identity | none | false | false | false | 单层 relation 实际消费的采样 token 数。 |
| relation_shape | cross_boundary | method_identity | none | false | false | false | 四通道 relation 的显式 `[token,token,4]` 形状。 |
| relation_values | cross_boundary | method_state | none | false | false | false | 从真实 Q/K 数值构造的两 token 轴四通道 float32 relation。 |
| projection_values | cross_boundary | method_state | none | false | false | false | geometry-key 上三角镜像、零对角、固定 polarity 四通道投影。 |
| relation_score | cross_boundary | method_statistic | none | true | false | false | 两登记层、四通道等权逐 row 中心化相关统计；无内容阳性语义。 |
| descriptor_digest | cross_boundary | provenance | none | false | false | false | 两登记层实际 relation 数值和层序绑定的摘要。 |
| projection_digest | cross_boundary | provenance | none | false | false | false | geometry-key 投影、层序和通道 polarity 绑定的摘要。 |
| operator_identity | cross_boundary | method_identity | none | false | false | false | runtime 提供的 Q/K 投影、归一化、head layout 与 scale 算子身份。 |
| geometry_config_digest | cross_boundary | method_identity | none | false | false | false | Q/K 层序、token 采样、rank、projection 与 row-score 配置摘要。 |
| accepted | cross_boundary | method_state | none | false | false | false | actual-dtype 几何同步回溯是否找到满足全部冻结约束的候选。 |
| status | cross_boundary | method_state | none | true | false | false | 几何同步、可靠性或回正边界的显式状态。 |
| geometry_ratio | cross_boundary | method_identity | none | false | false | false | `rho_geo/rho_content` 的冻结有限候选值。 |
| line_search_factor | cross_boundary | method_state | none | false | false | false | actual-dtype 回溯接受的首个 `lambda`；失败时为空。 |
| baseline_score | cross_boundary | method_statistic | none | false | false | false | 几何同步 actual-dtype 回溯前的同一 image-only relation score。 |
| accepted_score | cross_boundary | method_statistic | none | false | false | false | 回溯接受候选重新执行完整 image-only Q/K 路径后的 relation score。 |
| geometry_relative_l2_actual | cross_boundary | method_statistic | none | true | false | false | 实际 dtype 几何增量相对 callback baseline latent 的 L2。 |
| total_relative_l2_actual | cross_boundary | method_statistic | none | true | false | false | 内容与几何一次最终物化后实际总增量的相对 L2。 |
| content_projection_relative | cross_boundary | method_statistic | none | true | false | false | actual geometry delta 投影回完整 LF/HF 内容方向 span 的相对范数。 |
| written_latent | cross_boundary | method_state | none | false | false | false | 通过全部 actual-dtype 回溯约束后的 latent；失败时为空。 |
| transform | cross_boundary | method_state | none | true | false | false | estimator 输出的 canonical-to-observed similarity affine 与搜索坐标。 |
| dihedral | cross_boundary | method_identity | none | true | false | false | 冻结八个方形网格 dihedral 基元之一。 |
| residual_rotation_degrees | cross_boundary | method_statistic | none | true | false | false | dihedral 后的有界连续 residual rotation 角度。 |
| log_scale | cross_boundary | method_statistic | none | true | false | false | similarity scale 的自然对数参数。 |
| translation_x | cross_boundary | method_statistic | none | true | false | false | canonical-to-observed 规范坐标 x translation。 |
| translation_y | cross_boundary | method_statistic | none | true | false | false | canonical-to-observed 规范坐标 y translation。 |
| matrix | cross_boundary | method_state | none | true | false | false | canonical-to-observed float32 `2x3` affine matrix。 |
| is_exact_identity | cross_boundary | method_state | none | true | false | false | 首胜候选是否为精确 identity matrix 与零连续参数。 |
| continuous_parameter_on_search_boundary | cross_boundary | method_state | none | true | false | false | 任一连续估计参数是否落在冻结搜索支持边界。 |
| registered_objective | cross_boundary | method_statistic | none | true | false | false | 注册 geometry key 完整搜索的最高冻结 objective。 |
| second_registered_objective | cross_boundary | method_statistic | none | true | false | false | 去除 best 重复 matrix 后注册 key 的次高 objective。 |
| exact_identity_objective | cross_boundary | method_statistic | none | true | false | false | 注册 key 精确 identity candidate 的 objective。 |
| wrong_key_objectives | cross_boundary | method_statistic | none | true | false | false | 预登记索引 `0..7` 八个 wrong geometry key 各自完整搜索的最高 objective。 |
| canonical_score | cross_boundary | method_statistic | none | true | false | false | best candidate 的两层 canonical direction row-normalized score。 |
| observation_score | cross_boundary | method_statistic | none | true | false | false | best candidate 的两层 observation direction row-normalized score。 |
| coverage_forward | cross_boundary | method_statistic | none | true | false | false | canonical-to-observed sampling matrix 的有效 row 比例。 |
| coverage_backward | cross_boundary | method_statistic | none | true | false | false | observed-to-canonical sampling matrix 的有效 row 比例。 |
| uniqueness_forward | cross_boundary | method_statistic | none | true | false | false | forward valid row 的唯一 argmax token 覆盖比例。 |
| uniqueness_backward | cross_boundary | method_statistic | none | true | false | false | backward valid row 的唯一 argmax token 覆盖比例。 |
| coverage | cross_boundary | method_statistic | none | true | false | false | forward/backward coverage 的较小值。 |
| uniqueness | cross_boundary | method_statistic | none | true | false | false | forward/backward uniqueness 的较小值。 |
| gap | cross_boundary | method_statistic | none | true | false | false | 注册 key best 与 second-best objective 的差。 |
| identity_margin | cross_boundary | method_statistic | none | true | false | false | 注册 key best 与 exact identity objective 的差。 |
| key_margin | cross_boundary | method_statistic | none | true | false | false | 注册 key best objective 与八个 wrong-key best 最大值的差。 |
| inlier_ratio | cross_boundary | method_statistic | none | true | false | false | 十二个冻结 anchor 在拟合 `epsilon_inlier` 下的有效最近点比例。 |
| mean_residual | cross_boundary | method_statistic | none | true | false | false | 十二个冻结 anchor residual 的均值；任一越界时为非有限失败量。 |
| epsilon_inlier | cross_boundary | method_identity | none | false | false | false | 独立 geometry-reliability-fit 冻结并供 anchor inlier 使用的阈值。 |
| anchor_residuals | cross_boundary | method_statistic | none | true | false | false | 十二个冻结 anchor 到 observed grid 最近点的原始 residual。 |
| observation_descriptor_digest | cross_boundary | provenance | none | false | false | estimator 实际消费的两层 Q/K relation observation 摘要。 |
| observation_projection_digest | cross_boundary | provenance | none | false | false | false | estimator 已验证 Q/K observation 的 geometry-key projection、层序与 polarity 摘要。 |
| observation_geometry_config_digest | cross_boundary | method_identity | none | false | false | false | estimator 已验证 Q/K observation 的层、token/operator 和 relation 配置摘要。 |
| search_config_digest | cross_boundary | method_identity | none | false | false | dihedral/coarse/refine/objective/wrong-key roster 搜索配置摘要。 |
| estimation_identity_digest | cross_boundary | method_identity | none | true | false | transform、全部原始指标、key family、完整 Q/K observation identity 与 search config 的绑定摘要。 |
| gamma_coverage | cross_boundary | method_identity | none | false | false | false | 独立 reliability fit 冻结的 coverage 合取阈值。 |
| gamma_uniqueness | cross_boundary | method_identity | none | false | false | false | 独立 reliability fit 冻结的 uniqueness 合取阈值。 |
| gamma_gap | cross_boundary | method_identity | none | false | false | false | 独立 reliability fit 冻结的 best/second gap 合取阈值。 |
| gamma_key | cross_boundary | method_identity | none | false | false | false | 独立 reliability fit 冻结的 registered/wrong-key margin 合取阈值。 |
| gamma_inlier | cross_boundary | method_identity | none | false | false | false | 独立 reliability fit 冻结的 anchor inlier ratio 合取阈值。 |
| gamma_residual | cross_boundary | method_identity | none | false | false | false | 独立 reliability fit 冻结的 mean residual 上界。 |
| gamma_identity | cross_boundary | method_identity | none | false | false | false | 非 identity best 必须满足的 identity objective margin 阈值。 |
| fit_identity | cross_boundary | provenance | none | false | false | false | geometry reliability thresholds 所属独立拟合职责身份。 |
| reliable | cross_boundary | method_state | none | true | false | false | 冻结可靠性合取是否全部通过；无内容阳性语义。 |
| allow_rectification | cross_boundary | method_state | none | true | false | false | 是否允许 image rectifier 消费同一 identity-bound estimation。 |
| failure_reasons | cross_boundary | method_state | none | true | false | false | reliability fail-closed 的完整原因集合。 |
| fitted_reliability_thresholds | cross_boundary | method_identity | none | false | false | false | reliability 结果携带的完整不可变拟合阈值结构；未拟合结果为空。 |
| threshold_config_digest | cross_boundary | method_identity | none | false | false | false | 独立拟合阈值、fit identity 与冻结合取规则摘要。 |
| estimator_search_config_digest | cross_boundary | method_identity | none | false | false | false | reliability 输出回绑 estimator 搜索身份的摘要。 |
| reliability_identity_digest | cross_boundary | method_identity | none | true | false | false | reliability 的阈值全值、决策字段和 estimator/search/root 绑定的整体摘要。 |
| rectified_image | cross_boundary | method_state | none | false | false | false | PyTorch inverse warp 后按 clamp、乘 255、floor 产生的 RGB uint8 图像。 |
| rectified_image_digest | cross_boundary | provenance | none | true | false | false | rectifier 对回正 RGB8 dtype、shape 与逐值字节的稳定摘要；内容重判必须绑定同一回正图。 |
| valid_support_mask | cross_boundary | method_state | none | true | false | false | 同 grid 对全 1 输入以 nearest/zeros 得到的有效像素支持 mask。 |
| token_crop_support | cross_boundary | method_statistic | none | true | false | false | estimator forward/backward token coverage 的较小值。 |
| pixel_crop_support | cross_boundary | method_statistic | none | true | false | false | valid-support pixel mask 的有效比例。 |
| crop_support | cross_boundary | method_statistic | none | true | false | false | 同时保存 token 双向 coverage 与 pixel mask 比例的有序二元组。 |
| canonical_to_observed_matrix | cross_boundary | method_state | none | true | false | false | rectifier 实际作为 output-to-input theta 消费的 estimator affine matrix。 |
| rectification_config_digest | cross_boundary | method_identity | none | false | false | 图像/支持插值、padding、align-corners、量化与尺寸的配置摘要。 |
| content_detection_operation | cross_boundary | method_state | none | false | false | false | 同一联合调用在原图与回正图上复用的普通图像内容检测 operation 对象；不进入稳定序列化。 |
| content_replay_operation | cross_boundary | method_state | none | false | false | false | 内容结果保存的同一预处理/检测 operation；由调用方逐次显式传入 `detection_key`，重放实际图像到 observation、HF score 和完整 content result；不持有、持久化原始 key，且不进入稳定序列化。 |
| content_detector_binding | cross_boundary | method_state | none | false | false | false | 联合结果保存的不可变 content detector binding；公开 validator 复用其中同一 operation 执行 raw/rectified 数据链重放。 |
| preprocessing_identity | cross_boundary | method_identity | none | true | false | false | 原图与回正图共同绑定的普通图像编码和内容检测预处理身份。 |
| detector_binding_digest | cross_boundary | method_identity | none | true | false | false | content detector、正式模式、预处理与同一 operation 角色的联合绑定摘要。 |
| hf_detector_identity | cross_boundary | method_identity | none | false | false | false | joint binding 显式固定的正式 HF branch detector 身份；raw 与 rectified 结果必须一致。 |
| hf_detector_config_digest | cross_boundary | method_identity | none | false | false | false | joint binding 显式固定的正式 HF branch detector 配置摘要。 |
| hf_template_digest | cross_boundary | provenance | none | false | false | false | joint binding 对当前注册 key 的 HF 模板摘要；raw 与 rectified 重判必须一致。 |
| tau | cross_boundary | method_identity | none | true | false | false | 当前 content detector 身份在独立 calibration 中冻结的唯一阳性阈值。 |
| tau_rescue | cross_boundary | method_identity | none | true | false | false | 只控制几何恢复资格且严格低于 `tau` 的近阈值负区间下界。 |
| threshold_identity | cross_boundary | method_identity | none | true | false | false | `tau`、`tau_rescue`、detector binding 与 calibration provenance 的联合摘要。 |
| raw_content_result | cross_boundary | method_state | none | false | false | false | 联合判定首次对原始普通图像执行正式 content detector 得到的完整不可变结果。 |
| source_image | cross_boundary | method_state | none | false | false | false | 联合判定保存的普通 RGB8 原始输入防共享副本；用于重放内容绑定和几何回正。 |
| source_image_digest | cross_boundary | provenance | none | true | false | false | 联合判定与 rectifier 对原始 RGB8 dtype、shape 与逐值字节的稳定摘要；用于绑定 raw/rescue 结果来源。 |
| content_input_image_digest | cross_boundary | provenance | none | true | false | false | 内容链对本次实际普通 RGB8 输入的稳定摘要；joint 分别以原图和回正图调用内容侧 validator 复核。 |
| raw_content_score | cross_boundary | method_statistic | none | true | false | false | 联合判定从 `raw_content_result` 原样读取的正式 `D_M` 分数。 |
| geometry_triggered | cross_boundary | method_state | none | true | false | false | 原始内容分数是否位于 `[tau_rescue,tau)` 并实际惰性调用几何估计。 |
| trigger_reason | cross_boundary | method_state | none | true | false | false | raw 阳性、远阈值负样本、近阈值资格或 raw 内容失败对应的互斥触发原因。 |
| geometry_estimation | cross_boundary | method_state | none | false | false | false | 近阈值门后才产生并经公开 validator 复验的完整变换估计结果。 |
| geometry_reliability_result | cross_boundary | method_state | none | false | false | false | joint 委托独立 reliability 组件对同一 estimation 产生并复验的结果。 |
| image_rectification_result | cross_boundary | method_state | none | false | false | false | 仅可靠 geometry 后由真实 image rectifier 产生的完整回正结果。 |
| rectified_content_result | cross_boundary | method_state | none | false | false | false | 同一 detector operation、密钥语义、预处理和阈值对回正普通图像产生的内容结果。 |
| rectified_content_score | cross_boundary | method_statistic | none | true | false | false | 从 `rectified_content_result` 原样读取、以同一 `tau` 重判的正式内容分数。 |
| joint_content_positive | cross_boundary | method_state | none | true | false | false | 只由 raw 或 rectified content score 达到同一 `tau` 形成的方法整体联合内容判定。 |
| positive_source | cross_boundary | method_state | none | true | false | false | 最终阳性唯一允许的 `raw_content` 或 `rectified_content` 内容证据来源；负样本为空。 |
| full_ceg_wm_eligible | cross_boundary | method_state | none | true | false | false | 当前联合结果是否具备完整 CEG-WM 身份资格；HF-only construction candidate 固定为 false。 |
| positive_path | cross_boundary | method_state | none | true | false | false | 联合阳性的 `raw_positive` 或 `rescue_positive` 控制流路径；负样本为空，FPR 由实验层结合 null 标签计算。 |
| decision_identity_digest | cross_boundary | method_identity | none | true | false | false | 联合路径、原始 RGB8 防共享副本及摘要、分数、内容/几何/阈值身份、阳性来源与失败状态的一致性摘要。 |
| declared_deviation | persisted_protocol | protocol | none | false | false | false | baseline 相对上游实现的已声明语义偏差。 |
| methods | persisted_protocol | protocol | none | false | false | false | Comparison protocol 中参与方法规格的有序集合。 |
| method_code_revision | persisted_protocol | protocol | none | true | false | false | 当前 record 实际执行的方法代码 revision。 |
| model_revision | persisted_protocol | protocol | none | true | false | false | 当前 record 实际执行的生成模型 revision。 |
| runtime_schema_version | persisted_protocol | runtime_identity | none | false | false | false | runtime 配置 JSON 的严格 schema 版本。 |
| runtime_config_digest | cross_boundary | runtime_identity | none | false | false | false | 冻结 runtime 配置按 canonical JSON 计算的 SHA-256 身份。 |
| model_id | persisted_protocol | runtime_identity | none | true | false | false | runtime 实际加载的公开模型仓库身份。 |
| pipeline_class | persisted_protocol | runtime_identity | none | true | false | false | runtime 实际使用的完整 pipeline 类身份。 |
| scheduler_class | persisted_protocol | runtime_identity | none | true | false | false | runtime 实际使用的完整 scheduler 类身份。 |
| inference_steps | persisted_protocol | runtime_identity | none | true | false | false | 生成和检测 schedule 绑定的冻结 inference step 数。 |
| guidance_scale | persisted_protocol | runtime_identity | none | true | false | false | 生成路径冻结的 classifier-free guidance scale。 |
| image_height | persisted_protocol | runtime_identity | none | true | false | false | runtime 候选冻结的 RGB 输出高度。 |
| image_width | persisted_protocol | runtime_identity | none | true | false | false | runtime 候选冻结的 RGB 输出宽度。 |
| generation_seed_device | persisted_protocol | runtime_identity | none | false | false | false | 基础 latent 随机 generator 的冻结设备身份。 |
| latent_dtype | persisted_protocol | runtime_identity | none | true | false | false | runtime 实际物化 latent 的 dtype。 |
| template_dtype | persisted_protocol | runtime_identity | none | true | false | false | 方法模板计算的冻结 dtype。 |
| score_dtype | persisted_protocol | runtime_identity | none | true | false | false | 内容与几何分数计算的冻结 dtype。 |
| callback_index | persisted_protocol | runtime_identity | none | true | false | false | 内容及几何更新进入 scheduler 的冻结 callback index。 |
| callback_hold_scheduler_intervals | persisted_protocol | runtime_identity | none | true | false | false | callback 写入后必须保留更新的 scheduler interval 数。 |
| vae_decode_protocol | persisted_protocol | runtime_identity | none | true | false | false | 使用 VAE scaling 和 shift 解码 generation latent 的冻结公式身份。 |
| vae_encode_protocol | persisted_protocol | runtime_identity | none | true | false | false | 使用 posterior mode、VAE shift 和 scaling 编码检测图像的冻结公式身份。 |
| vae_scaling_factor_source | persisted_protocol | runtime_identity | none | false | false | false | VAE scaling factor 的实际模型配置来源路径。 |
| vae_shift_factor_source | persisted_protocol | runtime_identity | none | false | false | false | VAE shift factor 的实际模型配置来源路径。 |
| detection_schedule_index | persisted_protocol | runtime_identity | none | true | false | false | image-only Q/K 检测路径的冻结 scheduler index。 |
| detection_conditioning_protocol | persisted_protocol | runtime_identity | none | true | false | false | 三路空文本且无 CFG 的 image-only Q/K conditioning 身份。 |
| qk_layer_names | persisted_protocol | runtime_identity | none | true | false | false | runtime 必须按顺序捕获真实 Q/K 的登记 attention 层名。 |
| prompt | cross_boundary | runtime_identity | none | false | false | false | image-only Q/K backend 调用的第一路冻结空文本；不保存生成 Prompt。 |
| prompt_2 | cross_boundary | runtime_identity | none | false | false | false | image-only Q/K backend 调用的第二路冻结空文本。 |
| prompt_3 | cross_boundary | runtime_identity | none | false | false | false | image-only Q/K backend 调用的第三路冻结空文本。 |
| do_classifier_free_guidance | cross_boundary | runtime_identity | none | false | false | false | image-only Q/K forward 的实际 CFG 开关；冻结为 false。 |
| detection_timestep | cross_boundary | runtime_identity | none | true | false | false | 重新建立的冻结 detection schedule 在 index 7 选择的实际 timestep。 |
| public_noise_domain_digest | cross_boundary | provenance | none | true | false | false | image-only Q/K 公开确定性 scheduler noise 的职责域摘要。 |
| public_noise_values_float32_be_sha256 | cross_boundary | provenance | none | true | false | false | CPU row-major float32 公开噪声逐值 big-endian bytes 的可重放摘要。 |
| qk_actual_dtype | cross_boundary | runtime_identity | none | true | false | false | 两登记 attention 层实际 `to_q`/`to_k` 捕获并经模块 normalization 后的 dtype。 |
| qk_layer_observations | cross_boundary | method_state | none | false | false | false | runtime 按登记层序返回给 `main` 的真实 `QkLayerObservation` 集合；不得持久化原始 Q/K tensor。 |
| runtime_candidate_revision | persisted_protocol | provenance | none | true | false | false | execution package 与 qualification result 共同绑定的精确 40 位 Git revision。 |
| package_schema_version | persisted_protocol | protocol | none | false | false | false | runtime qualification 或 experiment-execution manifest 的局部结构版本；不创建新的治理 schema。 |
| profile_name | persisted_protocol | protocol | none | false | false | false | execution manifest 固定的 `experiment_execution_package` 提取档位名。 |
| package_ready | persisted_protocol | protocol | none | false | false | false | package builder 在精确 revision、干净树、allowlist 与安全检查全部通过后写入的布尔状态。 |
| profile | persisted_protocol | protocol | none | true | false | false | runtime qualification 运行档位：`smoke`、`qualification` 或可选 `replay`。 |
| run_status | persisted_protocol | runtime_state | none | true | false | false | runner 自动写入的 `passed`、`failed` 或 synthetic entrypoint `completed` 状态；不能表示 `runtime_verified` 或科学验证。 |
| callback_status | persisted_protocol | runtime_state | none | true | false | false | 当前 qualification record/summary 的 callback exactly-once 检查状态。 |
| actual_dtype_status | persisted_protocol | runtime_state | none | true | false | false | actual-dtype 完整性及 main-owned hard-budget 闭环的聚合状态。 |
| vae_status | persisted_protocol | runtime_state | none | true | false | false | VAE decode 与 detection-side posterior `mode()` encode 路径的完成状态。 |
| qk_status | persisted_protocol | runtime_state | none | true | false | false | 两登记层真实 `to_q`/`to_k` 捕获及身份检查的完成状态。 |
| determinism_status | persisted_protocol | runtime_state | none | true | false | false | qualification/replay 独立重复记录的一致性状态；smoke 明确为 `not_evaluated`。 |
| package_status | persisted_protocol | runtime_state | none | false | false | false | execution manifest 完整文件集、hash/size 和 revision 的启动前校验状态。 |
| dependency_status | persisted_protocol | runtime_state | none | false | false | false | requirements lock 与安装环境逐项匹配冻结 dependency lock 的状态；仅 `torch==2.11.0` 可保留符合冻结语法的 local build label。 |
| repetition_count | persisted_protocol | runtime_state | none | false | false | false | 当前结果 zip 实际完成并记录的独立执行次数。 |
| failure_class | persisted_protocol | runtime_state | none | true | false | false | qualification 的 runtime/resource/integrity/budget/QK/determinism/incomplete 分类，或内部 runner 的 `resource_failure`/`execution_failure`/`scientific_failure` 分类；只有显式方法结果可形成科学失败。 |
| failure_classes | persisted_protocol | runtime_state | none | true | false | false | run summary 按失败记录顺序保存的 failure class 列表。 |
| exception_type | persisted_protocol | runtime_state | none | false | false | false | failure record 保存的最外层 Python exception 类型名，仅供诊断。 |
| key_control | persisted_protocol | runtime_identity | none | true | false | false | qualification record 的 `registered` 或 `negative_identity` 路径；后者只验证另一 key identity 的 runtime 可执行性，不是正式 wrong-key 科学证据。 |
| key_public_digest | persisted_protocol | provenance | none | true | false | false | qualification 使用 key 的公开摘要；禁止持久化 root key 原文。 |
| prompt_identity | persisted_protocol | provenance | none | true | false | false | qualification 固定 prompt 契约的语义身份。 |
| prompt_sha256 | persisted_protocol | provenance | none | true | false | false | 本次 prompt UTF-8 bytes 的 SHA-256；不持久化私有输入。 |
| result_zip_sha256 | persisted_protocol | provenance | none | false | false | false | runner 结果 zip 的普通传输完整性摘要，不是方法或模型身份门。 |
| result_schema_version | persisted_protocol | protocol | none | false | false | false | runtime qualification 结果 JSON 的局部结构版本。 |
| started_at_utc | persisted_protocol | provenance | none | false | false | false | runner 记录的 UTC 开始时间。 |
| finished_at_utc | persisted_protocol | provenance | none | false | false | false | runner 记录的 UTC 结束时间。 |
| failure_count | persisted_protocol | runtime_state | none | false | false | false | 当前结果内自动记录的失败条目数量。 |
| checks | persisted_protocol | runtime_state | none | false | false | false | run summary 中对 `runtime_checks.jsonl` 记录的内嵌小型副本；不含原始 tensor。 |
| qk_layer_value_digests | persisted_protocol | provenance | none | true | false | false | 登记层实际 Q/K tensor 的逐层摘要集合，用于重复执行一致性检查，不持久化 tensor。 |
| qk_operator_identities | persisted_protocol | runtime_identity | none | true | false | false | 两登记层真实 attention/to_q/to_k/norm/head layout 的 operator identity 序列。 |
| query_sha256 | persisted_protocol | provenance | none | false | false | false | 单登记层实际 query tensor contiguous bytes 的 SHA-256。 |
| attention_key_sha256 | persisted_protocol | provenance | none | false | false | false | 单登记层实际 attention-key tensor contiguous bytes 的 SHA-256。 |
| package_filename | persisted_protocol | provenance | none | false | false | false | builder 创建的 execution package zip 基名。 |
| package_sha256 | persisted_protocol | provenance | none | false | false | false | execution package zip 的普通传输完整性摘要。 |
| expected_package_sha256 | persisted_protocol | provenance | none | false | false | false | 调用者从 package 外独立审核结果提供给 bootstrap 的完整 archive SHA-256；不得从 archive 同目录 sidecar 自动信任。 |
| bootstrap_schema_version | persisted_protocol | protocol | none | false | false | false | package 外可信 bootstrap 的局部协议版本；当前只支持 package schema version 1。 |
| bootstrap_failure_schema_version | persisted_protocol | protocol | none | false | false | false | runner 启动前失败诊断的局部结构版本；不属于 qualification result schema。 |
| bootstrap_sha256 | persisted_protocol | provenance | none | false | false | false | 实际执行的 package 外 bootstrap 完整文件 SHA-256，由冻结 Notebook trust anchor 先行核对。 |
| artifact_kind | persisted_protocol | runtime_state | none | false | false | false | bootstrap 输出的 qualification/experiment result、`bootstrap_failure` 或 `execution_entrypoint_failure`；任何 failure 与 synthetic result 均不得解释为 runtime/scientific evidence。 |
| failure_stage | persisted_protocol | runtime_state | none | false | false | false | bootstrap 失败发生的 arguments、secrets、archive_digest、archive_safety、manifest、dependency_install、runner_start、runner_result 或 result_copy 控制面阶段。 |
| bootstrap_exit_code | cross_boundary | runtime_state | none | false | false | false | bootstrap 自身预运行失败的退出码 3；与 runner 的 0/1/2 语义分离。 |
| runner_exit_code | cross_boundary | runtime_state | none | false | false | false | bootstrap 独立核验并复制正式结果后原样返回的 runner 退出码 0、1 或 2。 |
| diagnostic_zip | cross_boundary | provenance | none | false | false | false | runner 前失败时返回的独立 `bootstrap_failure` 诊断包绝对路径，不是 qualification result。 |
| copied_files | persisted_protocol | provenance | none | false | false | false | execution manifest 中 allowlist 实际复制文件及其摘要、大小列表。 |
| size_bytes | persisted_protocol | provenance | none | false | false | false | manifest 单文件条目的精确 byte size。 |
| sha256 | persisted_protocol | provenance | none | false | false | false | manifest 单文件内容 bytes 的 SHA-256。 |
| excluded_parts | persisted_protocol | protocol | none | false | false | false | execution package 明确禁止纳入的路径部分集合。 |
| committed_revision | persisted_protocol | provenance | none | false | false | false | experiment-execution package、bootstrap trust input 与 synthetic result 共同绑定的精确 40 位 Git revision。 |
| delivery_manifest_schema_version | persisted_protocol | protocol | none | false | false | false | package 外 delivery manifest 的局部结构版本。 |
| archive_sha256 | persisted_protocol | provenance | none | false | false | false | 确定性 experiment-execution archive 的完整 SHA-256；由 package 外 trust input 提供给 bootstrap。 |
| embedded_manifest_sha256 | persisted_protocol | provenance | none | false | false | false | delivery manifest 对 archive 内 experiment-execution manifest bytes 的 SHA-256 绑定。 |
| entrypoint_identity | persisted_protocol | provenance | none | false | false | false | package manifest、bootstrap 与结果共同绑定的包内 CLI module/function 身份。 |
| entrypoint_module | persisted_protocol | provenance | none | false | false | false | bootstrap 在全部 pre-run 检查通过后才可启动的包内 Python module。 |
| entrypoint_path | persisted_protocol | provenance | none | false | false | false | allowlist 中 package-contained CLI 的精确相对路径。 |
| evidence_scope | persisted_protocol | protocol | none | false | false | false | 明确结果只属于 infrastructure synthetic wiring、不能支撑科学 claim 的范围声明。 |
| execution_scope | persisted_protocol | protocol | none | false | false | false | package entrypoint 实际执行范围；当前固定为 `cpu_synthetic_wiring_only`。 |
| record_collection_relative_path | persisted_protocol | provenance | none | false | false | false | result root 内 governed record collection 的安全相对路径。 |
| record_collection_sha256 | persisted_protocol | provenance | none | false | false | false | package result 对 governed record collection bytes 的完整 SHA-256。 |
| scientific_claims_supported | persisted_protocol | protocol | none | false | false | false | result/diagnostic 明确是否支持科学 claim；当前 A3b synthetic 结果固定为 false。 |
| gpu_executed | persisted_protocol | runtime_state | none | false | false | false | 当前 package entrypoint 是否实际执行 GPU；synthetic wiring 固定为 false。 |
| held_out_evaluation_accessed | persisted_protocol | protocol | none | false | false | false | 当前 package entrypoint 是否访问 held-out evaluation；synthetic wiring 固定为 false。 |
| bootstrap_identity | persisted_protocol | provenance | none | false | false | false | package 外 bootstrap 的固定实现身份，必须在读取 package 前由调用者核对。 |
| entrypoint_schema_version | persisted_protocol | protocol | none | false | false | false | package-contained execution summary 的局部结构版本。 |
| metric_registry_digest | persisted_protocol | provenance | none | false | false | false | execution summary 实际 metric registry 的 SHA-256，与成功 record 的 `metric_set_digest` 一致。 |
| metric_evaluator_identity | persisted_protocol | provenance | none | false | false | false | replay 实际调用的已登记 metric evaluator 完整公开身份。 |
| metric_aggregate_identity | persisted_protocol | provenance | none | false | false | false | replay 实际返回的 aggregate 类型完整公开身份。 |
| metric_case_results | persisted_protocol | runtime_state | none | false | false | false | 从 replay 验证成功 records 实际求值得到的逐 case metric 结果，并绑定 record ID 与 canonical record digest。 |
| metric_aggregate_values | persisted_protocol | runtime_state | none | false | false | false | 已登记 evaluator 对逐 case 结果计算的 count、split、均值、改善比例及 detector/threshold 身份。 |
| metric_evidence_digest | persisted_protocol | provenance | none | false | false | false | 对 registry、evaluator/aggregate 身份、逐 case 结果及 aggregate values 的 canonical SHA-256。 |
| diagnostic_schema_version | persisted_protocol | protocol | none | false | false | false | bootstrap 或 entrypoint failure diagnostic 的局部结构版本；不属于科学结果。 |
| expected_archive_sha256 | cross_boundary | provenance | none | false | false | false | 调用者独立审核后交给 experiment bootstrap 的完整 archive SHA-256。 |
| expected_bootstrap_identity | cross_boundary | provenance | none | false | false | false | Notebook/调用者要求 package 外 bootstrap 精确匹配的实现身份。 |
| expected_bootstrap_schema_version | cross_boundary | protocol | none | false | false | false | Notebook/调用者要求 package 外 bootstrap 精确匹配的局部 schema version。 |
| expected_bootstrap_sha256 | cross_boundary | provenance | none | false | false | false | Notebook/调用者独立固定并在读取 package 前核对的 bootstrap 完整 SHA-256。 |
| expected_revision | cross_boundary | provenance | none | false | false | false | experiment bootstrap 必须与 package manifest 及结果共同核对的精确 revision。 |
| expected_candidate_config_digest | cross_boundary | method_identity | none | false | false | false | experiment bootstrap/entrypoint 必须与实际准备结果精确匹配的 candidate 摘要。 |
| expected_execution_config_digest | cross_boundary | protocol | none | false | false | false | experiment bootstrap/entrypoint 必须与实际准备结果精确匹配的 execution 摘要。 |
| expected_input_manifest_digest | cross_boundary | provenance | none | false | false | false | experiment bootstrap/entrypoint 必须与实际准备结果精确匹配的 frozen input manifest 摘要。 |
| record_count | persisted_protocol | runtime_state | none | false | false | false | synthetic execution result 中经 replay 验证的 governed record 总数。 |
| success_count | persisted_protocol | runtime_state | none | false | false | false | synthetic execution result 中成功记录数。 |
| resource_failure_count | persisted_protocol | runtime_state | none | false | false | false | synthetic execution result 中资源失败记录数。 |
| scientific_failure_count | persisted_protocol | runtime_state | none | false | false | false | synthetic execution result 中显式科学失败记录数；当前 wiring 通过不构成科学成功。 |
| execution_failure_count | persisted_protocol | runtime_state | none | false | false | false | synthetic execution result 中执行失败记录数。 |
| excluded_count | persisted_protocol | runtime_state | none | false | false | false | synthetic execution result 中预登记排除记录数。 |
| replay_digest | persisted_protocol | provenance | none | false | false | false | A3a replay 对完整 record collection 形成的稳定摘要。 |
| result_zip | persisted_protocol | provenance | none | false | false | false | runner/bootstrap 返回的结果 zip 路径或基名；权威归档位置由调用方边界决定。 |
| result_zip_filename | persisted_protocol | provenance | none | true | false | false | 写入 run summary、且必须包含同一 run ID 的最小结果 zip 基名。 |
| key_controls | persisted_protocol | runtime_identity | none | true | false | false | 当前结果按执行顺序记录的 `registered`/`negative_identity` key role 序列。 |
| replay_source_run_id | persisted_protocol | provenance | none | true | false | false | replay 绑定的既有 passed qualification run ID。 |
| replay_source_revision | persisted_protocol | provenance | none | true | false | false | replay 绑定的既有 qualification runtime candidate revision。 |
| replay_source_record_digests | persisted_protocol | provenance | none | true | false | false | 从既有 qualification `runtime_checks.jsonl` 独立重算并与 summary 对照的记录摘要序列。 |
| record_digests | persisted_protocol | provenance | none | true | false | false | 当前 runner 对完整 qualification records 逐条 canonical JSON 计算的摘要序列。 |
| clean_image_sha256 | persisted_protocol | provenance | none | false | false | false | clean VAE decode 输出 tensor contiguous bytes 的摘要；不持久化 tensor。 |
| watermarked_image_sha256 | persisted_protocol | provenance | none | false | false | false | watermarked VAE decode 输出 tensor contiguous bytes 的摘要；不持久化 tensor。 |
| detection_latent_sha256 | persisted_protocol | provenance | none | false | false | false | detection-side posterior-mode latent contiguous bytes 的摘要；不持久化 tensor。 |
| python | persisted_protocol | runtime_identity | none | false | false | false | qualification 环境实际 Python 版本。 |
| torch | persisted_protocol | runtime_identity | none | false | false | false | qualification 环境实际 PyTorch 版本。 |
| diffusers | persisted_protocol | runtime_identity | none | false | false | false | qualification 环境实际 diffusers 版本。 |
| transformers | persisted_protocol | runtime_identity | none | false | false | false | qualification 环境实际 transformers 版本。 |
| accelerate | persisted_protocol | runtime_identity | none | false | false | false | qualification 环境实际 accelerate distribution 版本。 |
| numpy | persisted_protocol | runtime_identity | none | false | false | false | qualification 环境实际 NumPy distribution 版本。 |
| pillow | persisted_protocol | runtime_identity | none | false | false | false | qualification 环境实际 Pillow distribution 版本。 |
| safetensors | persisted_protocol | runtime_identity | none | false | false | false | qualification 环境实际 safetensors distribution 版本。 |
| cuda_runtime | persisted_protocol | runtime_identity | none | false | false | false | PyTorch 报告的 CUDA runtime 版本。 |
| cuda_available | persisted_protocol | runtime_identity | none | false | false | false | runner 启动时 PyTorch 是否报告 CUDA 可用。 |
| gpu_name | persisted_protocol | runtime_identity | none | false | false | false | qualification 实际 CUDA device 0 名称；不是 GPU 能力证明的替代物。 |
| dependency_lock_evidence | persisted_protocol | runtime_identity | none | true | false | false | runner 对完整冻结 dependency lock 逐项核验后写入的期望/实际版本记录；PyTorch local build label 不得从实际版本中剥离。 |
| expected_version | persisted_protocol | runtime_identity | none | false | false | false | 单项 dependency lock 登记的冻结版本或 Python 版本约束。 |
| actual_version | persisted_protocol | runtime_identity | none | false | false | false | runner 通过 Python runtime 或 `importlib.metadata` 实际读取并原样保留的完整版本，包括获准的 PyTorch local build label。 |
| huggingface_hub | persisted_protocol | runtime_identity | none | false | false | false | environment summary 中 `huggingface-hub` distribution 的实际版本。 |
| materialization_attempt_count | persisted_protocol | runtime_state | none | true | false | false | 当前 qualification record 保存的 main-owned actual-dtype materialization 尝试次数。 |
| dependency_lock | persisted_protocol | runtime_identity | none | false | false | false | runtime 候选冻结的 Python 与模型执行依赖版本映射。 |
| package_name | persisted_protocol | runtime_identity | none | false | false | false | runtime dependency lock 中按冻结顺序登记的包名。 |
| version_specifier | persisted_protocol | runtime_identity | none | false | false | false | runtime dependency lock 中与包名绑定的精确版本或 Python 版本约束。 |
| cpu_available | cross_boundary | runtime_identity | none | false | false | false | backend 在加载模型前报告 CPU 是否可供控制流使用。 |
| cuda_device_count | cross_boundary | runtime_identity | none | false | false | false | backend 在加载模型前报告的非负 CUDA 设备数量。 |
| runtime_backend_name | cross_boundary | runtime_identity | none | false | false | false | 实际准备 runtime session 的 backend 实现身份。 |
| selected_device | cross_boundary | runtime_identity | none | false | false | false | adapter 根据请求与可用设备确定的实际执行设备。 |
| identity_schema_version | cross_boundary | protocol | none | false | false | false | runtime public execution identity 的局部 canonical mapping 版本。 |
| backend_type_identity | cross_boundary | runtime_identity | none | false | false | false | runtime adapter 构造时锚定且每次公开重验证的 backend 精确类型身份；不暴露 backend 对象。 |
| qk_observation_callable_identity | cross_boundary | runtime_identity | none | false | false | false | runtime adapter 构造时惰性锚定的 Batch-3 Q/K module 精确函数 qualified identity；公开值只含稳定字符串，不暴露 callable 对象。 |
| backend_resources_owned | cross_boundary | runtime_state | none | false | false | false | runtime public identity 中与 lifecycle state 联合复验的资源所有权布尔值。 |
| runtime_state | cross_boundary | runtime_state | none | false | false | false | runtime public execution identity 当前 `created`、`ready`、`failed` 或 `closed` 状态。 |
| runtime_session_identity_digest | cross_boundary | provenance | none | false | false | false | READY session 全部公开配置/设备/backend identity 的 canonical SHA-256；不包含模型私有状态。 |
| seed | persisted_protocol | protocol | none | true | false | false | 当前 record 实际使用的随机种子。 |
| metric_name | persisted_protocol | protocol | none | true | false | false | 实验记录中的指标名称。 |
| metric_value | persisted_protocol | protocol | none | true | false | false | 实验记录中的指标数值。 |
| execution_status | persisted_protocol | protocol | none | true | false | false | 当前尝试成功、失败或被排除的显式状态。 |

## 内部科学验证协议字段

以下字段属于 `ceg_wm_internal_sample_record_v4`，并由
`ceg_wm_internal_run_case_record_collection_v1` 聚合。可执行字段权威位于随执行包分发的
`experiments/protocol/internal_record_registry.py`；本页由开发侧治理测试检查同步，
不作为 runner 的运行时输入。既有的 `split`、
`source_cluster_id`、`execution_status`、`lf_score`、`hf_score`、`combined_score`、
`tau`、`tau_rescue`、`geometry_triggered`、`raw_content_score` 和
`rectified_content_score` 继续使用上表既有语义，不在此重复登记。

| field_name | governance_level | category | required_suffix | allowed_in_records | allowed_in_claims | replacement_required | description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| protocol_id | persisted_protocol | protocol | none | true | false | false | 冻结内部科学验证协议的语义身份。 |
| protocol_version | persisted_protocol | protocol | none | true | false | false | 冻结内部协议配置版本。 |
| record_schema_version | persisted_protocol | protocol | none | true | false | false | 逐样本内部验证 record schema 身份。 |
| record_collection_schema_version | persisted_protocol | protocol | none | true | false | false | 一个 run/case 有序 records 集合的冻结 schema 身份。 |
| record_collection_binding_fields | persisted_protocol | protocol | none | false | false | false | `internal_validation.validate_run_case_record_collection` 只接受精确冻结 protocol/manifest dataclass trust anchors，并把 collection 与两者重算摘要逐值绑定。 |
| maximum_record_attempts | persisted_protocol | protocol | none | true | false | false | 每个 unit/case/source cluster 在冻结协议中允许的最大执行尝试数。 |
| retryable_parent_statuses | persisted_protocol | protocol | none | false | false | false | 允许作为后续 attempt parent 的冻结 outcome 集合，当前为 `failed` 与 `retry`。 |
| retry_parent_required_after_attempt_zero | persisted_protocol | protocol | none | false | false | false | 是否要求所有 attempt index 大于零的 outcome 显式绑定 parent record。 |
| analysis_unit_identity | persisted_protocol | provenance | none | true | false | false | unit、case、source cluster 与 Prompt/seed/image-lineage/key-family 的不可拆分身份结构。 |
| unit_id | persisted_protocol | provenance | none | true | false | false | 一个 case 中被执行和记录的独立分析单位身份。 |
| prompt_digest | persisted_protocol | provenance | none | true | false | false | 分析单位所绑定 prompt 的内容摘要，不保存 prompt 明文。 |
| generation_seed | persisted_protocol | provenance | none | true | false | false | 分析单位用于 source-cluster 隔离的冻结生成 seed。 |
| image_lineage_digest | persisted_protocol | provenance | none | true | false | false | 分析单位所绑定图像 lineage 的内容摘要。 |
| registered_key_family_digest | persisted_protocol | provenance | none | true | false | false | 分析单位所绑定 registered-key family 的公共摘要。 |
| detector_mode | persisted_protocol | method_identity | none | false | false | false | 内部协议选择前置门的正式 detector mode；当前仅 `hf_only` 与 `combined`，缺失或未知值 fail closed。 |
| source_row | persisted_protocol | provenance | none | false | false | false | C1 pinned PartiPrompts snapshot 中从 1 开始的原始数据行身份。 |
| prompt_text | persisted_protocol | protocol_input | none | false | false | false | C1 离线执行所需的冻结 prompt 明文；其 UTF-8 摘要必须逐行等于 `prompt_digest`，不得由网络重取替代。 |
| roster_rows_digest | persisted_protocol | provenance | none | false | false | false | C1 prompt/category/challenge/source-row roster 的 canonical SHA-256。 |
| candidate_specification_path | persisted_protocol | provenance | none | false | false | false | C1 候选绑定所指向的权威候选规格文件路径；bundle loader 必须按 `candidate_specification_sha256` 对其实际字节 fail closed。 |
| candidate_specification_sha256 | persisted_protocol | provenance | none | false | false | false | `candidate_specification_path` 所指权威候选规格文件的原始字节 SHA-256。 |
| candidate_binding_digest | persisted_protocol | method_identity | none | true | false | false | C1 HF reference 候选、权威候选规格、完整 source bundle、method adapter、runtime config 与 qualification 事实的 canonical 摘要；不是结果门。 |
| hf_only_tau_frozen | persisted_protocol | protocol | none | false | false | false | threshold-fit 回传经独立审计后形成的冻结 tau artifact gate；未绑定 artifact SHA/revision/APPROVE 时 confirmation fail closed。 |
| run_phase_id | persisted_protocol | protocol | none | true | false | false | C1 threshold-fit 或 untouched-confirmation 的互斥执行阶段身份，禁止在同一 run 混跑。 |
| metric_id | persisted_protocol | metric_identity | none | true | false | false | C1-M 必须实现并由 split binding 授权的预注册 metric 身份；C1-P 仅冻结公式身份。 |
| case_id | persisted_protocol | protocol | none | true | false | false | 预登记内部科学问题、攻击和 control 组合的 case 身份。 |
| record_sequence_index | persisted_protocol | protocol | none | true | false | false | 一个 run/case record collection 中从零开始且连续的序列索引。 |
| record_attempt_index | persisted_protocol | protocol | none | true | false | false | 同一 unit/case/source cluster 执行尝试的从零开始连续索引；retry 必须大于零。 |
| exclusion_rule_id | persisted_protocol | protocol | none | true | false | false | `excluded` 状态绑定的预登记、与效果无关的排除规则身份。 |
| retry_of_record_id | persisted_protocol | provenance | none | true | false | false | 任一非初始 attempt 回绑的前一可重试 outcome record 身份；attempt zero 必须为空。 |
| detector_trace | persisted_protocol | method_state | none | true | false | false | raw/rectified detector、preprocessing 与对应内容分数的完整逐样本结构。 |
| raw_detector_identity | persisted_protocol | method_identity | none | true | false | false | 原图实际调用的冻结 content detector 身份。 |
| rectified_detector_identity | persisted_protocol | method_identity | none | true | false | false | 回正图绑定的 content detector 身份；必须等于 raw 身份。 |
| raw_detector_config_digest | persisted_protocol | method_identity | none | true | false | false | 原图 content detector 完整配置身份摘要。 |
| rectified_detector_config_digest | persisted_protocol | method_identity | none | true | false | false | 回正图 content detector 完整配置身份摘要；必须等于 raw 摘要。 |
| raw_preprocessing_identity | persisted_protocol | method_identity | none | true | false | false | 原图普通图像检测预处理身份。 |
| rectified_preprocessing_identity | persisted_protocol | method_identity | none | true | false | false | 回正图普通图像检测预处理身份；必须等于 raw 身份。 |
| branch_score_trace | persisted_protocol | method_statistic | none | true | false | false | LF、HF 和 combined 三个独立分数的逐样本结构。 |
| routing_trace | persisted_protocol | method_state | none | true | false | false | routing identity/control 与 observation/mask 摘要的逐样本结构。 |
| routing_identity | persisted_protocol | method_identity | none | true | false | false | routed 或 uniform-control 执行实际绑定的路由候选身份。 |
| routing_control | persisted_protocol | method_state | none | true | false | false | routed、uniform disabled 或分支禁用 control 身份。 |
| routing_observation_digest | persisted_protocol | provenance | none | true | false | false | 本样本路由 observation 结构的内容摘要。 |
| routing_mask_digest | persisted_protocol | provenance | none | true | false | false | 本样本实际 routing masks 的内容摘要。 |
| geometry_trace | persisted_protocol | method_state | none | true | false | false | 几何触发、估计、原始指标、可靠性、变换、失败与回正状态的完整结构。 |
| geometry_operation_identity | persisted_protocol | method_identity | none | true | false | false | 当前 record 执行前已冻结的公开几何估计 callable 声明身份。 |
| geometry_reliability_config_digest | persisted_protocol | protocol | none | true | false | false | experiments 层对当前 record 执行前几何可靠性阈值声明计算的 canonical 配置摘要；无阈值时为空，不等同于 main 方法结果的 threshold config identity。 |
| geometry_estimation_identity | persisted_protocol | method_identity | none | true | false | false | 实际几何估计结果及 search config 的绑定身份。 |
| geometry_reliability_identity | persisted_protocol | method_identity | none | true | false | false | 独立 reliability fit 与合取结果的绑定身份。 |
| geometry_reliable | persisted_protocol | method_state | none | true | false | false | 当前逐样本几何结果是否满足冻结独立可靠性门。 |
| geometry_transform | persisted_protocol | method_statistic | none | true | false | false | 估计的有界 crop/scale/rotation/translation 参数 mapping。 |
| geometry_raw_metrics | persisted_protocol | method_statistic | none | true | false | false | coverage、uniqueness、gap、key margin、inlier、residual 等 estimator 原始指标 mapping。 |
| geometry_failure_reason | persisted_protocol | method_state | none | true | false | false | 几何估计、可靠性或回正失败的明确原因。 |
| rectification_status | persisted_protocol | method_state | none | true | false | false | `not_attempted`、`succeeded` 或 `failed`。 |
| threshold_trace | persisted_protocol | method_identity | none | true | false | false | raw/rectified threshold identity 与同一 `tau`、`tau_rescue` 的结构。 |
| raw_threshold_identity | persisted_protocol | method_identity | none | true | false | false | 原图判定实际使用的 threshold/calibration 身份。 |
| rectified_threshold_identity | persisted_protocol | method_identity | none | true | false | false | 回正重判 threshold 身份；必须等于 raw 身份。 |
| key_control_trace | persisted_protocol | method_identity | none | true | false | false | registered key family、实际 detection key role 和 control 身份结构。 |
| registered_key_public_digest | persisted_protocol | provenance | none | true | false | false | 当前 source cluster 注册 key family 的不可逆公开摘要。 |
| detection_key_public_digest | persisted_protocol | provenance | none | true | false | false | 本次逐样本检测实际使用 key 的不可逆公开摘要。 |
| control_identity | persisted_protocol | protocol | none | true | false | false | registered、wrong-key、unwatermarked、route-disabled 或其他预登记 control 身份。 |
| decision_trace | persisted_protocol | method_state | none | true | false | false | final decision、只允许的 content positive source 与决策原因结构。 |
| watermark_decision | persisted_protocol | method_state | none | true | false | false | 水印判定的 `positive`、`negative`、`failed`、`excluded` 或 `retry`。 |
| decision_reason | persisted_protocol | method_state | none | true | false | false | 最终判定或非成功状态的明确原因。 |
| provenance_trace | persisted_protocol | provenance | none | true | false | false | protocol/split/method/model/environment/input/attack/metric 的完整不可变摘要结构。 |
| promotion_gate_assessments | persisted_protocol | protocol | none | true | false | false | 一个 run/case 中按顺序保存、由具体 records 支撑的结构化晋升门裁决。 |
| promotion_stop_gate_id | persisted_protocol | protocol | none | true | false | false | 首个失败且触发停止的已登记 promotion gate 身份；无失败时为空。 |
| gate_id | persisted_protocol | protocol | none | true | false | false | 13 职责矩阵中已登记的 promotion gate 身份。 |
| gate_status | persisted_protocol | protocol | none | true | false | false | promotion gate 的 `passed` 或 `failed` 状态。 |
| evidence_record_ids | persisted_protocol | provenance | none | true | false | false | promotion gate 实际消费且必须存在于同一 run/case collection 的 record IDs。 |
| stop_outcome | persisted_protocol | protocol | none | true | false | false | 失败 gate 对应的冻结负结果或返回前置门 outcome。 |
| environment_digest | persisted_protocol | provenance | none | true | false | false | 当前执行环境与依赖锁的内容摘要。 |
| input_manifest_digest | persisted_protocol | provenance | none | true | false | false | 当前 runner 消费的冻结逐 case 输入 manifest 内容摘要，独立于 split manifest。 |
| candidate_config_digest | persisted_protocol | method_identity | none | true | false | false | 当前执行所绑定候选集合与候选配置的内容摘要。 |
| execution_config_digest | persisted_protocol | protocol | none | true | false | false | method adapter、attack registry、metric registry 与 runner 配置的联合摘要。 |
| resource_identity_digest | persisted_protocol | runtime_identity | none | true | false | false | 当前设备、dtype、依赖与资源分配身份的冻结摘要；资源失败不得解释为科学失败。 |

以下 input-manifest 字段只描述 runner 的冻结输入边界，不直接支撑 claim：

| field_name | governance_level | category | required_suffix | allowed_in_records | allowed_in_claims | replacement_required | description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| manifest_schema_version | persisted_protocol | protocol | none | false | false | false | 内部逐 case input manifest 的结构身份。 |
| manifest_id | persisted_protocol | protocol | none | false | false | false | 冻结 input manifest 的稳定身份。 |
| manifest_revision | persisted_protocol | provenance | none | false | false | false | 冻结 input manifest 的修订身份。 |
| entries | persisted_protocol | protocol | none | false | false | false | input manifest 中按 unit 保存的公开 case entries；原始密钥不进入该集合。 |
| execution_expectation | persisted_protocol | method_identity | none | false | false | false | 每个 unit 在执行前冻结的 detector、threshold、几何操作与可靠性阈值声明集合。 |
| content_detector_binding_digest | persisted_protocol | method_identity | none | false | false | false | 冻结 content detector callable 公开声明、配置、预处理与 key role 的联合摘要。 |
| content_operation_config_digest | persisted_protocol | method_identity | none | false | false | false | content callable 通过 formal runner semantic declaration 暴露并可重复重算的 experiments 配置摘要。 |
| geometry_operation_config_digest | persisted_protocol | method_identity | none | false | false | false | geometry callable 通过 formal runner semantic declaration 暴露并可重复重算的 experiments 配置摘要。 |
| input_artifact_digest | persisted_protocol | provenance | none | true | false | false | 当前普通输入或生成产物的稳定内容摘要。 |
| attack_config_digest | persisted_protocol | provenance | none | true | false | false | 当前 case 实际绑定的预登记攻击配置摘要。 |

## A-2 内部执行组件字段

以下字段属于 `internal_execution_components.json` 及 methods、attacks、metrics
公开 dataclass 表面。它们只支持内部组件执行与分析；进入 governed records 时仍须
由 runner 显式映射到已冻结 record schema，不能直接支撑科学 claim。

| field_name | governance_level | category | required_suffix | allowed_in_records | allowed_in_claims | replacement_required | description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| schema_version | persisted_protocol | protocol | none | false | false | false | A-2 内部执行组件 JSON 的冻结 schema 身份。 |
| registry_version | persisted_protocol | protocol | none | false | false | false | method、attack 或 metric registry 的冻结版本身份。 |
| method_adapter | persisted_protocol | method_identity | none | false | false | false | 方法薄适配器配置段。 |
| attack_registry | persisted_protocol | protocol | none | false | false | false | 几何攻击 registry 配置段。 |
| metric_registry | persisted_protocol | protocol | none | false | false | false | 内部指标 registry 配置段。 |
| adapter_id | persisted_protocol | method_identity | none | false | false | false | CEG-WM 内部实验适配器身份。 |
| adapter_version | persisted_protocol | method_identity | none | false | false | false | CEG-WM 内部实验适配器版本。 |
| component_bindings | persisted_protocol | method_identity | none | false | false | false | 13 个方法职责到公开调用及结果身份字段的有序绑定。 |
| key_schedule_operations | persisted_protocol | method_identity | none | false | false | false | 四种 key schedule 操作到实际公开调用的冻结有序绑定。 |
| operation_id | persisted_protocol | method_identity | none | false | false | false | key schedule 操作在实验适配器中的稳定身份。 |
| responsibility | cross_boundary | method_identity | none | false | false | false | 一次适配调用实际承担的冻结方法职责。 |
| public_callable | persisted_protocol | method_identity | none | false | false | false | 方法职责委托的公开 Python 调用身份。 |
| result_identity_field | persisted_protocol | method_identity | none | false | false | false | 适配器从真实结果读取的不可空身份字段名。 |
| adapter_config_digest | cross_boundary | provenance | none | false | false | false | 一次组件调用绑定的适配器配置摘要。 |
| result_type | cross_boundary | provenance | none | false | false | false | 一次真实委托结果的完整 Python 类型身份。 |
| result_identity | cross_boundary | provenance | none | false | false | false | 一次真实委托结果公开、不可空的身份值。 |
| upstream_runtime_identity | cross_boundary | provenance | none | false | false | false | Q/K 同步调用所绑定的 runtime public observation 摘要。 |
| result | cross_boundary | method_state | none | false | false | false | 薄适配器原样返回的真实 `main` 或 `runtime` 公开结果。 |
| attack_ids | persisted_protocol | protocol | none | false | false | false | 几何攻击 registry 的冻结有序攻击身份集合。 |
| attack_id | cross_boundary | protocol | none | false | false | false | 当前几何攻击或攻击结果身份。 |
| image_interpolation | persisted_protocol | protocol | none | false | false | false | 攻击 registry 冻结的图像插值方式。 |
| image_padding | persisted_protocol | protocol | none | false | false | false | 攻击 registry 冻结的越界 padding 方式。 |
| align_corners | persisted_protocol | protocol | none | false | false | false | affine grid 与 sampling 是否使用对齐角点语义。 |
| output_quantization | persisted_protocol | protocol | none | false | false | false | 攻击输出从浮点图像回到 RGB8 的冻结量化规则。 |
| output_size_policy | persisted_protocol | protocol | none | false | false | false | 攻击输出空间尺寸策略。 |
| parameter_bounds | persisted_protocol | protocol | none | false | false | false | crop、scale 与 rotation 参数边界配置段。 |
| crop_fraction_bounds | cross_boundary | protocol | none | false | false | false | 攻击 registry 解析出的 crop fraction 上下界。 |
| scale_factor_bounds | cross_boundary | protocol | none | false | false | false | 攻击 registry 解析出的 scale factor 上下界。 |
| rotation_degrees_bounds | cross_boundary | protocol | none | false | false | false | 攻击 registry 解析出的 rotation degree 上下界。 |
| registry_digest | cross_boundary | provenance | none | false | false | false | 当前 attack 或 metric registry 的稳定摘要。 |
| attack_registry_digest | cross_boundary | provenance | none | false | false | false | 一次攻击结果绑定的攻击 registry 摘要。 |
| crop_fraction | cross_boundary | protocol | none | false | false | false | output-to-input affine 使用的裁剪保留比例。 |
| scale_factor | cross_boundary | protocol | none | false | false | false | output-to-input affine 使用的尺度因子。 |
| rotation_degrees | cross_boundary | protocol | none | false | false | false | output-to-input affine 使用的旋转角度。 |
| image | cross_boundary | artifact | none | false | false | false | 攻击组件消费或产生的 RGB8 `[1,3,H,W]` tensor。 |
| image_digest | cross_boundary | provenance | none | false | false | false | RGB8 dtype、shape 和逐值字节的稳定摘要。 |
| source_artifact_digest | cross_boundary | provenance | none | false | false | 攻击前源 RGB8 artifact 摘要。 |
| attacked_artifact | cross_boundary | artifact | none | false | false | 保留分析单位身份的攻击后 RGB8 artifact。 |
| output_to_input_matrix | cross_boundary | protocol | none | false | false | 攻击实际使用的二维 output-to-input affine 矩阵。 |
| interpolation | cross_boundary | protocol | none | false | false | 单次攻击结果实际绑定的插值方式。 |
| padding | cross_boundary | protocol | none | false | false | 单次攻击结果实际绑定的 padding 方式。 |
| analysis_unit | persisted_protocol | protocol | none | false | false | false | metric registry 冻结的 unit/case/source-cluster 分析单位。 |
| forbidden_split | persisted_protocol | protocol | none | false | false | false | metric 层明确拒绝访问的 split。 |
| metric_ids | persisted_protocol | protocol | none | false | false | false | 内部 metric registry 的冻结有序指标身份集合。 |
| metric_split_bindings | persisted_protocol | protocol | none | false | false | false | 每个内部 metric 身份到合法 split 集合的冻结有序绑定。 |
| allowed_splits | persisted_protocol | protocol | none | false | false | false | 单个内部 metric 获准消费的 split 身份有序集合。 |
| metric_registry_digest | cross_boundary | provenance | none | false | false | false | 指标输入或结果绑定的 metric registry 摘要。 |
| target_fpr | cross_boundary | method_statistic | none | false | false | false | fixed-FPR 或 rescue safety 使用的预登记目标假阳性率。 |
| threshold | cross_boundary | method_statistic | none | false | false | false | primary-null calibration 得到的有限冻结内容分数阈值；通用 metric 层不臆定 detector-specific 范围。 |
| false_positive_count | cross_boundary | method_statistic | none | false | false | false | fixed-FPR threshold fit 中观察到的 primary-null 假阳性数量。 |
| primary_null_count | cross_boundary | method_statistic | none | false | false | false | primary-null case 数量；wrong-key 不计入。 |
| empirical_fpr | cross_boundary | method_statistic | none | false | false | false | threshold fit primary null 上的经验 FPR。 |
| confidence_level | cross_boundary | protocol | none | false | false | false | FPR 单侧置信上界使用的冻结置信水平。 |
| fpr_upper_confidence_bound | cross_boundary | method_statistic | none | false | false | false | threshold-fit FPR 的单侧 Clopper-Pearson 上界。 |
| source_cluster_digest | cross_boundary | provenance | none | false | false | false | threshold fit 所消费 source-cluster 身份有序集合的摘要。 |
| calibration_case_digest | cross_boundary | provenance | none | false | false | false | threshold fit 所消费 split、unit、case、source-cluster、key-role 与 score 规范序列的摘要。 |
| decisions | cross_boundary | method_statistic | none | false | false | false | fixed-threshold evaluation 的逐 case 检测决定集合。 |
| positive | cross_boundary | method_statistic | none | false | false | false | 单个检测 case 是否达到冻结阈值。 |
| registered_tpr | cross_boundary | method_statistic | none | false | false | false | registered-positive cases 的真阳性率。 |
| registered_positive_count | cross_boundary | method_statistic | none | false | false | false | registered-positive case 数量。 |
| primary_null_fpr | cross_boundary | method_statistic | none | false | false | false | evaluation primary-null cases 的经验 FPR。 |
| primary_null_fpr_upper_confidence_bound | cross_boundary | method_statistic | none | false | false | false | evaluation primary-null FPR 的单侧 Clopper-Pearson 上界。 |
| wrong_key_positive_rate | cross_boundary | method_statistic | none | false | false | false | 独立 wrong-key attribution null 的阳性率。 |
| wrong_key_count | cross_boundary | method_statistic | none | false | false | false | wrong-key attribution case 数量。 |
| condition_identity | cross_boundary | protocol | none | false | false | false | 图像质量 case 的实际方法或 control 条件身份。 |
| budget_identity | cross_boundary | protocol | none | false | false | false | matched-budget 质量比较绑定的共同预算身份。 |
| reference_values | cross_boundary | method_statistic | none | false | false | false | 质量度量 reference 向量。 |
| candidate_values | cross_boundary | method_statistic | none | false | false | false | 质量度量 candidate 向量。 |
| relative_l2 | cross_boundary | method_statistic | none | false | false | false | 单 case candidate-reference 相对 L2。 |
| mean_relative_l2 | cross_boundary | method_statistic | none | false | false | false | matched-budget cases 的相对 L2 宏平均。 |
| mean_squared_error | cross_boundary | method_statistic | none | false | false | false | 单 case 或 matched-budget cases 的均方误差。 |
| cases | cross_boundary | method_statistic | none | false | false | false | 保留 unit、case、source-cluster 身份的逐 case 结果集合。 |
| routed_positive | cross_boundary | method_statistic | none | false | false | false | routed 条件是否产生内容阳性。 |
| uniform_control_positive | cross_boundary | method_statistic | none | false | false | false | matched uniform-control 条件是否产生内容阳性。 |
| routed_quality_mse | cross_boundary | method_statistic | none | false | false | false | routed 条件的 matched-budget 质量 MSE。 |
| uniform_control_quality_mse | cross_boundary | method_statistic | none | false | false | false | uniform-control 条件的 matched-budget 质量 MSE。 |
| routed_budget_identity | cross_boundary | protocol | none | false | false | false | routed 条件绑定的预算身份。 |
| uniform_control_budget_identity | cross_boundary | protocol | none | false | false | false | uniform-control 条件绑定的预算身份。 |
| detection_gain | cross_boundary | method_statistic | none | false | false | false | 单 case routed 阳性相对 uniform-control 的差值。 |
| quality_non_degradation | cross_boundary | method_statistic | none | false | false | false | 单 case uniform MSE 减 routed MSE；非负表示未劣化。 |
| per_case_detection_gain | cross_boundary | method_statistic | none | false | false | false | routing cases 的 detection gain 有序集合。 |
| per_case_quality_non_degradation | cross_boundary | method_statistic | none | false | false | false | routing cases 的 quality non-degradation 有序集合。 |
| mean_detection_gain | cross_boundary | method_statistic | none | false | false | false | routing detection gain 宏平均。 |
| mean_quality_non_degradation | cross_boundary | method_statistic | none | false | false | false | routing quality non-degradation 宏平均。 |
| hf_positive | cross_boundary | method_statistic | none | false | false | false | 单 case HF 分支是否阳性。 |
| lf_positive | cross_boundary | method_statistic | none | false | false | false | 单 case LF 分支是否阳性。 |
| combined_positive | cross_boundary | method_statistic | none | false | false | false | 单 case 预登记组合判定是否阳性。 |
| lf_complements_hf | cross_boundary | method_statistic | none | false | false | false | 单 registered case 是否 HF 阴性但 LF 阳性。 |
| combined_gain_over_hf | cross_boundary | method_statistic | none | false | false | false | 单 registered case 是否组合相对 HF 增加阳性。 |
| combined_regression_from_hf | cross_boundary | method_statistic | none | false | false | false | 单 registered case 是否组合丢失 HF 阳性。 |
| registered_count | cross_boundary | method_statistic | none | false | false | false | LF/HF complementarity 中 registered cases 数量。 |
| lf_complements_hf_count | cross_boundary | method_statistic | none | false | false | false | HF 阴性而 LF 阳性的 registered case 数量。 |
| combined_gain_over_hf_count | cross_boundary | method_statistic | none | false | false | false | 组合相对 HF 新增阳性的 registered case 数量。 |
| combined_regression_from_hf_count | cross_boundary | method_statistic | none | false | false | false | 组合相对 HF 丢失阳性的 registered case 数量。 |
| wrong_key_combined_positive_rate | cross_boundary | method_statistic | none | false | false | false | wrong-key cases 上组合阳性率。 |
| expected_rotation_degrees | cross_boundary | protocol | none | false | false | false | 攻击配置提供的预期旋转角。 |
| estimated_rotation_degrees | cross_boundary | method_statistic | none | false | false | false | 几何估计器输出的旋转角。 |
| expected_scale | cross_boundary | protocol | none | false | false | false | 攻击配置提供的预期尺度。 |
| estimated_scale | cross_boundary | method_statistic | none | false | false | false | 几何估计器输出的尺度。 |
| expected_translation_x | cross_boundary | protocol | none | false | false | false | 攻击配置提供的预期横向位移。 |
| estimated_translation_x | cross_boundary | method_statistic | none | false | false | false | 几何估计器输出的横向位移。 |
| expected_translation_y | cross_boundary | protocol | none | false | false | false | 攻击配置提供的预期纵向位移。 |
| estimated_translation_y | cross_boundary | method_statistic | none | false | false | false | 几何估计器输出的纵向位移。 |
| rotation_absolute_error | cross_boundary | method_statistic | none | false | false | false | 单 case wrap-aware 旋转绝对误差。 |
| scale_absolute_error | cross_boundary | method_statistic | none | false | false | false | 单 case 尺度绝对误差。 |
| translation_euclidean_error | cross_boundary | method_statistic | none | false | false | false | 单 case 二维位移欧氏误差。 |
| mean_rotation_absolute_error | cross_boundary | method_statistic | none | false | false | false | 旋转绝对误差宏平均。 |
| mean_scale_absolute_error | cross_boundary | method_statistic | none | false | false | false | 尺度绝对误差宏平均。 |
| mean_translation_euclidean_error | cross_boundary | method_statistic | none | false | false | false | 位移欧氏误差宏平均。 |
| mean_coverage | cross_boundary | method_statistic | none | false | false | false | 几何 estimation coverage 宏平均。 |
| expected_recoverable | cross_boundary | protocol | none | false | false | false | reliability case 的预登记可恢复标签。 |
| recoverable_accept_rate | cross_boundary | method_statistic | none | false | false | false | 预期可恢复 cases 被 reliability 接受的比例。 |
| unrecoverable_reject_rate | cross_boundary | method_statistic | none | false | false | false | 预期不可恢复 cases 被 reliability 拒绝的比例。 |
| false_reliable_rate | cross_boundary | method_statistic | none | false | false | false | 预期不可恢复 cases 被错误标为可靠的比例。 |
| recoverable_count | cross_boundary | method_statistic | none | false | false | false | 预期可恢复 case 数量。 |
| unrecoverable_count | cross_boundary | method_statistic | none | false | false | false | 预期不可恢复 case 数量。 |
| rectified_score | cross_boundary | method_statistic | none | false | false | false | 同 detector/threshold 回正图内容分数。 |
| score_delta | cross_boundary | method_statistic | none | false | false | false | 单 case rectified score 减 raw score。 |
| per_case_score_delta | cross_boundary | method_statistic | none | false | false | false | rectification cases 的 score delta 有序集合。 |
| mean_score_delta | cross_boundary | method_statistic | none | false | false | false | rectification score delta 宏平均。 |
| improved_fraction | cross_boundary | method_statistic | none | false | false | false | rectified score 严格高于 raw score 的 case 比例。 |
| raw_positive | cross_boundary | method_statistic | none | false | false | false | primary-null case 是否在 raw 路径假阳性。 |
| rescue_triggered | cross_boundary | method_state | none | false | false | false | primary-null case 是否实际触发 rescue 路径。 |
| rectified_positive | cross_boundary | method_statistic | none | false | false | false | 已触发 rescue 的回正重判是否阳性。 |
| watermark_decision_positive | cross_boundary | method_statistic | none | false | false | false | 严格由 raw 阳性或已触发 rescue 后的 rectified 阳性导出的最终水印判定。 |
| raw_false_positive | cross_boundary | method_statistic | none | false | false | false | 单 primary-null case 的 raw 假阳性标记。 |
| rescue_additional_false_positive | cross_boundary | method_statistic | none | false | false | false | 单 case 由 rescue 新增的假阳性标记。 |
| global_false_positive | cross_boundary | method_statistic | none | false | false | false | 单 case raw 或 rescue 联合路径假阳性标记。 |
| raw_fpr | cross_boundary | method_statistic | none | false | false | false | primary null 上 raw 路径经验 FPR。 |
| rescue_additional_fpr | cross_boundary | method_statistic | none | false | false | false | primary null 上 rescue 路径新增经验 FPR。 |
| global_fpr | cross_boundary | method_statistic | none | false | false | false | primary null 上完整 raw+rescue 联合经验 FPR。 |
| global_fpr_upper_confidence_bound | cross_boundary | method_statistic | none | false | false | false | 完整联合 FPR 的单侧 Clopper-Pearson 上界。 |
| global_fpr_within_target | cross_boundary | method_statistic | none | false | false | false | 经验 global FPR 与单侧上界是否同时不超过 target。 |

## 通用记录、产物与 baseline 字段（续）

| field_name | governance_level | category | required_suffix | allowed_in_records | allowed_in_claims | replacement_required | description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| failure_reason | persisted_protocol | protocol | none | true | false | false | 执行失败时必须保存的原因。 |
| exclusion_reason | persisted_protocol | protocol | none | true | false | false | 按预先规则排除时必须保存的原因。 |
| artifact_id | persisted_protocol | artifact | none | false | false | false | 受治理论文产物的稳定标识。 |
| artifact_type | persisted_protocol | artifact | none | false | false | false | 受治理论文产物类型，例如 table、figure、report 或 manifest。 |
| input_paths | persisted_protocol | artifact | none | false | false | false | 产物重建所需输入路径。 |
| output_paths | persisted_protocol | artifact | none | false | false | false | 产物重建生成输出路径。 |
| config_digest | persisted_protocol | provenance | none | false | false | false | 方法规格或产物重建配置的稳定内容摘要。 |
| code_version | persisted_protocol | artifact | none | false | false | false | 产物重建所用代码版本。 |
| rebuild_command | persisted_protocol | artifact | none | false | false | false | 产物重建命令。 |
| metadata | persisted_protocol | provenance | none | true | false | false | Record 或 manifest 的受治理扩展元数据；新增键仍需登记。 |
| baseline_name | persisted_protocol | baseline | none | false | false | false | 外部对比方法的稳定登记名称。 |
| source | persisted_protocol | baseline | none | false | false | false | Baseline 论文、官方仓库或正式发布来源。 |
| pinned_version | persisted_protocol | baseline | none | false | false | false | Baseline 的不可变版本、commit 或内容 digest。 |
| license | persisted_protocol | baseline | none | false | false | false | Baseline 代码或资产许可证。 |
| adapter_path | persisted_protocol | baseline | none | false | false | false | Baseline 实验适配器路径。 |
| config_path | persisted_protocol | baseline | none | false | false | false | Baseline 固定配置路径。 |
| deviations | persisted_protocol | baseline | none | false | false | false | 相对上游 baseline 的所有已声明语义偏差。 |
