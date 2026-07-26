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
| protocol_digest | cross_boundary | protocol | none | false | false | false | Preflight approval 携带的 comparison protocol 内容摘要。 |
| sample_manifest_digest | persisted_protocol | protocol | none | true | false | false | 当前运行所用样本 manifest 的内容摘要。 |
| split_manifest_digest | persisted_protocol | protocol | none | false | false | false | calibration 与 evaluation 切分 manifest 的内容摘要。 |
| generation_conditions_digest | persisted_protocol | protocol | none | false | false | false | 各方法共享生成条件的内容摘要。 |
| seed_policy_digest | persisted_protocol | protocol | none | false | false | false | 随机种子分配规则的内容摘要。 |
| output_specification_digest | persisted_protocol | protocol | none | false | false | false | 可比输出规格的内容摘要。 |
| attack_matrix_digest | persisted_protocol | protocol | none | false | false | false | 对比使用攻击矩阵的内容摘要。 |
| metric_set_digest | persisted_protocol | protocol | none | false | false | false | 对比使用指标集合的内容摘要。 |
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
| carrier_config_digest | cross_boundary | method_identity | none | false | false | false | carrier 算法、shape、mask 与 key schedule 配置身份摘要。 |
| delta_content | cross_boundary | method_state | none | false | false | false | content embedder 产生、尚未由 runtime 物化的理论内容更新。 |
| delta_content_digest | cross_boundary | provenance | none | false | false | false | 理论 `delta_content` 的 float32 字节摘要。 |
| latent_norm | cross_boundary | method_state | none | false | false | false | content embedder 计算共同总预算时消费的 callback latent 理论 L2 norm。 |
| target_total_norm | cross_boundary | method_state | none | false | false | false | runtime 物化前共同内容更新的目标总 L2 norm。 |
| target_relative_l2 | cross_boundary | method_state | none | false | false | false | runtime 物化前共同内容更新相对 latent 的目标 L2 比例。 |
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
| observation_protocol | cross_boundary | method_identity | none | false | false | false | 普通检测图像进入 LF/HF detector 的公共编码协议身份。 |
| observation_digest | cross_boundary | provenance | none | false | false | false | 普通检测图像侧 LF/HF 编码观测的 float32 字节摘要。 |
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
| routing_map | cross_boundary | method_state | none | false | false | false | `routing_stqr` 输出并按 latent channels 广播的空间权威图 `A`。 |
| mask_lf | cross_boundary | method_state | none | false | false | false | content router 输出、供 LF carrier 消费的空间 mask。 |
| mask_hf | cross_boundary | method_state | none | false | false | false | content router 输出、供 HF carrier 消费的空间 mask。 |
| routing_map_digest | cross_boundary | provenance | none | false | false | false | router `A` 的 row-major float32 字节摘要。 |
| mask_lf_digest | cross_boundary | provenance | none | false | false | false | router LF mask 的 row-major float32 字节摘要。 |
| mask_hf_digest | cross_boundary | provenance | none | false | false | false | router HF mask 的 row-major float32 字节摘要。 |
| observation_digests | cross_boundary | provenance | none | false | false | false | S/T/R/Q_sens 数值摘要与各自 runtime 来源身份的有序集合。 |
| semantic | cross_boundary | method_state | none | false | false | false | content router 消费的 runtime-provided 语义观测 `S`。 |
| texture | cross_boundary | method_state | none | false | false | false | content router 消费的 runtime-provided 纹理观测 `T`。 |
| response | cross_boundary | method_state | none | false | false | false | content router 消费的 callback latent 响应观测 `R`。 |
| sensitivity | cross_boundary | method_state | none | false | false | false | content router 消费的 public-probe 局部敏感性观测 `Q_sens`。 |
| mean_routing_map | cross_boundary | method_statistic | none | true | false | false | router `A` 在完整 latent CHW 广播后的均值。 |
| mean_mask_lf | cross_boundary | method_statistic | none | true | false | false | LF routing mask 在完整 latent CHW 广播后的均值。 |
| mean_mask_hf | cross_boundary | method_statistic | none | true | false | false | HF routing mask 在完整 latent CHW 广播后的均值。 |
| route_config_digest | cross_boundary | method_identity | none | false | false | false | 路由公式、插值、候选和 latent shape 的配置摘要。 |
| route_identity | cross_boundary | method_identity | none | true | false | false | 路由配置、observation 与 mask 摘要共同形成的路由身份。 |
| score | persisted_protocol | method_statistic | none | true | false | false | empirical CDF null record 保存的分支 float64 分数。 |
| source_cluster_id | persisted_protocol | provenance | none | true | false | false | calibration null record 所属 Prompt、seed、lineage 与 key family 聚类身份。 |
| sample_id | persisted_protocol | provenance | none | true | false | false | calibration null record 在 source cluster 内的稳定样本身份。 |
| branch | cross_boundary | method_identity | none | false | false | false | empirical CDF 与标准化结果所属的 `hf` 或 `lf` 分支。 |
| partition_identity | persisted_protocol | provenance | none | true | false | false | empirical CDF 所属互斥 calibration 职责 partition 身份。 |
| records | cross_boundary | method_state | none | false | false | false | 分支 empirical CDF 消费的稳定排序 primary-null record 集合。 |
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
| declared_deviation | persisted_protocol | protocol | none | false | false | false | baseline 相对上游实现的已声明语义偏差。 |
| methods | persisted_protocol | protocol | none | false | false | false | Comparison protocol 中参与方法规格的有序集合。 |
| method_code_revision | persisted_protocol | protocol | none | true | false | false | 当前 record 实际执行的方法代码 revision。 |
| model_revision | persisted_protocol | protocol | none | true | false | false | 当前 record 实际执行的生成模型 revision。 |
| seed | persisted_protocol | protocol | none | true | false | false | 当前 record 实际使用的随机种子。 |
| metric_name | persisted_protocol | protocol | none | true | false | false | 实验记录中的指标名称。 |
| metric_value | persisted_protocol | protocol | none | true | false | false | 实验记录中的指标数值。 |
| execution_status | persisted_protocol | protocol | none | true | false | false | 当前尝试成功、失败或被排除的显式状态。 |
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
