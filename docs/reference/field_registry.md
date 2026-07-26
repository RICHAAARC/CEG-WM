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
| keyed_prg_version | cross_boundary | method_identity | none | false | false | false | 密钥流使用的 KDF/PRG 算法身份。 |
| root_key_public_digest | persisted_protocol | provenance | none | true | false | false | root key 的不可逆公开身份；原始 root key 和派生材料不得持久化。 |
| domain_digest | cross_boundary | provenance | none | false | false | false | 规范 key material、职责字段与 shape 共同形成的密钥流域摘要。 |
| values_float32_be_sha256 | cross_boundary | provenance | none | false | false | false | row-major CPU float32 流按 IEEE-754 big-endian 拼接后的输出摘要。 |
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
