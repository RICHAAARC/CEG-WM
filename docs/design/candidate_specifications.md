# CEG-WM Candidate Specifications

历史兼容组合候选的输出标签只使用 `hf_only_standardized_score`、
`weighted_hf_lf_standardized_score`、`maximum_hf_lf_standardized_score`；
`C_0`、`C_1(w)`、`C_2` 仅保留为本地数学记号。语义—纹理软路由方法的组合身份唯一为
`content_combination_semantic_texture_max_standardized`，不使用上述历史输出标签替代
新 detector identity。

## Authority And Scope

本文关闭“实现时自行发明算法”的空白，登记可实施、可证伪的候选规格。

registry 固定为 **20 个候选 ID**：其中 19 个是具名的
method/runtime/key 候选，`routing_uniform_control` 是强制保留、不得晋升为方法的
同预算禁用对照。计数如下：

- `key_schedule_sha256_counter`；
- `runtime_sd35_flowmatch`；
- `hf_sparse_tail`；
- `lf_low_pass`；
- `lf_null_whitened_matched_score`；
- `routing_stqr`，并以 `routing_uniform_control` 为同预算禁用对照；
- `content_combination_calibrated`；
- `routing_inspyrenet_salient_local_lf`；
- `content_embedding_global_hf_local_lf`；
- `lf_saliency_masked_null_whitened_matched_score`；
- `content_combination_saliency_max_standardized`；
- `routing_semantic_texture_soft`；
- `content_embedding_semantic_texture_soft_lf_hf`；
- `lf_semantic_texture_soft_whitened_matched_score`；
- `hf_semantic_texture_soft_direct_score`；
- `content_combination_semantic_texture_max_standardized`；
- `qk_relation_similarity`；
- `rectification_similarity`；
- `joint_conditional_recovery`。

实现者只能实现这些身份及其明确列出的有限候选值。增加 relation、objective、write、score、observation、backbone、runtime 或搜索策略，必须先修订本文并重新接受候选规格审计。

候选 registry 与实现职责是两个不同计数：上面的 20 个 ID 描述算法/runtime/key
候选身份；未来实现固定为 13 项职责组件。每项职责只消费下表列出的现有候选，
不得据此新增候选、别名组件或把多个职责集中到一个代理模块：

| component | responsibility | method path | candidate binding |
| --- | --- | --- | --- |
| `key_schedule` | `root_key_derivation_and_prg` | `main/shared/key_schedule.py` | `key_schedule_sha256_counter` |
| `content_router` | `content_observation_and_adaptive_routing` | `main/content_chain/routing.py` | `key_schedule_sha256_counter`, `routing_stqr`, `routing_uniform_control`, `routing_inspyrenet_salient_local_lf`, `routing_semantic_texture_soft` |
| `lf_carrier` | `low_frequency_carrier_template_and_write_direction` | `main/content_chain/lf_carrier.py` | `key_schedule_sha256_counter`, `lf_low_pass` |
| `hf_carrier` | `high_frequency_carrier_template_and_write_direction` | `main/content_chain/hf_carrier.py` | `key_schedule_sha256_counter`, `runtime_sd35_flowmatch`, `hf_sparse_tail` |
| `content_embedder` | `lf_hf_combined_embedding_and_total_budget` | `main/content_chain/embedder.py` | `runtime_sd35_flowmatch`, `hf_sparse_tail`, `lf_low_pass`, `routing_stqr`, `routing_uniform_control`, `routing_inspyrenet_salient_local_lf`, `content_embedding_global_hf_local_lf`, `routing_semantic_texture_soft`, `content_embedding_semantic_texture_soft_lf_hf` |
| `lf_detector` | `low_frequency_blind_scoring` | `main/content_chain/lf_detector.py` | `key_schedule_sha256_counter`, `lf_low_pass`, `lf_null_whitened_matched_score`, `routing_inspyrenet_salient_local_lf`, `lf_saliency_masked_null_whitened_matched_score`, `routing_semantic_texture_soft`, `lf_semantic_texture_soft_whitened_matched_score` |
| `hf_detector` | `high_frequency_direct_scoring` | `main/content_chain/hf_detector.py` | `key_schedule_sha256_counter`, `hf_sparse_tail`, `routing_semantic_texture_soft`, `hf_semantic_texture_soft_direct_score` |
| `content_detector` | `lf_hf_score_standardization_and_content_detection` | `main/content_chain/detector.py` | `hf_sparse_tail`, `lf_low_pass`, `content_combination_calibrated`, `content_combination_saliency_max_standardized`, `content_combination_semantic_texture_max_standardized` |
| `qk_geometry_sync` | `keyed_qk_geometry_synchronization_and_relation_observation` | `main/geometry_chain/qk_sync.py` | `key_schedule_sha256_counter`, `runtime_sd35_flowmatch`, `qk_relation_similarity` |
| `geometric_transform_estimator` | `blind_bounded_geometric_transform_estimation` | `main/geometry_chain/transform_estimator.py` | `key_schedule_sha256_counter`, `qk_relation_similarity`, `rectification_similarity` |
| `geometry_reliability` | `independent_geometry_reliability_conjunction` | `main/geometry_chain/reliability.py` | `key_schedule_sha256_counter`, `qk_relation_similarity`, `rectification_similarity` |
| `image_rectifier` | `image_coordinate_rectification` | `main/geometry_chain/rectifier.py` | `rectification_similarity` |
| `conditional_recovery_decision` | `conditional_same_detector_recovery` | `main/joint_decision/detector.py` | `joint_conditional_recovery` |

`content_embedder` 独占 combined direction、nominal/actual hard limit、组合 delta 的
materialization reconciliation、realized norm/relative L2 和 active 零方向失败。
兼容 `a/u_content(a)` 身份拥有 mixing coefficients 与 direction dot/c；
语义—纹理路线只拥有
`normalize(normalize(m_hf*T_hf)+normalize(m_lf*T_lf))`，不存在 `a/w` grid。
`lf_detector`
独占盲 `s_lf`；`geometry_reliability` 独占 estimator
原始指标上的可靠性合取门。这三项不能由 carrier、content detector 或 transform
estimator 代行。候选绑定表示该组件必须实现或消费的规格身份。

候选身份与 13 项职责分离；增加候选不增加组件职责，也不允许一个实现 symbol 代行
多个职责。

## Candidate `key_schedule_sha256_counter`

### Root-Key Contract And Stable Serialization

CEG-WM 的公开 key API 只接受非空 Unicode `str`。该字符串按原始 Unicode
code points 直接编码为 UTF-8；不执行 NFC/NFKC 或大小写归一化，因此规范等价但
字节不同的字符串是不同密钥。`bytes`、空字符串、含 surrogate 的字符串以及无法
编码为严格 UTF-8 的输入 fail closed。实现内部称其为 `root_key_text`，不得写入
records、日志、图像元数据或 artifacts。

规范序列化 `stable_json_utf8` 唯一定义为：

```text
json.dumps(value,
           ensure_ascii=false,
           sort_keys=true,
           separators=(",", ":"),
           allow_nan=false).encode("utf-8", errors="strict")
```

允许的 domain value 仅为 JSON `null`、bool、十进制整数、有限字符串、上述值的
array，以及 key 为字符串的 map；禁止 float、set、tuple 的隐式类型身份、tensor、
path 和自定义对象。shape 先转为非空正整数 array。所有 domain map 必须具有本文
列出的精确字段集；缺字段和额外字段都失败。

公开不可逆身份为：

```text
root_key_public_digest =
  hex(SHA256(stable_json_utf8({
    "candidate_id": "key_schedule_sha256_counter",
    "record_role": "root_key_public_digest",
    "root_key_text": root_key_text
  })))
```

只有该 64 位小写 hex digest 可持久化。它是高熵密钥的身份指纹，不是密钥材料，
也不构成低熵口令安全承诺。

### KDF, Counter Stream And Distributions

候选的 PRG 算法身份字符串固定为
`sha256_counter_normal_icdf_table20_float32`。对输出 shape、根材料和职责字段：

```text
domain_digest = SHA256(stable_json_utf8({
  "keyed_prg_version": "sha256_counter_normal_icdf_table20_float32",
  "key_material": key_material,
  "domain_fields": domain_fields,
  "shape": shape
}))
block_c = SHA256(domain_digest || uint128_be(c)), c = 0,1,2,...
```

计数器从 0 开始，恰为 16-byte unsigned big-endian；溢出 fail closed。uniform
输出按每个 SHA-256 block 的 byte offsets `0,8,16,24` 读取四个 uint64 big-endian
word，取高 53 bits 为 `m`，映射为 `(m+1)/(2^53+2)`，再按 IEEE-754 binary32
round-to-nearest-ties-to-even 物化到 CPU float32。

Gaussian 输出把连续 `block_c` 拼成 MSB-first bitstream，跨 block 连续切成 20-bit
unsigned index。index `i` 查表：

```text
q_i = round_binary32(Phi^-1((i+0.5)/1048576))
```

表共 `2^20` 项，IEEE-754 big-endian 原始字节 SHA-256 固定为
`70abf440a7f3670147965ffa52f5aaa639dab97f6282b68f3a9a1b1ce5e6cf5a`；
运行时不得调用平台 `NormalDist`、`erfinv` 或其他动态 inverse-CDF 代替冻结表。
负半表由正半表逆序设置 float32 sign bit，最终输出在 CPU 以 row-major/C-order
reshape 为 float32 tensor，之后才允许搬到设备。

### Responsibility Domains

秘密职责都使用原始 `root_key_text` 作为 `key_material`，并要求以下精确字段：

- HF：`candidate_id=hf_sparse_tail`、`operator=carrier_template`、
  `responsibility_domain=hf_carrier`、`model_revision`、`tensor_role=base_gaussian`；
- LF：`candidate_id=lf_low_pass`、`operator=carrier_template`、
  `responsibility_domain=lf_carrier`、`model_revision`、`tensor_role=base_gaussian`；
- Q/K projection：`candidate_id=qk_relation_similarity`、
  `operator=attention_relation_signs`、`responsibility_domain=geometry_sync`、
  `model_revision`、`layer_name`、`token_count`、`tensor_role=pair_uniform`。

错误 key roster 在访问图像或分数前由 manifest 冻结。第 `j` 个错误根材料为：

```text
wrong_key_material_j =
  "ceg-wm-wrong-key:" + hex(SHA256(stable_json_utf8({
    "candidate_id": "key_schedule_sha256_counter",
    "derivation_role": "geometry_and_content_wrong_key",
    "registered_root_key_public_digest": root_key_public_digest,
    "wrong_key_index": j
  })))
```

`j` 必须是从 0 起、无重复的预登记非负整数；roster 大小由 attribution protocol
预登记，不继承历史 32-key 数字。错误材料随后走与注册 key 相同的职责 domain。

公开噪声不消费 secret root。其 `key_material` 固定为 ASCII literal
`ceg-wm-public-noise:key-schedule-sha256-counter`，并使用：

- image-only Q/K：`candidate_id=qk_relation_similarity`、
  `operator=public_image_only_qk_detection_noise`、
  `responsibility_domain=public_noise`、`model_revision`、`schedule_index=7`、
  `conditioning_protocol=sd3_empty_text_triplet_without_cfg`、
  `tensor_role=scheduler_noise`；该 noise 只由公共身份与 tensor shape 决定，对所有
  图像相同，因此盲检不需要 sample manifest 或 embed record；
- routing sensitivity probe：`candidate_id=routing_stqr`、
  `operator=local_sensitivity_public_probe`、
  `responsibility_domain=public_noise`、`model_revision`、`sample_index`、
  `tensor_role=latent_probe`。

routing 的 `sample_index` 来自预登记 generation manifest，不得以运行顺序、分数或
重试次数替代；它不进入 image-only Q/K 检测 noise。

### Golden Vector, Failures And Gates

规范 Gaussian golden vector：

```text
shape = [2,3]
root_key_text = "ceg-wm-golden-root-π"
domain_fields = {
  "candidate_id": "key_schedule_sha256_counter",
  "operator": "golden_vector",
  "responsibility_domain": "key_schedule_test",
  "tensor_role": "gaussian"
}
domain_digest =
  e5b8e35d13815c1d23a09286da0bfe661e0330e38eda19e239f19224f7b1998f
indices = [172059,964892,707530,322430,968250,915318]
float32_be_bits =
  [bf7a508b,3fb40402,3ee7f9d3,bf00c274,3fb6d22b,3f91f4c9]
concatenated_float32_be_sha256 =
  c82e2f254ab05f4502d397aa444d8facefaa64e0c4df4f1617e12948acecb8d0
uniform_float32_be_bits_from_same_domain =
  [3e2806fb,3f6b7eec,3f1ca35d,3e6af2b7,3f213aef,3ef25444]
root_key_public_digest =
  51ad81701f05213fbd7ee5cecc0987ffca7d8be76cff58394dc0da4fe8e1423d
wrong_key_material_0 =
  ceg-wm-wrong-key:843d3aa0d4d81ed3b17c7d0bd970145ef912ed3188db3079237214da185c985f
```

CPU gate 必须逐项复验 stable JSON bytes、domain digest、跨 block bit ordering、表
摘要、golden indices/bits、uniform open interval、HF/LF/geometry domain
separation、wrong-key determinism 和 public-noise secret independence。真实 runtime
gate 必须证明 CPU 生成后设备搬运不改变 tensor bytes，并验证 registered/wrong/public
三类调用只使用允许的材料。任何不支持的 root 类型、非法 domain、counter overflow、
表摘要漂移、非 IEEE-754 binary32、golden mismatch、职责碰撞或 secret 落盘均 fail
closed。该候选只继承历史文件的 provisional algorithm source，不继承历史方法身份或
成功证据。

## Candidate `runtime_sd35_flowmatch`

### Inputs And Outputs

输入是 Prompt、negative Prompt、生成 seed、普通检测图像和冻结 runtime 配置。输出
是 clean/watermarked 普通 RGB 图像、用于方法 API 的当前 latent、最终图像 VAE mode
latent，以及指定层的真实 Q/K observation。执行内容 delta 时，runtime 还返回实际
dtype 的物化张量、`delta_content_actual`，以及该 combined delta 的 realized
combined total norm/relative L2。runtime 只负责物化、测量和返回；target 与
realized 是否合格仍由 `content_embedder` 判定，runtime 不拥有 mixing 或 budget，
也不输出阳性判定。

### Frozen Actual-Dtype Content Budget

当前候选把名义写入强度与 actual-dtype 总内容扰动硬上限统一冻结为：

```text
content_relative_l2_nominal = 3/250 = 0.012
content_relative_l2_limit   = 3/250 = 0.012
```

这不承诺每个样本的 actual realized ratio 接近名义值。令 `z0` 为 callback 18
写入前已经按登记 actual dtype（当前为 binary16）物化的 baseline，
`delta_content_nominal` 为 `content_embedder` 按 LF/HF/routing 最终合成方向产生的
binary32 名义 delta。对一个正 binary32 scale `s<=1`，唯一物化对象为：

```text
d_s[i]            = f32(f32(delta_content_nominal[i]) * f32(s))
precast[i]        = f32(f32(z0[i]) + d_s[i])
z_s[i]            = binary16_RNE(precast[i])
delta_actual_s[i] = f32(z_s[i]) - f32(z0[i])
```

binary16 转换使用 round-to-nearest-ties-to-even；有限 subnormal 按 binary16
语义保留或舍入到零，overflow、NaN/Inf 和非法 baseline 全部 fail closed。runtime
必须对 `z_s` 做独立逐 bit replay，并把实际张量、实际 delta、身份与测量返回给
`content_embedder`。写入完整性要求 finite、bitwise replay 相同和最终 actual delta
非零；中间搜索点舍入为零时只构成 zero plateau，不是可接受写入。

所有向量按 row-major 顺序展平。binary32 L2 唯一定义为从 `S_0=f32(0)` 开始，
逐项执行 `q_i=f32(x_i*x_i)`、`S_{i+1}=f32(S_i+q_i)`，最后
`norm32(x)=f32(sqrt(S_n))`。硬预算右侧为：

```text
L = f32(norm32(fp32(z0)) * f32(3/250))
A = norm32(delta_actual_s)
accept iff A <= L
```

权威 gate 直接比较 `A` 与 `L`；`realized_relative_l2` 和
`budget_utilization=A/L` 仅为诊断。不得使用比值边界、`q_budget`、
`tau_actual_budget`、经验 tolerance 或 actual 强度下限。

`content_embedder` 先请求 `s=1`；若完整且 `A<=L`，立即接受。若超限，则以
binary32 `[0,1]` 为区间，冻结 midpoint 为
`m=f32(f32(f32(lower)+f32(upper))*f32(0.5))`。非零合格点更新 lower 并成为当前
最大可行 observation；超限点更新 upper；actual delta 为零的 plateau 点只推进
lower，不得成为可行 observation。当 midpoint 与任一边界 bitwise 相同、区间内
没有新的 representable midpoint 时停止，返回最大的非零合格 scale；若从未出现
非零合格 observation，则 fail closed。不得用 `1,1/2,1/4` 粗回退、GPU 拟合阈值
或结果后 tolerance 代替此协议。

接受、重试、scale 选择和最终失败全部属于现有 `content_embedder`。runtime 只按
请求物化、测量和执行完整性检查，不拥有预算语义。该 hard limit 只约束
LF/HF/routing 最终合成的 combined content delta；不得相加分支 norm，不存在
actual branch decomposition。`content_direction`、`active_lf_direction`、
`active_hf_direction` 和 target component 只作为 nominal formula witnesses。
geometry delta 与已有 geometry/total budget 保持独立，不并入
`content_relative_l2_limit`。低 utilization 不得成为未来实验的事后筛选条件。

### Frozen Candidate Algorithm

首个且唯一登记的 backbone/runtime 候选为：

- model：`stabilityai/stable-diffusion-3.5-medium`；
- model revision：`b940f670f0eda2d07fbb75229e779da1ad11eb80`；
- pipeline：`StableDiffusion3Pipeline`；
- scheduler：`FlowMatchEulerDiscreteScheduler`；
- `512 x 512`、20 inference steps、guidance `4.5`；
- latent dtype `float16`，模板和分数计算使用 `float32`；
- generation seed 在 CPU generator 上生成基础 latent；
- 图像解码使用 VAE scaling factor 和 shift factor；
- 检测编码使用 VAE posterior mode，不采样；
- `content_embedder` 产生的内容 delta 由 runtime 在 callback index 18 物化，写入后
  保留一个 scheduler interval；HF-only 也走同一 embedder/runtime 边界；
- Q/K observation 使用同一模型 revision：待检图像经 VAE posterior mode 编码，在重新建立的 20-step scheduler 上以 index 7 的 timestep 加入公开确定性噪声，再以三路空字符串条件、无 classifier-free guidance 做一次 Transformer 前向；不读取生成缓存。

候选依赖锁来自 FlowHF：Python `>=3.12`、diffusers `0.38.0`、torch `2.11.0`、transformers `5.12.1`、accelerate `1.14.0`、numpy `2.0.2`、Pillow `11.3.0`、safetensors `0.8.0`、huggingface-hub `1.20.1`。若该组合不能在获授权环境解析，不允许实现者自行升级；应登记 runtime candidate failure 并先修订候选规格。

### Configuration Identity

身份必须包含上述 model/revision、pipeline/scheduler 类、steps、guidance、尺寸、latent/VAE dtype、VAE scaling/shift、callback index、检测 schedule/conditioning、Q/K 层名、依赖锁和张量预处理。

### Failure Semantics

模型或 revision 不可解析、scheduler 类漂移、随机状态不确定、callback 未触发或重复触发、VAE mode 编码失败、Q/K 层不存在、Q/K 非真实投影输出、非有限张量或实际 dtype 写入消失均 fail closed。

### Checks And Gate

- CPU：配置摘要稳定、FlowHF golden vector 可读取、VAE scaling/shift 公式和 callback 状态机的纯函数检查。
- 真实 runtime：模型/revision/类身份、同 seed clean 配对、callback index、实际 dtype、final-image VAE mode latent、指定层真实 Q/K 内容和重复运行确定性。
- 晋升：全部身份相符且 CEG-WM HF candidate holdout 复现通过。
- 淘汰：任何身份无法复现；不得静默换 backbone、revision、scheduler、steps 或写入位置。

## Candidate `hf_sparse_tail`

### Inputs And Outputs

这是一个跨职责候选，不表示由单一模块包办载体、写入和检测：

- `hf_carrier` 消费 `[1,C,H,W]` latent 形状、HF 派生密钥、model identity 和
  `mask_hf`，只输出单位 L2 稀疏模板 `T_hf`、masked unit direction `u_hf`、支持集
  和必要身份元数据/摘要；HF-only 对照的 `mask_hf` 恒为全 1；
- `content_embedder` 消费 `u_hf`。HF-only 时它仍独占 nominal/limit、
  `delta_content_nominal`、actual-dtype hard-budget reconciliation、realized
  combined total relative L2 和零方向失败；组合模式下它还消费 LF direction 与
  router 输出；
- `hf_detector` 只消费普通待检图像、检测 key 和公共 identity/资产，独立重构
  `T_hf` 并输出盲分数 `s_hf`；
- `content_detector` 消费独立标准化的 soft-routed `s_hf` 与 `s_lf` 并形成固定 max
  statistic；它不读取 carrier direction、callback latent 或写入记录；
- runtime 只在冻结 callback/model/dtype 边界物化 `content_embedder` 给出的更新，
  并把实际 dtype latent 与 combined delta 的 realized total norm/relative L2 测量
  返回给 embedder 判定；不拥有模板、组合、预算判定或评分算法。

### HF Carrier Template And Direction

```text
G_hf = key_schedule_sha256_counter.gaussian(
  shape, root_key_text, hf_domain_fields)
P_hf = avg_pool2d(G_hf, kernel=5, stride=1, padding=2,
                  count_include_pad=true)
H_hf = G_hf - P_hf
m = ceil(0.20 * number_of_elements)
J = first m indices under (-abs(H_hf[i]), flat_index[i])
S_hf[i] = H_hf[i] if i in J else 0
T_hf = S_hf / ||S_hf||_2
u_hf = normalize(mask_hf * T_hf)
```

模板构造不得中心化；tail 外必须保持精确零。tail `0.20`、5x5 zero-padded
average、平局顺序、callback 18 与 `3/250` 构成该候选语义。

### Content Embedder And Runtime Boundary

HF-only 对照的全 1 mask 使 `u_hf=T_hf`；`content_embedder` 按共同内容总预算产生：

```text
delta_content_nominal = (3/250) * norm32(fp32(z0)) * u_hf
z_s = cast_actual(fp32(z0) + s * delta_content_nominal)
```

`z_s` 由 runtime 在 callback 18 按上述 actual-dtype 协议物化；正式 runtime
对象中的对应字段为 `written_latent_actual`。runtime 把该 actual-dtype latent、
replay identity 以及 `delta_content_actual` 的 realized total
norm/relative L2 返回；hard limit 比较、必要的 binary32 bisection 和最终
accepted/fail-closed 属于 `content_embedder`。runtime 不得自行改变强度、方向、
重新分配预算或输出自己的 budget decision。

### HF Detector Boundary

最终图像经同一 VAE mode 重编码为 `Y`。评分为：

```text
s_hf = dot(center(float32(Y)) / ||center(float32(Y))||,
           center(T_hf) / ||center(T_hf)||)
```

检测模板只从 key、shape 和公共 identity 重构。

### Configuration And Historical Source Boundary

跨组件候选身份由各责任共同组成，但所有权不得混合：`hf_carrier` 拥有 PRG
版本/domain、normal-quantile protocol、shape、low-pass 参数、tail fraction/order
和模板归一顺序及 mask identity；`content_embedder` 拥有 nominal/limit、
hard-budget direct comparison、retry/scale/final failure 和 combined realized
记录；runtime candidate 拥有
callback/model/dtype 物化边界；
`hf_detector` 拥有 VAE encode 与 score operator；model
identity 和必要公共预处理身份由三者一致引用。该 CEG-WM 候选的算法顺序来自
historical DirectHF，来源边界是 FlowHF 的逐文件 SHA；历史名称、inversion、oracle
callback latent、32 wrong-key roster 大小和四 Prompt 结果不进入当前 detector 身份。

### Failure, Checks And Gate

- 失败归属：模板零/非有限、support 错误、tail 外非零或 template-time centering 由
  `hf_carrier` fail closed；零方向、完整性失败、hard limit 超限且不存在非零
  可行 scale 由 `content_embedder` fail closed；VAE/shape 漂移或评分零中心化能量由
  `hf_detector` fail closed。
- CPU：分别检查 carrier 的 golden template/support/平局/单位 L2/tail 外零，
  embedder 的 nominal formula、hard-budget 算术、单调/终止/最大非零可行搜索；
  真实 SD3.5 actual dtype 的 realized combined total norm/relative L2 留到真实
  runtime gate，以及 detector 的
  score-time-only centering 和 key-only 重构。
- 真实 runtime：同 seed clean/watermarked 配对，final-image VAE mode 评分，registered/wrong-key 分离，写入后一个 interval，质量与失败留分母。
- 晋升：参数级 golden 与真实 holdout 都通过。
- 淘汰：任一参数需改变时登记新候选，不能继承 FlowHF 证据。

## Candidate `lf_low_pass`

### Inputs And Outputs

这是一个跨 `lf_carrier`、`content_embedder`、`lf_detector` 的候选，不表示候选由
单一模块实施：

- `lf_carrier` 消费 latent 形状、独立 LF 派生密钥、model identity 和
  `mask_lf`，只输出 `T_lf`、masked unit direction `u_lf` 及必要载体身份元数据；
- `content_embedder` 消费 `u_lf`、由 `hf_carrier` 提供的 `u_hf` 和 router 输出，
  独占 `a`、`u_content(a)`、nominal/limit、`delta_content_nominal`、hard-budget
  accept/retry/scale/final failure、realized combined total norm/relative L2 和
  任一 active 零方向失败；
- `lf_detector` 只消费普通待检图像、检测 key 和公共 identity/资产，在不读取
  routing mask、callback latent、embed record 或参考图的条件下独立重构未遮罩
  `T_lf`，并输出盲分数 `s_lf`；
- `content_detector` 只有在组合候选晋升后才消费 `s_lf`；它不读取 `u_lf`、
  routing mask、callback latent 或实际写入记录；
- runtime 只在冻结 callback/model 边界物化 `content_embedder` 的
  `delta_content`，不拥有 LF/HF 组合算法。

### LF Carrier Template And Direction

```text
G_lf = key_schedule_sha256_counter.gaussian(
  shape, root_key_text, lf_domain_fields)
L_lf = avg_pool2d(G_lf, kernel=5, stride=1, padding=2,
                  count_include_pad=true)
T_lf = center_per_sample(L_lf) / ||center_per_sample(L_lf)||_2
u_lf = normalize(mask_lf * T_lf)
```

`lf_carrier` 不产生实际写入，也不计算 `s_lf`。`mask_lf` 只影响嵌入方向；盲检
模板 `T_lf` 不依赖生成时 routing 观测。

### Content Embedder And Runtime Boundary

`content_embedder` 从两个 carrier 接收已遮罩单位方向，其中
`u_hf=normalize(mask_hf*T_hf)`，然后唯一执行：

```text
gamma_lh = dot(u_lf, u_hf)
v_content(a) = a * u_lf + (1-a) * u_hf
c(a) = ||v_content(a)||_2
     = sqrt(a^2 + (1-a)^2 + 2*a*(1-a)*gamma_lh)
u_content(a) = v_content(a) / c(a)
delta_content_nominal =
  (3/250) * norm32(fp32(z0)) * u_content(a)
```

双载体候选的有限 mixing-coefficient 集合是
`a in {0.25, 0.50, 0.75}`。`a` 与 `1-a` 是方向混合系数，不是可加的方向份额。
LF-only、HF-only、route-disabled 和 routed 对照都使用相同
nominal relative L2 `rho_content_nominal=3/250`，且实际 combined content delta
共同受 `content_relative_l2_limit=3/250` 硬上限约束；若任一 active masked direction
为零，或 `c(a)=0`/非有限，则该样本显式失败，不重新分配能量。

runtime 只按 `content_embedder` 请求的 binary32 scale 把名义 `delta_content`
加到 callback 18 actual baseline 并以实际 dtype 物化。其权威测量对象仅为：

```text
delta_content_actual =
  float32(z_s) - float32(z0)
realized_total_l2 = ||delta_content_actual||_2
realized_relative_l2 = realized_total_l2 /
                       ||float32(z0)||_2
```

其中 `z_s` 在正式 runtime 结果对象中记录为 `written_latent_actual`，`z0` 记录为
`baseline_latent_actual`。

`content_embedder` 按本候选的冻结 binary32 直接硬比较、必要二分和最大非零可行
选择判定 accepted 或 fail closed；不要求 realized 值接近 nominal，也不设置
actual 下限。
若实验需要报告 LF/HF contribution，只能记录可重建的 `a`、`gamma_lh`、`c(a)`，
以及 target component vectors
`delta_lf_target=(3/250)*norm32(fp32(z0))*(a/c)*u_lf`、
`delta_hf_target=(3/250)*norm32(fp32(z0))*((1-a)/c)*u_hf` 及各自 norm。两向量
之和才是 nominal `delta_content`；由于交叉项存在，它们的 norm/energy 不得相加冒充 total。
未定义也不可观测的分支级实际写入量不得出现在 runtime 或 evidence 中。

### Raw LF Detector Boundary

原始 LF 分数使用与 HF 相同 final-image VAE mode latent 和 score-time centered normalized correlation：

```text
s_lf_raw = normalized_correlation(center(Y), center(T_lf))
```

该 raw score 仍绑定 `lf_low_pass`，只作为独立比较 control；它不得与 whitened score
组成结果后 ensemble，也不得用于拟合 soft-routed LF candidate。

### Identity And Source Boundary

跨组件候选身份由各责任共同组成，但所有权唯一：`lf_carrier` 拥有独立 PRG domain、
filter、模板中心化顺序与 mask identity；`content_embedder` 拥有 `a`、方向组合顺序、
`gamma_lh`/`c(a)`、nominal/limit、hard-budget scale 选择与 combined realized
norm/relative L2；`lf_detector` 拥有 final-image VAE observation 与 score
operator。历史固定
`0.70/0.30` 检测权重与 `0.0025` LF 写入强度被排除；
它们不是本候选默认值。

### Failure, Checks And Gate

- 失败归属：模板非有限/零能量或 domain 漂移由 `lf_carrier` fail closed；active
  masked direction 为零、`c(a)` 为零/非有限、完整性失败或 hard limit 超限且无
  非零可行 scale 由 `content_embedder`
  fail closed；依赖私有嵌入状态、模板重构失败或分数非有限由 `lf_detector`
  fail closed。
- CPU：分别检查 carrier 的低通响应、单位 L2、LF/HF key-domain 分离与复现，
  embedder 的 active 零方向/组合零方向失败、全部对照共同 nominal/limit 和冻结
  binary32 搜索性质，以及 detector 的
  key-only 模板重构、正确 key/错误 key 分离和盲评分输入边界。
- 真实 runtime：`content_embedder`/runtime 检查 realized combined total
  norm/relative L2；`lf_detector` 独立检查
  LF-only attribution、unwatermarked null；候选层再检查 HF 易损攻击互补性、质量和
  HF-only 非退化。
- 晋升：`lf_detector` 的 LF-only attribution/FPR 先过门，再由 candidate-selection
  confirmation 选择供 `content_embedder` 使用的 `a`。
- 淘汰：LF 无独立密钥归属、只检测通用低频偏移、detector 依赖参考图/record，或
  所有 `a` 均无稳定互补增益。

## Candidate `lf_null_whitened_matched_score`

### Scientific Role And Fixed Observation

该候选只改变 `lf_detector` 的盲评分统计，不改变 `lf_low_pass` 的 carrier、模板、
写入方向、`3/250` 内容 hard budget、runtime 或 key schedule。它要检验的唯一问题是：
在公共 clean-null LF 观测的低维二阶结构下白化后，registered `T_lf` 是否比 wrong-key
模板获得稳定更高的 matched score。

检测观测固定为普通 `512 x 512` RGB8 待检图像，经
`runtime_sd35_flowmatch` 同一 public VAE encode 和 posterior `mode()` 得到
binary32 latent `Y`，shape 必须精确为 `[1,16,64,64]`。检测只允许消费该 RGB8、
检测 key、冻结 model/preprocess identity 和下面的公开 whitening artifact；不得读取
参考图、fit images、embed record、callback latent、写入模板记录、routing mask 或
其他私有生成状态。

### Centering, Detrending And Frequency Coordinates

对任一 latent 或由检测 key 重构的未遮罩 `T_lf`，先去掉 batch 维。对 channel `c`
和零基坐标 `h,w in {0,...,63}`，定义：

```text
x_h = (2*h - 63) / 63
y_w = (2*w - 63) / 63
a_c = sum_{h,w} Z[c,h,w] / (64*64)
b_c = sum_{h,w} x_h*Z[c,h,w] / (64*sum_h x_h^2)
g_c = sum_{h,w} y_w*Z[c,h,w] / (64*sum_w y_w^2)
D(Z)[c,h,w] = Z[c,h,w] - a_c - b_c*x_h - g_c*y_w
```

上述对称坐标使常数、`x`、`y` 三列两两正交，因此给出唯一 affine-plane
least-squares 解，不存在 detrending 候选或拟合网格。输入先转换为 binary32，
上述求和、乘除和后续 fit/DCT/score 运算均用 binary64。每个 affine sum 固定按
`h=0..63` 外层、`w=0..63` 内层累加，先完成 `a_c`，再分别完成 `b_c`、`g_c`，
最后按同一顺序形成 residual；不得先应用 `W`、按 key 改变 detrending，或对单个
待检样本重新拟合 null 统计。

令 `F(Z)` 为对 `D(Z)` 每个 channel 施加的 orthonormal 2D DCT-II：

```text
alpha_64(0) = sqrt(1/64)
alpha_64(k) = sqrt(2/64), k>0
F(Z)[c,u,v] = alpha_64(u)*alpha_64(v) *
  sum_{h,w} D(Z)[c,h,w]
             * cos(pi*(h+1/2)*u/64)
             * cos(pi*(w+1/2)*v/64)
```

第一轴 `h/u` 是 latent height，第二轴 `w/v` 是 latent width；每个 coefficient 的
内和固定按 `h` 外层、`w` 内层 binary64 累加。`pi` 固定为 binary64
`0x1.921fb54442d18p+1`，DCT 不允许换成 FFT 周期边界或非正交 normalization。

DC coefficient `(u,v)=(0,0)` 永久排除。其余频率按
`r=max(u,v)` 和 `band=floor(log2(r))` 唯一分成 6 个 dyadic Chebyshev rings：
`{1}`、`{2,3}`、`{4,...,7}`、`{8,...,15}`、`{16,...,31}`、
`{32,...,63}`。没有其他 band count、band boundary 或 transform 候选。

### Single Clean-Null Fit And Whitening Operator

`W` 只允许从一个预登记、与旧 LF 8-cluster diagnostic、全部 development
screening、candidate selection、calibration 和 evaluation source clusters 零交集的
fit manifest 产生。manifest 必须恰含 32 个独立 source clusters；每个 cluster 只贡献
一个 clean public RGB8，经同一 public RGB-to-VAE 路径得到 `Y_i`。旧 8 个样本、
candidate/wrong-key 图像、攻击图像和任何 observed score 均禁止进入 fit。

对 channel `c`、band `q`，令 `B_q` 为该 ring 的 `(u,v)` 集合、`n_q=|B_q|`：

```text
v[c,q] = sum_{i=0}^{31} sum_{(u,v) in B_q} F(Y_i)[c,u,v]^2
         / (32*n_q)
v_global = sum_{c,q} n_q*v[c,q] / (16*sum_q n_q)
lambda = 2^-10
W[c,q] = binary32_rne((v[c,q] + lambda*v_global)^(-1/2))
```

这是唯一允许的 `W`：channel-by-dyadic-band stationary diagonal whitening，固定
`16*6=96` 个 binary32 权重。`v[c,q]` 的累加顺序固定为 fit manifest 中的 cluster
顺序，然后 `u` 外层、`v` 内层；`v_global` 固定按 `c=0..15` 外层、`q=0..5`
内层并按真实 coefficient count 加权。正则语义唯一是给每个 `v[c,q]` 加上同一个
strictly-positive ridge energy `2^-10*v_global`，不是与 global variance 做 convex
shrinkage。不做 full
dense covariance、per-pixel diagonal、shrinkage family、epsilon grid、结果后 band
合并或多个 whitening variant screening。fit 输入、所有 `v`、`v_global` 和全部
`W` 必须有限，且 `v_global>0`、每个正则化分母严格大于零；否则整个 fit fail closed，
不生成可用 artifact。

### Blind Whitened Matched Score

检测时按上述顺序分别计算 public observation 和 key-only template 的 detrended DCT，
再把 `W[c,band(u,v)]` 广播到非 DC coefficients：

```text
Q_W(Z)[c,u,v] = W[c,band(u,v)] * F(Z)[c,u,v]
s_lf_whitened =
  dot(Q_W(Y), Q_W(T_lf)) /
  (norm2(Q_W(Y)) * norm2(Q_W(T_lf)))
```

dot 与 norm 只遍历非 DC coefficients，并使用固定 channel-major、`u`-major、
`v`-minor binary64 累加；`norm2(A)=sqrt(sum A^2)`，分子与两个平方和各自从
binary64 positive zero 开始独立累加。分母是两个 whitened L2 norm 的乘积，不得换成未白化 norm、
fit-set经验标准差或 score-dependent normalization。missing/mismatched artifact、
artifact payload 重算 digest 与声明值不一致、shape/channel/dtype 漂移、任何非有限量或任一 norm 不严格为正都使该检测调用 fail
closed，不得回退 raw score。registered、wrong-key 与 primary-null 必须复用同一
`W`、VAE/preprocess/config 和 score operator；`W` 对 key 不变。

### Artifact Identity And Responsibility Boundary

fit 责任属于未来获授权的独立 experiment fit runner；`main.lf_detector` 只读消费已
冻结 artifact，不得读取 32 张 fit 图、在检测中更新 `W` 或对 evaluation 重拟合。
artifact canonical payload 使用本文 `stable_json_utf8`，字段至少精确包含：

```text
candidate_id = "lf_null_whitened_matched_score"
artifact_role = "lf_clean_null_whitening_operator"
observation_protocol = "final_image_vae_posterior_mode"
latent_shape = [1,16,64,64]
fit_source_cluster_count = 32
detrend_identity = "per_channel_affine_plane_normalized_coordinates"
transform_identity = "orthonormal_dct_ii"
band_identity = "six_dyadic_chebyshev_frequency_rings_without_dc"
regularization_ratio = "0x1.0000000000000p-10"
fit_manifest_sha256 = <64 lowercase hex>
weights_binary32_be_hex = <channel-major list of exactly 96 eight-hex words>
```

每个权重先按 IEEE-754 round-to-nearest-ties-to-even 物化为 binary32，再序列化为
4-byte big-endian 的 8 位小写 hex；不得用十进制文本重新决定数值。artifact digest
唯一为 `SHA256(stable_json_utf8(payload))` 的 64 位小写 hex，并与 candidate、model、
preprocess 和 detector config identity 一起进入后续 records。raw fit images 不属于
正式检测资产。

### Checks And Evidence Boundary

- 设计/CPU 检查只证明 shape、affine detrend、DCT/ring 分配、单一正则、96 权重
  serialization/digest、有限值/零 norm fail-closed 和 raw-score 非回退。
- 真实 GPU fit 必须在任何候选验证结果之前一次性冻结 32-clean manifest 与 artifact；
  该 fit 不得消费旧 8-cluster artifact，也不得输出 mechanism outcome。
- 随后的全新 validation 比较 registered/wrong-key/paired primary-null；不得由 fit
  本身推导效果、阈值、FPR 或论文 claim。
- `lf_low_pass` raw normalized-correlation 继续作为独立 historical/control score，
  但不得与本候选组成结果后 ensemble，也不得充当本候选失败时的静默 fallback。

## Candidates `routing_stqr` And `routing_uniform_control`

本节冻结 `routing_stqr` 兼容身份及 `routing_uniform_control` causal control 的公式，
仅用于重放对应方法身份，不改变语义—纹理软路由方法。

### Observations

`routing_stqr` 只允许以下四个生成时内容观测：

- `S`：使用 `openai/clip-vit-base-patch32` revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`。把 callback 18 写入前 latent 解码为 `[0,1]` RGB，经官方 image processor 以 `do_rescale=false` 形成 `224 x 224` float32；取 vision `last_hidden_state` 去 CLS 后的 49 个 patch，经 `post_layernorm` 和 `visual_projection` 得到 512 维单位向量。Prompt 以 max length 77、padding、truncation 编码，取 text `pooler_output` 经 `text_projection` 得到单位向量。逐 patch cosine `c` 唯一映射为 `S=clamp((c+1)/2,0,1)` 的 `7 x 7` 图。
- `T`：对同一 RGB 以固定亮度权重 `(0.299,0.587,0.114)` 转灰度，replicate-pad 1 pixel，使用标准 3x3 Sobel x/y，取 `sqrt(gx^2+gy^2)`，除以冻结 `reference_gradient` 后 clamp `[0,1]`。
- `R`：只消费 callback 17 与 callback 18 写入前 scheduler latent。逐空间位置计算 channel RMS difference，除以两帧各自 channel RMS 之和加 `1e-12`，再除以冻结 `reference_response` 后 clamp `[0,1]`。
- `Q_sens`：在 callback 18 写入前 latent 上按
  `key_schedule_sha256_counter` 的 routing public-noise domain 生成绑定 SD3.5
  revision 与 manifest `sample_index` 的公开 Gaussian probe；全 CHW 去均值并归一到
  单位 RMS，以相对 latent RMS 的 `1e-3` 步长扰动一次。分别 VAE 解码原 latent与
  扰动 latent，逐空间位置取 RGB RMS difference 除以实际 probe step，再除以冻结
  `reference_sensitivity` 后 clamp `[0,1]`。

`reference_gradient`、`reference_response`、`reference_sensitivity` 的算法固定为 candidate-selection selection partition 中所有严格正观测值的 exact nearest-rank P95：排序后取索引 `ceil(0.95*n)-1`。它们只能在该 partition 冻结；confirmation、其他 calibration 职责和 evaluation 不得重新拟合。任一正值总体为空即候选失败。

在正式 candidate-selection 之前，development exploration 另行使用隔离的
development-only 四折 cross-fit reference：每个 probe cluster 只能使用其余三折的
严格正观测值并应用同一 exact nearest-rank P95 规则。该 reference 只用于机制探索，
必须在 development 结束时作废，禁止复制到 candidate-selection、confirmation、
calibration 或 evaluation；它不改变上一段唯一正式 reference 的冻结职责。

### Formula And Outputs

四图以 bilinear、`align_corners=false` 映射到 latent grid：

```text
A = ((1-S) * (1-R) * (1-Q_sens)) ** (1/3)
mask_lf = A * (1-T)
mask_hf = A * T
```

`content_router` 的权威输出仅是生成时 observation 结果、`A`、两 mask、
observation/config digests 和 route identity；不输出 `a`、标量分支预算、任何
分支级写入量或 actual-dtype 写入量。`routing_uniform_control` 固定令
`A=mask_lf=mask_hf=1`，并且不读取 S/T/R/Q。配对实验中它的总内容能量、`a`、key、
Prompt、seed 和 write position 与 routed 样本一致，这些由 `content_embedder`、
runtime identity 和实验配对约束保证，不是 router 输出。

检测器不重建 S/T/R/Q，也不读取 Prompt、生成 latent 或 route record；它始终使用未 mask 的 key template 在最终图像 latent 上评分。若候选不能证明这种 marginal score 保持密钥归属，routing 淘汰。

### Source Boundary, Failure And Gate

SLM-WM 源文件只定义候选算法。其 reference registry、历史固定风险权重、私有 reference image 和旧成功/失败记录不迁入。

- CPU/synthetic：router 检查四 observation 边界、resize identity、
  `mask_lf + mask_hf == A`、范围 `[0,1]`、不返回标量预算，以及禁用对照
  `A=mask_lf=mask_hf=1` 且不读取 observations；carrier/embedder 检查 mask 被方向
  消费、零支持失败和 routed/disabled 目标总预算相同。
- 真实 runtime：router 检查四观测真实产生与摘要稳定；embedder/runtime 检查
  route-disabled 配对的 actual-dtype 能量相等；blind detector 独立检查 unmasked
  attribution。
- 晋升：相同预算下 routed 在 confirmation partition 对至少一个预登记攻击族增益，并保持 identity、wrong-key、FPR 和质量边界。
- 淘汰：任一 observation 需 evaluation/攻击标签、检测需私有 route、router 输出
  标量预算、embedder 配对预算不守恒，或无稳定增益。

## Candidate `content_combination_calibrated`

本节冻结旧组合候选以支持 historical package/record replay 与语义对照；它已不是
current content-combination candidate，不得恢复其 execution authority。

输入是独立保存的 `s_hf`、`s_lf`，以及与当前职责 partition 绑定的两份 primary-null
经验分布。对任一分支 `b`、null multiset `X_b={x_1,...,x_n}` 和有限查询分数
`s_b`，唯一变换为：

```text
less  = count(x_i < s_b)
equal = count(x_i == s_b)       # 按保存的 float64 数值精确相等
u_raw = (less + 0.5*equal) / n  # ties 使用 mid-rank
epsilon_n = 1 / (2*n)
u_b = clip(u_raw, epsilon_n, 1-epsilon_n)
i_b = min(1048575, floor(u_b * 1048576))
z_b = float64(frozen_normal_table_float32[i_b])
```

`n >= 2`，null、查询分数和 `z_b` 必须有限。null 保存为 float64 后按
`(score, source_cluster_id, sample_id)` 稳定排序；后两个字段只打破记录顺序，不改变
tie 计数。`frozen_normal_table_float32` 与 `key_schedule_sha256_counter` 使用同一
`2^20` midpoint table 和 table SHA-256；查表结果先按其 IEEE-754 float32 bits
恢复，再无损提升到 float64 参与组合。不得调用另一 inverse-CDF library。任何分支
缺失、null 为空、非有限值、partition identity 不匹配或从别的 detector identity
复用 CDF 都 fail closed。

只比较以下冻结函数：

```text
C_0 = z_hf
C_1(w) = w*z_hf + sqrt(1-w^2)*z_lf, w in {0.25, 0.50, 0.75}
C_2 = max(z_hf, z_lf)
```

候选选择职责内部进一步预登记为 selection/confirmation：

1. 只在 candidate-selection selection primary null 上为每个分支拟合 provisional CDF；
2. 对每个 `hf_only_standardized_score`、
   `weighted_hf_lf_standardized_score` 与
   `maximum_hf_lf_standardized_score` 只在同一 selection partition 上按预登记
   `alpha_selection` 拟合 `tau_provisional`；
3. 用固定 provisional CDF、函数和 `tau_provisional` 在 untouched confirmation
   上执行晋升比较；
4. 选中函数后丢弃全部 provisional CDF 与 `tau_provisional`。

`alpha_selection` 只是候选比较 operating point，数值在 candidate-selection manifest
中预登记；它不是论文 FPR budget，也不得进入正式 records。之后
content-threshold-fit 只对已选定函数，用其独立 primary null 重新拟合 formal branch
CDF 和唯一 `tau`。rescue-window-fit、geometry-reliability-fit、end-to-end check 与
evaluation 均不得重拟合 CDF、组合函数或 `tau`。每个函数有自己的 provisional/formal
阈值，不能共享 HF-only 阈值。

若没有包含 LF 的候选通过门，完整 CEG-WM 内容分支形成负结果。CPU 检查必须覆盖
ties、低/高尾 clipping、排列不变性、严格单调区间、分支交换/权重身份、缺失分支失败
和分数独立持久化。真实检查 LF 增益、HF-only 非退化、wrong-key、primary null、质量
和完整联合 FPR。

## Salient-Object Local-LF Candidate Family

### Candidate Identities

以下四个身份构成一个不可拆分但仍由既有 13 项职责实现的设计候选：

- `routing_inspyrenet_salient_local_lf`；
- `content_embedding_global_hf_local_lf`；
- `lf_saliency_masked_null_whitened_matched_score`；
- `content_combination_saliency_max_standardized`。

这些身份完整定义 hard salient-object local-LF 方法变体。

### Frozen InSPyReNet Authority And Forward

模型资产唯一绑定 Hugging Face repo `plemeri/InSPyReNet` exact revision
`d94c2baaa4d023ab018c6f97be6ef37548e3bd1f` 的 `ckpt_base.pth`：LFS oid
SHA-256 `0a6fe2a73ab0532d6d0b8d82849a9760a226df719e3063d09b4149ece6f80fcd`，
size `367520613` bytes，MIT。source 唯一绑定 `plemeri/transparent-background`
exact revision `f0fa91701a98cfc8e955c554e84522f365ec6da3`，MIT，入口为
`transparent_background/InSPyReNet.py`。Windows `Zone.Identifier`、下载 URL、
本地路径和文件时间永久排除于 checkpoint、package、config 与 record identity。

checkpoint 必须 strict `state_dict` load。唯一前向为直接调用
`InSPyReNet.forward_inspyre(x)`，确认返回 dict 且
`out["saliency"]=[d3,d2,d1,d0]` 后取 `out["saliency"][-1]` 的 finest raw `d0`
logit，再执行 `torch.sigmoid` 恰一次。
禁止 `Remover.process`、`model.forward` 或 `forward_inference`；尤其禁止其 sigmoid
后的 per-image min-max normalization。模型、revision、checkpoint、预处理、raw-output
selector、sigmoid count、resize、threshold、erosion 与 coverage rule 全部进入 mask
identity。

### Single Mask Construction Rule

embed 输入只允许 callback 18 non-terminal latent 的临时 VAE decode RGB8；detect
输入只允许普通待检 RGB8。两侧分别、独立运行同一规则：

1. 以 static `1024 x 1024` RGB、ImageNet mean/std、float32 构造模型输入；
2. 按上一节取得 raw finest `d0` 并 sigmoid exactly once；
3. probability 以 bilinear、`align_corners=false` resize 到 latent grid `64 x 64`；
4. hard threshold `p>=0.5`；
5. 在 `64 x 64` 上执行固定 `3 x 3` square erosion 恰一次，边界 zero padding；
6. eroded mask coverage 必须为 `64..3072` spatial pixels（含端点）。

不存在 connected-component/object selection、soft mask、adaptive threshold、第二次
erosion、dilation 或结果后 coverage 调整。无支持、非局部支持、coverage 越界或 mask 非有限均为
固定分母内失败；禁止 global LF fallback。raw 和 rectified 图必须分别重跑同一模型与
mask rule。detector 不得读取 embed mask。development mask-stability protocol 固定
比较同一预登记单位的 embed/detect mask IoU `>=0.5`，8-unit pilot 至少 `7/8`；该门
不是 formal reliability 或 threshold evidence。

### Global-HF Plus Local-LF Write

`routing_inspyrenet_salient_local_lf` 只提供 `M_embed`、全一 HF support 和 mask
identity。`hf_carrier` 与 `lf_carrier` 仍按原职责生成 `T_hf` 与 `T_lf`；
`content_embedding_global_hf_local_lf` 由既有 `content_embedder` 唯一定义：

```text
u_hf = normalize(T_hf)
u_lf = normalize(M_embed * T_lf)
u_content = normalize(u_hf + u_lf)
```

masked LF 或 sum 为零/非有限即 fail closed。不存在 `a`、`w`、`0.70/0.30`、
`0.50/0.50`、权重 grid 或按 cluster/result 改系数。runtime 按现有 actual-dtype
materialization 执行；最终 combined actual delta 仍受 canonical binary32 `3/250`
total relative-L2 hard limit，不得建立分支预算或放宽总预算。

每个科学 probe 必须有独立 LF-only causal witness：actual LF delta 非零、mask 外
逐 bit 为零、mask 内具有非零能量。witness 只证明局部写入因果边界，不是检测阳性。
combined arm 的 actual delta 含非正交与 actual-dtype 物化效应，不得伪造或声称可
分解为 HF/LF branch contributions。

### Blind Masked-LF Score And Null Fit

`lf_saliency_masked_null_whitened_matched_score` 在检测侧从 ordinary public RGB8
重新构造 `M_detect`；该 mask 同时作用于 public VAE posterior observation 与由检测
key 重构的 LF template。检测不得消费 reference image、Prompt、embed record、private
latent、Q/K、embed mask 或任何生成时 mask state。

masked-LF 必须在新的、预登记且与旧 fit/screening/directional/candidate-selection/
calibration/evaluation source clusters 零交集的 32-clean-null manifest 上重新拟合
自己的 whitening `W`。fit 使用 masked public observation，固定分母 32；不得继承、
转换或 fallback 到旧 unmasked `lf_null_whitened_matched_score` 的 `W`。新的 W、fit
manifest、InSPyReNet/mask identity、VAE/preprocess、LF template 与 detector identity
共同进入 artifact digest。registered、primary-null 和 wrong-key 均使用同一新 W、
同一 mask rule 和同一 public scoring operator。

### Max-Statistic Content Detection

`z_hf` 与 `z_lf_masked` 必须分别从其 detector identity 专属、互不替代的 primary-null
分布标准化。`content_combination_saliency_max_standardized` 唯一统计为：

```text
D_saliency_max = max(z_hf, z_lf_masked)
```

不存在 weight/function search、attack-conditioned switching 或用 LF failure 回退 HF-only
后仍声称同一候选。未来 formal threshold 必须直接对 max statistic 自身，在独立
content-threshold-fit primary null 上重新拟合；不得沿用 HF-only、旧 unmasked LF 或
旧 `content_combination_calibrated` 的 CDF/threshold。raw 与 rectified image 必须使用
同一 max-statistic detector identity、key、preprocess、mask rule 和 threshold。

### Permanent Result-Separation Rules

- 不从旧 8 probes 选择 mask、threshold、erosion、coverage、`a`、`w` 或 function；
- 不补样、删 cluster、重跑、增加 attempt、放宽 `3/250` 或重写历史 artifact；
- 不用 margin-only 子集形成 winner、promotion 或 candidate selection；
- hard-mask、unmasked 或其他 detector 的 null/threshold 不得传递到该候选；
- 不把代码存在或其他身份可重放解释为本候选效果证据。

### Falsification Gates

验证必须覆盖 external asset license/checkpoint/source/API、精确 mask golden、
coverage/failure、causal witness、
32-clean-null W fit、public blind key attribution、mask-stability、total-budget 与
max-statistic identity tests。任一环节需要改变 checkpoint、forward path、threshold、
erosion、coverage、写入公式、W fit 或 max statistic 时，必须登记新候选身份；不得在
实现或 GPU 结果后静默调参。

## Semantic-Texture Soft-Routing Candidate Family

以下五个身份共同定义 CEG-WM 的语义—纹理软路由内容链：

- `routing_semantic_texture_soft`；
- `content_embedding_semantic_texture_soft_lf_hf`；
- `lf_semantic_texture_soft_whitened_matched_score`；
- `hf_semantic_texture_soft_direct_score`；
- `content_combination_semantic_texture_max_standardized`。

五者仍由既有 `content_router`、LF/HF carrier、`content_embedder`、LF/HF detector 和
`content_detector` 分工，不增加方法组件。

### Public Semantic And Texture Observations

嵌入端输入是 callback 18 non-terminal latent 的临时 VAE decode RGB8；原图检测与
回正检测输入都是各自的普通 RGB8 图像。每次调用都独立重建观察量，不共享 Prompt、
embed record、private latent、embed-side map 或参考图。

语义图 `M` 复用本文件登记的 InSPyReNet source/checkpoint/strict-load 和
`forward_inspyre` finest raw `d0`/sigmoid-once 规则。输入为 static `1024 x 1024`
RGB、ImageNet mean/std、float32；probability 以 bilinear、`align_corners=false`
映射到 `64 x 64`。输出直接作为 `M in [0,1]`；禁止 hard threshold、erosion、
connected-component selection、per-image min-max 和 coverage fallback。

纹理图 `T` 从同一个 RGB8 构造。按 `(0.299,0.587,0.114)` 转为 binary32 灰度，
replicate-pad 1 pixel，执行标准 3x3 Sobel x/y，令
`G=sqrt(gx^2+gy^2)`，再以 area downsample 映射到 `64 x 64`。对严格正的 `G`
按 `(value, flat_index)` 稳定升序，取 `q95=sorted[ceil(0.95*n)-1]`；若 `n=0`，
令 `T=0`，否则 `T=clamp(G/q95,0,1)`。不从跨样本 reference、攻击标签、分数或
evaluation 结果拟合纹理尺度。

### Soft Route And Causal Control

`content_router` 唯一输出：

```text
m_hf = (1 + M*T) / (2 + M)
m_lf = (1 + M*(1-T)) / (2 + M)
```

逐元素要求 finite、`M,T in [0,1]`、`m_hf,m_lf>0` 且按 binary32 协议验证
`m_hf+m_lf=1`。router 返回四张 map 的 identity/digests，但不返回标量预算或攻击
类别。route-disabled causal control 固定 `m_hf=m_lf=0.5` 且不得读取 `M/T`。

### Dual-Frequency Embedding

carrier 仍按独立职责域生成 `T_hf` 与 `T_lf`。`content_embedder` 唯一构造：

```text
u_hf = normalize(m_hf_embed*T_hf)
u_lf = normalize(m_lf_embed*T_lf)
u_content = normalize(u_hf+u_lf)
delta_content_nominal =
  (3/250)*norm32(fp32(z0))*u_content
```

`m_hf/m_lf` 是空间调制，不是 branch energy。不存在 `a/w`、固定 `0.70/0.30`、
攻击条件切换或结果后分配。actual-dtype materialization、binary32 hard comparison、
最大非零可行 scale 和最终 `3/250` combined total limit 完全复用
`runtime_sd35_flowmatch` 规格。任一 active direction 或 combined direction 为零、
非有限或身份不一致时 fail closed。

### Blind Branch Scores

检测端从待检 RGB8 独立重建 `m_hf_detect/m_lf_detect`。HF 分数固定为对
`m_hf_detect*Y` 与 `m_hf_detect*T_hf` 做 score-time centered normalized
correlation。LF 分数沿用 affine detrend、orthonormal DCT-II 和 six dyadic
Chebyshev bands，但在 observation/template 进入该算子前同时施加
`m_lf_detect`；其 `W` 必须由专属、互斥的 32-clean primary-null fit 产生，不能
继承 unmasked 或 hard-mask candidate 的 `W`。

registered、wrong-key 与 unwatermarked 调用必须使用同一 public route、detector、
`W` 和预处理。任一 route、artifact、shape、norm 或 finite 检查失败均显式失败，
不得回退 global HF 或 unmasked LF 后仍声称同一候选。

### Fixed Content Statistic

HF 与 LF 分支分别从各自专属 primary-null 分布标准化为 `z_hf_soft` 和
`z_lf_soft`。组合唯一为：

```text
D_soft_route = max(z_hf_soft, z_lf_soft)
```

每个分支的标准化算子固定使用有限样本 mid-rank empirical CDF、`1/(2n)` tail
clipping 和与 key schedule 同摘要的 `2^20` midpoint float32 normal quantile
table。这里只复用共享算法原语；该原语和 table digest 进入
`content_combination_semantic_texture_max_standardized` 身份。旧
`content_combination_calibrated` 的 candidate identity、CDF artifact、split、
threshold、选择结果和效果证据均不得继承。

不搜索 weight/function，不按攻击类型切换。candidate-selection 的两条 provisional
branch CDF 必须从 soft-route 候选专属 primary null 拟合，并在 confirmation 后
丢弃。formal branch CDF 与 `tau` 必须对该 max statistic 身份在独立
content-threshold-fit 上重新拟合；raw 与 rectified 调用共享完全相同的 route、
detector、key、preprocessing、`W`、标准化身份和 `tau`。

### Falsification Gates

- `M/T` 及软路由必须确定、可从普通 RGB8 盲重建，并通过 route-disabled 因果对照；
- LF/HF 两条分支都必须单独具有 registered/wrong-key 与 primary-null 可校准性；
- soft-routed combination 必须在同一总预算下提供预登记增益，且不降低 HF-only
  归属、质量或 FPR；
- route、branch scores、budget、quality 和所有失败保留固定分母；
- 任一公式、观察量、纹理 normalization、InSPyReNet forward、总预算或 max statistic
  改变都产生新候选身份。

## Candidate `qk_relation_similarity`

### Cross-Component Ownership

该候选同时约束 `qk_geometry_sync` 与 `geometric_transform_estimator`，但不把两项
职责合并。`qk_geometry_sync` 消费 runtime 暴露的登记层原始 Q/K，独占 relation
tensor、geometry-key projection、relation score、同步目标和几何写入方向，并在
盲检侧输出冻结的 `R_obs` 与 `R_key`；`geometric_transform_estimator` 只消费这些
relation 输出，把同一 relation scorer 用作 `rectification_similarity` 的变换搜索
目标，并输出 best transform 与原始估计指标。runtime 只按冻结 identity
执行 image-only forward 和捕获真实 Q/K，不拥有 relation 公式。该候选不输出
`reliable`，可靠性属于独立 `geometry_reliability`。

### Runtime Observation

嵌入同步和盲检观察都使用 schedule index 7。盲检先以 VAE posterior mode 得到
image latent；公开噪声使用 `key_schedule_sha256_counter` 中冻结的 image-only Q/K
public-noise domain，绑定 shape、model revision、schedule index、conditioning 和
tensor role，由 scheduler `scale_noise` 在 index 7 加噪。条件协议固定为
`sd3_empty_text_triplet_without_cfg`：`prompt`、`prompt_2`、`prompt_3` 全为空字符串且
`do_classifier_free_guidance=false`。缺少 `scale_noise` 或任一身份不匹配即失败。

随后从 `runtime_sd35_flowmatch` 的 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn` 直接调用 `to_q`、`to_k`，使用模块实际 head layout、Q/K normalization 和 `1/sqrt(head_width)` scale。只取方形图像 token grid，按等距规则最多采样 `8 x 8` token；层、head、token index、Q/K 内容和算子元数据全部进入身份。

每层每个 head 先构造 `A_h = Q_h K_h^T / sqrt(d)`，key axis 是最后一维。
row-centering 在 key axis 上逐 query row 执行，之后才跨 head 求均值：

```text
L_h[i,j] = A_h[i,j] - mean_k A_h[i,k]
L[i,j] = mean_h L_h[i,j]
P[i,j] = mean_h softmax(A_h[i,:])[j]
```

令 `n` 为 token 数。降序 differentiable row-rank 的唯一公式为：

```text
rank[i,j] =
  (1 + sum_{k != j} sigmoid((L[i,k]-L[i,j])/0.25)) / n
P_row[i,j] = P[i,j] / max(sum_k P[i,k], 1e-12)
```

原始方形 grid side 为 `g`，原 token index `t` 的坐标固定为
`x=-1+2*(t mod g)/(g-1)`、`y=-1+2*floor(t/g)/(g-1)`；这与
`align_corners=true` 一致。`d[i,j]=||p_i-p_j||_2/(2*sqrt(2))`，
`P_c=P_row-mean_k(P_row)`，`d_c=d-mean_k(d)`。四通道唯一为：

```text
R[:,:,:,0] = L
R[:,:,:,1] = rank
R[:,:,:,2] = P_row
R[:,:,:,3] = P_c * d_c
```

### Geometry-Key Projection And Score

每层用 `key_schedule_sha256_counter` 的 Q/K domain 生成 `[n,n]` uniform。阈值
`>=0.5` 为 `+1`，否则 `-1`；只取严格上三角，再镜像到下三角，对角严格为 0。
同一 base sign matrix 通过 polarity `(+, -, +, +)` 扩展到四通道。不存在逐通道
独立随机图。

每个通道逐 query row 排除对角并做加权中心化相关。首个候选的 pair weights 全部为
1；不存在实现时选择 stable-token weighting：

```text
c_row = sum_j w_ij (R_ij-rbar_i)(Pi_ij-pibar_i)
        / sqrt(sum_j w_ij(R_ij-rbar_i)^2
               * sum_j w_ij(Pi_ij-pibar_i)^2)
c_layer_channel = mean(c_row over rows with both energies > 1e-24)
c_layer = mean(c_layer_channel[0:4])
relation_score = mean(c_layer over the two registered layers)
```

`w_ii=0,w_ij=1`；均值 `rbar/pibar` 只在有效 `j` 上计算。任一层/通道没有有效 row
即 fail closed，不能以 0 替代。两层和四通道均为等权，不能在实现或 calibration
中改权；descriptor digest、projection digest、层序和 polarity 都进入身份。

### Synchronization Objective And Write

嵌入端目标与盲检端都必须走同一 `image-only empty-condition forward`，不得把生成
时 conditional forward 的 Q/K 与检测端混用。对每个待评分 latent，先执行剩余
scheduler suffix 得到普通 RGB 图像，再以 VAE posterior mode 重编码，使用
`key_schedule_sha256_counter` 的公开噪声在 schedule index 7 调用
`scheduler.scale_noise`，以三路空字符串且无 CFG 的 Transformer forward 提取上述
两层 Q/K。盲检从输入普通图像执行完全相同的 image-only 路径。生成 forward Q/K、
callback Q/K cache 或 embed record 都不是该候选观测。

为使嵌入梯度的 forward 值与普通 RGB8 检测输入一致，suffix 解码后的 `[0,1]`
float 图像使用唯一 straight-through 量化：

```text
I_q = floor(clamp(I,0,1)*255)/255
I_q_ste = I + stop_gradient(I_q-I)
```

梯度 forward 把 `I_q_ste` 送入同一 VAE-mode image-only 路径；因此 forward pixels
逐值等于 RGB8，backward 只对量化使用 identity STE。actual-dtype line search 与
最终检测不使用 STE，而是物化 `uint8=floor(clamp(I,0,1)*255)` 后重新读取和评分。

在 callback index 18 的 content-only float32 baseline 上，对上述 final-image
relation score 求梯度。令 active content direction matrix `U` 的列依固定顺序
`LF,HF` 取实际 masked direction 的单位化 flatten 值；未启用分支不入列。非正交
投影固定为：

```text
g_geo = grad_z relation_score(Q(z), K(z), geometry_key)
P_U = U (U^T U)^dagger U^T
g_perp = g_geo - P_U g_geo
u_geo = g_perp / ||g_perp||_2
rho_geo / rho_content in {1/16, 1/8, 1/4}
delta_geo = rho_geo * ||z||_2 * u_geo
```

`dagger` 用 float64 SVD Moore-Penrose inverse，保留
`sigma_i > 1e-6*sigma_max` 的奇异方向；`g_perp` 非有限或范数不超过
`1e-12*||g_geo||` 时失败。content update 先物化为
`z_content_actual=cast(z+delta_content)`；几何只在其上追加一次最终 dtype 写入。
对 ratio `r`，总预算为
`rho_total=(3/250)*sqrt(1+r^2)`。

按 `{1,1/2,...,1/128}` 顺序测试 `lambda`。每次必须完整 replay suffix 和同一
image-only score，并对实际 dtype 张量计算：

```text
d_geo_actual = float32(z_candidate_actual) - float32(z_content_actual)
d_total_actual = float32(z_candidate_actual) - float32(cast(z))
```

接受第一个同时满足：score 严格大于 baseline；`||d_geo_actual||>0` 且不超过
`lambda*rho_geo*||float32(z)||`；`||d_total_actual||` 不超过
`rho_total*||float32(z)||`；以及
`||P_U d_geo_actual|| <= 1e-4*||d_geo_actual||` 的候选。比较对象、范数和投影均为
上述实际 dtype 差值，不得用理论 update 冒充。全部失败则 geometry
synchronization 明确失败，不能退化为未记录的 content-only full method。

历史 `sync_update.py` 是算法来源候选，但其固定 geometry strength、旧风险预算和旧 evidence 不迁入。

### CPU/Synthetic And Runtime Gates

- CPU/synthetic：对 `qk_geometry_sync`，必须消费真实 Q 与 K 数值；只改 Q 或只改 K
  会改变 relation，摘要不能替代 tensor，且 key domain 分离、回溯只接受实际 score
  提升。对 `geometric_transform_estimator` 的消费边界，合成
  identity/rotation/scale/translation/crop 下关系图按 `W R W^T` 变换，错误 key
  不产生正确 projection 高分。
- 真实 runtime：runtime 完成两登记层真实捕获且 Q/K 原子内容与算子身份完整；
  `qk_geometry_sync` 的同步写入改善正确 key relation、错误 key 不改善；
  estimator 只消费冻结 relation 输出；内容分数和质量不超预登记退化。
- 晋升：所有真实观察与合成可辨识检查通过。
- 淘汰：使用 hidden-state proxy、缓存 embed Q/K、reference image、摘要代替 Q/K，或无法稳定提高 relation。

## Candidate `rectification_similarity`

### Transform Estimation

估计参数是规范坐标到观测坐标的 similarity affine。token 坐标严格复用
`qk_relation_similarity` 的 `[-1,1]` xy、row-major、corner-endpoint、
`align_corners=true` 约定：

```text
A(d,phi,ell,tx,ty) =
  [ exp(ell) R(phi) D_d | (tx,ty)^T ]
```

`D_d` 按本文列出的 index 先作用于 canonical 坐标，之后依次施加 residual
rotation/scale，最后在 observed 坐标加 translation；禁止交换次序。八个
`D_d` 依次为：

```text
identity  [[ 1, 0],[ 0, 1]]
x_flip    [[-1, 0],[ 0, 1]]
y_flip    [[ 1, 0],[ 0,-1]]
xy_flip   [[-1, 0],[ 0,-1]]
rot90     [[ 0,-1],[ 1, 0]]
rot_minus90 [[0, 1],[-1, 0]]
diag      [[ 0, 1],[ 1, 0]]
anti_diag [[ 0,-1],[-1, 0]]
```

crop 由双向有效支持和
translation 联合表征，不假装恢复被删除像素。支持域固定为：

- rotation residual `[-32, 32]` degrees，并显式包含方形网格八个 dihedral 基元；
- scale `[1/sqrt(2), sqrt(2)]`；
- normalized translation `[-0.28, 0.28]`；
- 最低双向 coverage 候选门 `0.45`。

粗搜索枚举顺序唯一为：

```text
d = [identity,x_flip,y_flip,xy_flip,rot90,rot_minus90,diag,anti_diag]
phi = [0,-32,-16,16,32] degrees
ell = [0,-log(sqrt(2)),+log(sqrt(2))]
tx = [0,-0.28,+0.28]
ty = [0,-0.28,+0.28]
```

五层循环按上述从左到右，重复 matrix 只保留第一次出现。随后固定 `d` 做 3 轮
局部细化；每轮相对当前 best 枚举 offsets
`[0,-delta,+delta]` 的 `phi,ell,tx,ty` 全笛卡尔积，组合仍按
`exp(dell)R(dphi)` 左乘当前连续 linear part、translation 直接相加。第一轮
`delta=(8 degrees, log(sqrt(2))/2, 0.14, 0.14)`，后两轮依次除以 3。越界候选在
评分前删除。每轮只用本轮 best 进入下一轮；最终 best 从 coarse 与三轮全部候选中
选择。score 严格 `>` 才替换，float32 完全相等时保留枚举中首次出现者；不能使用
攻击参数、随机重启或库排序平局。

### Sampling Matrices And Objective

给定完整规则 token grid `p_j`，对每个 canonical row `i` 计算
`q_i=A(theta)p_i`。若 `q_i` 在 grid xy 两轴闭区间内，则 `W[i,:]` 是其四个相邻
grid points 的标准 separable bilinear weights，和为 1；否则该行全 0 且
`valid_forward[i]=false`。`V` 对 `A(theta)^-1 p_i` 使用完全相同规则。两个 matrix
都按原 token index 顺序构造，禁止 nearest-neighbor 或 padding。

每层四通道分别计算：

```text
canonical_R_c = W R_obs,c W^T
expected_observation_R_c = V R_key,c V^T
canonical_score_l = row_normalized_score(canonical_R, R_key,
                                         valid_forward)
observation_score_l = row_normalized_score(R_obs,
                                           expected_observation_R,
                                           valid_backward)
canonical_score = mean_l(canonical_score_l)
observation_score = mean_l(observation_score_l)
```

`row_normalized_score` 精确复用 `qk_relation_similarity` 的逐 row 排除对角、加权
中心化相关、四通道等权聚合；两登记层再等权平均。canonical 的 pair weight 由
`W 1` 的 token weight 外积，observation 使用全 1 off-diagonal weight。

四个 deficit 唯一为：

```text
coverage_forward  = mean(valid_forward)
coverage_backward = mean(valid_backward)
unique_forward =
  count(unique(argmax_j W[i,j]) for valid_forward rows) / count(valid_forward)
unique_backward =
  count(unique(argmax_j V[i,j]) for valid_backward rows) / count(valid_backward)
deficits = (1-coverage_forward) + (1-coverage_backward)
           + (1-unique_forward) + (1-unique_backward)
objective = 0.10*canonical_score + 0.90*observation_score
            - 0.01*deficits
crop_support = min(coverage_forward, coverage_backward)
```

零 valid row 时对应 coverage/unique 都为 0。最优 candidate 必须相对精确 identity
candidate 有正 objective margin；second-best 是去除与 best matrix float32
逐元素完全相等的重复项后 objective 次高的候选，平局沿用上述首次出现规则。

### Estimator Raw Metrics

`geometric_transform_estimator` 对注册 geometry key 和由
`key_schedule_sha256_counter` 按预登记 index `0..7` 派生的 8 个错误 key 分别运行
同一搜索，只输出 best transform、搜索目标及以下原始指标：

```text
gap = best_registered - second_registered
identity_margin = best_registered - exact_identity_registered
key_margin = best_registered - max(best_wrong_key[0:8])
coverage = min(forward_coverage, backward_coverage)
uniqueness = min(forward_uniqueness, backward_uniqueness)
```

12 个锚点固定为 `(-1,-1),(-1,1),(1,-1),(1,1)`、`(-1,0),(1,0),
(0,-1),(0,1)` 和 `(-0.5,-0.5),(-0.5,0.5),(0.5,-0.5),(0.5,0.5)`。
每个 anchor 经 best transform 后到 observed grid 最近点的欧氏距离为 residual；
越界或映射到已被其他 anchor 占用的同一最近点不是 inlier。inlier ratio 的分母始终
是 12，使用同一个拟合阈值 `epsilon_inlier`；mean residual 是 12 个 residual 的
均值，越界 residual 记 `+inf`。estimator 不得把这些指标合并成 `reliable`，也不得
自行应用门限。

### Geometry Reliability Conjunction

`geometry_reliability` 独占可靠性判定。可靠性计算不是实现者可选择的分类器；它消费
estimator 的 best transform、原始指标和拟合阈值，并固定执行以下合取，不允许换成
学习分类器或加权分数：

```text
reliable =
    all_metrics_finite
    and coverage >= max(0.45, gamma_coverage)
    and uniqueness >= gamma_uniqueness
    and gap >= gamma_gap
    and key_margin >= gamma_key
    and inlier_ratio >= gamma_inlier
    and mean_residual <= gamma_residual
    and not continuous_parameter_on_search_boundary
    and (best_is_exact_identity or identity_margin >= gamma_identity)
```

`gamma_coverage`、`gamma_uniqueness`、`gamma_gap`、`gamma_key`、`gamma_inlier`、`gamma_residual`、`gamma_identity` 和 `epsilon_inlier` 只由独立 geometry-reliability-fit 职责冻结；在拟合前实现必须返回全部原始指标和 `reliability_not_fitted`，不得自行选择阈值。

development exploration 的未拟合 estimator 调用允许令 `epsilon_inlier=None`，只返回
变换搜索及 residual 等原始量，并明确令 inlier 统计未拟合；随后仅从隔离的
development COMMITTED residuals 做 cross-fit 的 exploratory epsilon/gamma，再执行其
依赖单元。该组值在 development 结束时作废，绝不构成上述正式
geometry-reliability-fit 阈值。

低 coverage、高残差、多峰歧义、边界解、错误 key、非有限 metric/matrix 或
`reliability_not_fitted` 全部由 `geometry_reliability` fail closed。可靠性输出
不能进入内容分数，也不能直接产生阳性。

### Image Rectifier Boundary

只有 `geometry_reliability` 返回可靠时，`image_rectifier` 才消费 estimator 的
canonical-to-observed matrix。它对 RGB uint8 `[1,3,H,W]` 直接将该 matrix 作为
PyTorch output-to-input inverse-warp `theta`（对每个 canonical output coordinate
查询 observed input coordinate），由 `affine_grid/grid_sample` 执行：

- bilinear；
- padding `border`；
- `align_corners=true`；
- 输出同尺寸；
- clamp `[0,1]`、乘 255、floor、转 RGB uint8；
- 同时返回 valid-support mask。

valid-support mask 以同一 grid 对全 1 input 做 zero-padding、nearest sampling 后取
`>0.5`；图像本身仍使用 `border` padding，因此 mask 才是 crop/越界的权威支持，
border 像素不计作已恢复。`crop_support` 必须同时报告 token 双向 coverage 与该
pixel mask 的有效比例。

`image_rectifier` 不重新估计参数、不计算可靠性；非有限 matrix、无效图像或 warp
失败由它 fail closed。padding 像素不表示恢复了 crop 内容。

CPU/synthetic 必须分别覆盖 estimator 的 identity/单变换/组合变换/crop raw
metrics、reliability 的错误 key/双峰/越界合取失败，以及 rectifier 的 inverse-warp
和 valid-support mask；真实 runtime 必须分别报告参数误差、coverage、reliability、
回正质量及同 detector 内容分数变化。

## Candidate `joint_conditional_recovery`

输入是普通 RGB 图像、检测 key、一个不可变 content detector identity、`tau`、`tau_rescue` 和上述 geometry/rectification config。输出保存 raw score、触发原因、几何结果、rectified score、最终判定和失败状态。

算法唯一为：

```text
s_raw = D_M(I, key)
if s_raw >= tau:
    positive
elif s_raw < tau_rescue:
    negative_without_geometry
else:
    geometry = estimate(I, geometry_key)
    if not geometry.reliable:
        negative_with_geometry_failure
    else:
        I_rect = rectify(I, geometry.transform)
        s_rect = D_M(I_rect, key)
        positive iff s_rect >= tau
```

`D_M` object identity、content configuration digest、key semantics、preprocessing 和 `tau` 在两次调用中必须相同。geometry score、confidence 或 wrong-key relation 不能直接阳性。

development exploration 的 `tau_rescue` 只由同一四折 fit-fold 的已验证
COMMITTED primary-null 产生：先使用该折 development exploratory `tau`，对所有严格
小于 `tau` 的分数计算正 margin `tau-score`，按 exact nearest-rank P05 取 margin，
并令 `tau_rescue=tau-margin`。probe clusters 不得参与拟合，只能验证；该值在
development 结束时作废，不能替代未来独立 rescue-threshold-fit 的正式值。

CPU 检查三路门控、几何失败、可靠但内容仍负、同 detector/threshold identity 和几何不可直接阳性。真实 runtime 检查 raw/rectified 完整重编码、rescue 触发成本和 end-to-end FPR。

## Candidate Specification Closure

本文为候选集合关闭 key/KDF/PRG、relation/objective、LF/HF write/score、
routing observations、backbone/runtime、搜索、可靠性指标、回正和联合判定的实现
选择空白。registry 合计 20 个 ID：19 个具名候选，加上
`routing_uniform_control` 这一项强制同预算禁用对照。对照只用于因果验证，不参与
方法身份，也不把 20 个候选 ID 误写为 13 项实现职责。

### Frozen Specification Values And Empirical Quantities

候选规格固定：key encoding/KDF/PRG 与 golden bits；SD3.5 revision 和
runtime protocol；HF sparse-tail/filter/write/score 顺序与候选强度；LF
filter/write/raw score、唯一 clean-null whitening fit/matched score 与有限 `a` 集；
S/T/R/Q observations；empirical-CDF/tie/clip/table
规则与三条语义化组合函数；Q/K 层、前向、四通道、projection、聚合、subspace 和 line
search；similarity/dihedral 搜索、W/V、objective、raw reliability metrics 与
rectification；conditional recovery 控制流；InSPyReNet soft `M`、Sobel/P95 `T`、
正软路由图、soft-routed LF/HF write、两条盲 branch scores、独立 null fit 与固定
max-statistic identity。

软路由 write 没有 `a/w` grid，检测统计固定为 max。只能由预登记实证决定的是：
候选是否通过 mechanism validation 与独立 confirmation、
`alpha_selection`、formal branch CDF、`tau`、`tau_rescue`、七个 geometry reliability
`gamma` 与 `epsilon_inlier`、各职责样本量、候选是否晋升/淘汰、runtime 是否可复现，
以及 FPR/TPR、鲁棒性、质量、成本和完整/负结果/reduced-scope outcome。
