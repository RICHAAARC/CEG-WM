# CEG-WM Candidate Specifications

候选规格中的组合身份只使用 `hf_only_standardized_score`、
`weighted_hf_lf_standardized_score`、`maximum_hf_lf_standardized_score`；
`C_0`、`C_1(w)`、`C_2` 仅保留为本地数学记号。

## Authority And Status

本文关闭“实现时由 Codex 自行发明算法”的空白。这里登记的是可实施、可证伪的候选规格，不是已验证项目参数、有效性结论或实施授权。

当前 registry 固定为 **11 个候选 ID**：其中 10 个是待实施与晋升的
method/runtime/key 候选，`routing_uniform_control` 是强制保留、不得晋升为方法的
同预算禁用对照。计数如下：

- `key_schedule_sha256_counter`；
- `runtime_sd35_flowmatch`；
- `hf_sparse_tail`；
- `lf_low_pass`；
- `lf_null_whitened_matched_score`；
- `routing_stqr`，并以 `routing_uniform_control` 为同预算禁用对照；
- `content_combination_calibrated`；
- `qk_relation_similarity`；
- `rectification_similarity`；
- `joint_conditional_recovery`。

实现者只能实现这些身份及其明确列出的有限候选值。增加 relation、objective、write、score、observation、backbone、runtime 或搜索策略，必须先修订本文并重新接受候选规格审计。

候选 registry 与实现职责是两个不同计数：上面的 11 个 ID 描述算法/runtime/key
候选身份；未来实现固定为 13 项职责组件。每项职责只消费下表列出的现有候选，
不得据此新增候选、别名组件或把多个职责集中到一个代理模块：

| component | responsibility | planned path | candidate binding |
| --- | --- | --- | --- |
| `key_schedule` | `root_key_derivation_and_prg` | `main/shared/key_schedule.py` | `key_schedule_sha256_counter` |
| `content_router` | `content_observation_and_adaptive_routing` | `main/content_chain/routing.py` | `key_schedule_sha256_counter`, `routing_stqr`, `routing_uniform_control` |
| `lf_carrier` | `low_frequency_carrier_template_and_write_direction` | `main/content_chain/lf_carrier.py` | `key_schedule_sha256_counter`, `lf_low_pass` |
| `hf_carrier` | `high_frequency_carrier_template_and_write_direction` | `main/content_chain/hf_carrier.py` | `key_schedule_sha256_counter`, `runtime_sd35_flowmatch`, `hf_sparse_tail` |
| `content_embedder` | `lf_hf_combined_embedding_and_total_budget` | `main/content_chain/embedder.py` | `runtime_sd35_flowmatch`, `hf_sparse_tail`, `lf_low_pass`, `routing_stqr`, `routing_uniform_control` |
| `lf_detector` | `low_frequency_blind_scoring` | `main/content_chain/lf_detector.py` | `key_schedule_sha256_counter`, `lf_low_pass`, `lf_null_whitened_matched_score` |
| `hf_detector` | `high_frequency_direct_scoring` | `main/content_chain/hf_detector.py` | `key_schedule_sha256_counter`, `hf_sparse_tail` |
| `content_detector` | `lf_hf_score_standardization_and_content_detection` | `main/content_chain/detector.py` | `hf_sparse_tail`, `lf_low_pass`, `lf_null_whitened_matched_score`, `content_combination_calibrated` |
| `qk_geometry_sync` | `keyed_qk_geometry_synchronization_and_relation_observation` | `main/geometry_chain/qk_sync.py` | `key_schedule_sha256_counter`, `runtime_sd35_flowmatch`, `qk_relation_similarity` |
| `geometric_transform_estimator` | `blind_bounded_geometric_transform_estimation` | `main/geometry_chain/transform_estimator.py` | `key_schedule_sha256_counter`, `qk_relation_similarity`, `rectification_similarity` |
| `geometry_reliability` | `independent_geometry_reliability_conjunction` | `main/geometry_chain/reliability.py` | `key_schedule_sha256_counter`, `qk_relation_similarity`, `rectification_similarity` |
| `image_rectifier` | `image_coordinate_rectification` | `main/geometry_chain/rectifier.py` | `rectification_similarity` |
| `conditional_recovery_decision` | `conditional_same_detector_recovery` | `main/joint_decision/detector.py` | `joint_conditional_recovery` |

`content_embedder` 独占 `u_content(a)`、nominal/actual hard limit、mixing
coefficients、组合 delta 的 materialization reconciliation 与 realized
norm/relative L2 和 active 零方向失败；`lf_detector`
独占盲 `s_lf`；`geometry_reliability` 独占 estimator
原始指标上的可靠性合取门。这三项不能由 carrier、content detector 或 transform
estimator 代行。候选绑定表示该组件必须实现或消费的规格身份，不表示候选已经晋升。

## Provisional Historical Provenance

### Read-Only Revisions

| source | read-only revision | state observed | authority |
| --- | --- | --- | --- |
| `SLM-WM-FlowHF` | `a7f33825d0913d4707af5723b236beb65f53f4e5` | tracked worktree clean | historical DirectHF source for the CEG-WM HF candidate only |
| `SLM-WM` | `47bd9a1850c434aa47ee03caa7377706f4d283de` | tracked files clean; `.codex/config.toml` and `docs/ceg_wm_direct_hf_scope_decision.md` untracked | LF/routing/QK candidate source only |
| `SLM-WM` FlowHF baseline | `34825098553d22f68f188afcd938d0aa72132caf` | Git object verified | upstream identity referenced by FlowHF |

本表是 provisional provenance：它证明读过哪个历史 Git object，不证明任何代码已经迁入 CEG-WM，也不替代未来 CEG-WM revision。

两个历史仓库根目录均未发现许可证或 copying 文件。用户可以在此缺口仍存在时授权建立 CEG-WM 版本身份，但在实际复制历史代码前必须由用户确认复用权，或明确授权按本文公式进行不复制源码的独立重写。缺口未关闭时，代码迁移 fail closed。

### Historical DirectHF Source Files

`SLM-WM-FlowHF` revision `a7f33825...`：

| path | SHA-256 |
| --- | --- |
| `flowhf/hf_injector.py` | `03dab6c32d801b712362264584c8b30567e2ab44b88678af2e0c44f27c433cf4` |
| `flowhf/direct_detector.py` | `ea5c5d8ffa34faea4cf7b88d03f78296a3ddd9e44cfbc3e767c366898ea9fd1c` |
| `flowhf/evaluate_keys.py` | `3ce54b65f72f59ac0cde7c132cb58947c05f3af2a1012a1f8b1d78b49a5f372d` |
| `flowhf/key_plan.py` | `c83808d07a6400cfeb3405be5faaeb893d5cb408485a18fb58661ab48f3a9837` |
| `flowhf/model_runtime.py` | `35fcb73c5c78250fc7ea11620f8d1ceb360c13dd298d81a2bbe914c39d7f6de9` |
| `flowhf/run_spec.py` | `cccd166439f0f0be5cfa5281ce8d6eaf9a61005dd8f8452b22516c14a19aee9c` |
| `tests/fixtures/hf_template_golden.json` | `d3f7e9c77ffeecd6f0a5615582bb09b1a2aa170169a71ef4da30ed7ad5483b25` |

FlowHF 只提供 historical DirectHF 的四 Prompt 小样本来源证据。它不提供 CEG-WM HF 成功结论，也不提供 population、fixed-FPR、攻击鲁棒性、LF、路由或 Q/K 成功证据。

### LF, Routing And Q/K Candidate Files

`SLM-WM` revision `47bd9a...`：

| responsibility | path | SHA-256 |
| --- | --- | --- |
| LF template | `main/methods/carrier/low_frequency.py` | `c5d2a4f7cf0879987801372e135e5e537ea2bbe28b3c505300e2759add95bf24` |
| routed composition | `main/methods/carrier/content_update.py` | `f85f2bee8efa5019f1cf34b9e02035b2bf50baec4b81a5cfe87faa22e9f1d170` |
| S/T/R/Q routing | `main/methods/content/routing.py` | `37bf9eac26f85ff667d99dc23678486d9e7ee2962c53547211de18a4d4f3a97a` |
| semantic observation | `main/methods/content/saliency.py` | `07ff1e94fea816333269ca77a3fc89ce54463e92d31e0ce067326b63a82578dc` |
| semantic runtime | `main/methods/content/prompt_saliency_runtime.py` | `47dcd16391a46142dafd8058a414866d672b12b92fb33d2e5093bbe24eeba1b0` |
| texture observation | `main/methods/content/texture.py` | `584d3f6ce24d6a86bacc2f5a46f7a3d69cc2362133c79aa0c4ade5df6b8e2122` |
| response observation | `main/methods/content/latent_response.py` | `947af3114806c50984123b6f6b475ad9de753ea007b7675d4067619b9711f736` |
| sensitivity observation | `main/methods/content/local_sensitivity.py` | `de0eee215e1fe77ba7559c99a7fed7747d09d22da40328690516e7ddf4316331` |
| reference P95 rule | `experiments/protocol/content_routing_reference_quantile.py` | `a9f1d407b08e3ba59a7354a3b804048e5ab823350230f572072400295ae538fd` |
| historical runtime config | `configs/model_sd35.yaml` | `dabebea3fa5c9c06fdc880f093debec6913bf5ce4da31f00be51578bfe2e1670` |
| direct Q/K relation | `main/methods/geometry/differentiable_attention.py` | `6c48f69e005b2c3f450de1ec2531910b9f076d25a60e03bee1ac2db61ee138b3` |
| Q/K synchronization write | `main/methods/geometry/sync_update.py` | `1590ac04e9bcdbc265e62383469808a06cefbd68457903e86a63afbc557863cc` |
| affine estimation and rectification | `main/methods/geometry/attention_alignment.py` | `134fd1e32b4542c7904540093a1279b85a36908c44dc2f37f36c5ac9bae2c8c2` |
| Q/K runtime protocol | `experiments/protocol/method_runtime_config.py` | `8619aa4e4ec3e87d1b80558878ff1e91e6f6c501c2c70534dd59b59df16a2da9` |
| image-only Q/K extraction source | `experiments/runners/semantic_watermark_runtime.py` | `87ec13fc86b843289505cb855f232fd6a6cea494265c2ab16370ba1295866424` |
| keyed PRG | `main/core/keyed_prg.py` | `9fd5f24023862afef4743dc6aca1cf0b4401f1ffb8d848c4d52f86616945cea2` |
| normal quantile table | `main/core/normal_quantile_table.py` | `e98c2a0d76080d5080b8d22eb20cb7559c8291a668cf810aa508d89bc7b8776e` |

FlowHF 明确把旧 LF/HF 双载体和 Q/K coupled route 记为 historical non-passing route。因此这些文件只提供具体候选语义，不能向 CEG-WM 传递成功、阈值或论文证据。

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
- `content_detector` 只消费 `s_hf`（当前 HF-only `D_M`），组合候选晋升后才同时
  消费 `s_lf`；它不读取 carrier direction、callback latent 或写入记录；
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
average、平局顺序、callback 18 与 `3/250` 都已冻结为当前 CEG-WM 候选语义；
其中参数来源具有历史 provenance，但真实模型/runtime 可执行性与科学效果仍未验证。

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

该 raw score 仍绑定 `lf_low_pass`，保留为已实现的历史候选和后续独立比较 control；
既有 8-cluster transmission diagnostic 的阴性/部分信号结果不被改写，也不能用于
拟合下面的新候选。新增候选并不使当前 readiness、正式 detector 或科学结论自动变化。

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
模板获得稳定更高的 matched score。登记候选不表示实现完成、机制有效、可晋升、已有
`tau` 或支持 FPR 声明。

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
- 随后的全新 validation 才能比较 registered/wrong-key/paired primary-null；本候选
  当前没有效果证据、阈值、FPR、candidate promotion 或论文 claim。
- `lf_low_pass` raw normalized-correlation 继续作为独立 historical/control score，
  但不得与本候选组成结果后 ensemble，也不得充当本候选失败时的静默 fallback。

## Candidates `routing_stqr` And `routing_uniform_control`

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

未晋升的 diagnostic combination 可以显式消费
`LfNullWhitenedDetectionResult`，但必须绑定 public blind RGB-to-VAE
observation、冻结的 `W` asset，以及相同 detector/config/preprocess identity；该路径
不得 fallback 到 raw `lf_low_pass`。这一诊断接线不表示 LF 或组合已经晋升；正式
`D_M` 仍为 HF-only，formal detector、`tau` 与 joint decision 语义均不改变。

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

本文已经为当前候选集合关闭 key/KDF/PRG、relation/objective、LF write/score、
routing observations、backbone/runtime、搜索、可靠性指标、回正和联合判定的实现
选择空白。registry 合计 11 个 ID：10 个待实施/晋升候选，加上
`routing_uniform_control` 这一项强制同预算禁用对照。对照只用于因果验证，不参与
方法晋升，也不把 11 个候选 ID 误写为 13 项实现职责。仍待实验决定的是明确有限候选
中的晋升结果和 calibration 数值，不是实现算法。

### Frozen Specification Values Versus Evidence Outcomes

现在已成为候选规格值的是：key encoding/KDF/PRG 与 golden bits；SD3.5 revision 和
runtime protocol；HF sparse-tail/filter/write/score 顺序与候选强度；LF
filter/write/raw score、唯一 clean-null whitening fit/matched score 与有限 `a` 集；
S/T/R/Q observations；empirical-CDF/tie/clip/table
规则与三条语义化组合函数；Q/K 层、前向、四通道、projection、聚合、subspace 和 line
search；similarity/dihedral 搜索、W/V、objective、raw reliability metrics 与
rectification；conditional recovery 控制流。

仍只能由预登记 calibration/实证晋升决定的是：哪个 `a`、哪条语义化组合函数、
`alpha_selection`、formal branch CDF、`tau`、`tau_rescue`、七个 geometry reliability
`gamma` 与 `epsilon_inlier`、各职责样本量、候选是否晋升/淘汰、runtime 是否可复现，
以及 FPR/TPR、鲁棒性、质量、成本和完整/负结果/reduced-scope outcome。文档中的历史
参数级来源值是待验证候选值，不是 CEG-WM 已验证事实。

因此，本文可以在未来用户授权后直接指导实现；它本身不授权实施。下一步顺序是：

1. 独立审计本文的完整性与历史边界；
2. 用户决定历史代码复用权处理，并授权建立 CEG-WM 可审计版本身份；
3. 以不含 `main/` 的独立 revision 进入 `method_construction_authorized`；
4. 后续 revision 严格按候选 ID 实现和迁移；
5. 实现测试与独立语义审计通过后才可进入 `method_implemented`。
