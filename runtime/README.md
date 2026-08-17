# Runtime Layer

`runtime/` 保存模型适配、HF carrier/HF detector 执行、Q/K 观测、生成与检测执行
能力，只允许依赖 `main/` 公开接口。模型后端、设备选择和推理流程放在此层；
near-threshold、几何救援和最终判定仍属于 `main/`。

当前 runtime_configuration_and_adapter 已实现：

- `runtime_sd35_flowmatch` 冻结配置的严格解析与稳定摘要；
- CPU/CUDA 设备选择及不可用设备的 fail-closed 错误；
- 后端协议、实际后端身份核验和单次初始化生命周期；
- 构造时私有锚定原始 backend 对象及精确类型，并通过
  `revalidate_execution_identity()` 公开返回不可变 canonical identity；每次调用
  复验 backend 对象/类型、冻结配置 digest、state、资源所有权及原始 READY
  session；构造期还惰性锚定 qk_observation Q/K module 的精确公开函数，并在执行前后
  复验 module/function 对象身份，identity 只公开稳定 qualified-name 字符串，不
  暴露 callable、backend 或模型私有状态；
- 任一初始化、content/QK execution 或 close 失败进入唯一 clean `FAILED`
  不变量：不再拥有 backend 资源且清空 session、session anchor 与 digest；
  公开 identity 发现任何 residual state 都 fail closed；
- 不加载模型的 mock backend 控制流测试。

content_write_and_vae 的本地 CPU/mock 路径当前已实现：

- clean/watermarked 同基础 float16 latent 的防共享配对与 callback 全序列绑定；
- callback index 18 exactly-once 内容写入、binary16 RNE 独立重放和 actual tensor /
  `delta_content_actual` / realized total-relative L2 测量；
- runtime 按 `main.content_embedder` 请求的 binary32 scale 物化并返回全部 attempt
  identity；`main` 以 nominal=limit=`3/250` 的冻结 direct hard comparison 驱动
  full-scale 接受或 binary32 最大非零可行 scale 搜索；
- generation VAE scaling/shift decode 与 detection posterior-mode encode；
- missing/duplicate/wrong callback、pair drift、非有限、overflow、写入消失和禁止
  posterior sampling 的 CPU fake/backend 失败语义。

runtime 不拥有 accept/retry/scale/final-failure 语义；它只物化、测量和执行
finite/bitwise/nonzero 完整性检查。权威 gate 是 `main` 对 row-major binary32
`realized_total_l2 <= limit_norm` 的直接比较；ratio/utilization 仅诊断，不存在
`q_budget`、`tau_actual_budget`、容差或 actual 下限。hard limit 只约束最终
combined content delta；nominal LF/HF directions 不构成 actual branch
decomposition，geometry budget 独立。

qk_observation 的本地 CPU/mock 路径当前已实现：

- 普通 `512 x 512` RGB 检测图像经同一 VAE posterior `mode()` 重建 detection
  latent，禁止 posterior sampling；
- 由 `main` 顶层公开的 key schedule 生成 schedule index 7 的公开确定性噪声，
  backend 必须重新建立冻结 20-step schedule 并调用 `scale_noise`；
- 三路空文本、无 CFG 的单次 image-only forward 期间，runtime 直接 hook 两个登记
  attention module 的真实 `to_q`/`to_k` 输出，再使用模块实际 heads 与
  `norm_q`/`norm_k` 形成 `QkLayerObservation`；
- 缺层、投影别名、重复/缺失捕获、shape/dtype/device/nonfinite、模型、scheduler、
  conditioning 和层序身份漂移全部 fail closed；
- 该入口只消费普通待检图像，不接受 generation cache、embed record、参考图或
  私有嵌入状态，也不计算 relation、可靠性或最终判定。

runtime_qualification_delivery 的本地交付代码当前提供：

- 只在 `prepare()` 时导入 diffusers、记录所选 model locator，并要求可用且已选择的 CUDA 设备
  的 `Sd35PipelineBackend`，真实连接 content_write_and_vae callback/VAE 与 qk_observation
  schedule/attention/QK 接口；
- `smoke`、`qualification` 和可选 `replay` runner；qualification 对登记 key
  重复执行并另跑一个确定性 negative-key identity control，结果 zip 区分
  runtime/resource/integrity/budget/QK/determinism failure；
- 只允许干净精确 HEAD 的 execution package builder、固定 Drive 输入输出边界的
  唯一薄 Colab Notebook，以及不加载模型的 CPU fake/安全边界测试。

上述本地路径随后已由 candidate
`8b2344756c4c247906ff0d4eab68e46a773e13f5` 的真实 SD3.5 GPU qualification 闭合
callback、actual float16、VAE、两层 Q/K 和 registered-key 重复确定性边界。
runner 的 ordinary Python 异常会形成最小 failure zip；解释器硬崩溃、进程被系统
直接杀死或存储本身不可写无法由进程内逻辑保证打包，必须按
incomplete/resource failure 交接。该 qualification 支持 `runtime_verified`，
但不是 LF/routing 晋升、完整 FPR、鲁棒性或科学证据，低 utilization 也不得在
未来实验中用于结果后筛除。
