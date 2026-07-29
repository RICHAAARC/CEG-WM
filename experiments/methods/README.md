# Experiment Method Adapters

`ceg_wm.py` 以薄适配器把 `main/` 与 `runtime/` 的真实公开能力接到内部实验表面。
它逐项委托 key schedule、content router、LF/HF carrier、content embedder、三种
content detector、runtime Q/K observation 与同步、transform estimator、
geometry reliability、image rectifier 和 conditional recovery decision，并为每次
调用返回实际结果、结果身份及适配器配置摘要。

适配器不重写算法，不接受 reference image、embed record 或私有 embedding state，
也不写 governed records。全部项目方法符号只从顶层 `main` public surface 导入；
注册密钥、wrong-key 和 public-noise 分别调用其真实公开派生接口；Q/K 路径只消费
`runtime.RuntimeQkObservationResult`，并校验 runtime/method model revision。
四种 key schedule 操作各自冻结并记录实际 public callable；wrong-key 流显式记录
`main.derive_wrong_key_material -> main.derive_wrong_key_stream` 两步 provenance，
不能用共同 responsibility 或只记录第二步来替代。13 个 component binding 必须逐项
精确匹配 responsibility、public callable 与 result identity field；配置或结果身份
缺失、伪造或漂移时 fail closed。

该表面只提供内部实验组件调用能力，不是 runner、baseline、GPU qualification 或
科学效果证据。
