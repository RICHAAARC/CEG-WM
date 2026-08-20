# Attack Transformations

`geometric.py` 实现冻结的 `identity`、`crop`、`scale`、`rotation` 和至少两个
活动变换的 `crop_scale_rotation`。每个攻击都绑定
`AnalysisUnitIdentity`、源图像摘要、攻击配置摘要和攻击 registry 摘要。

几何攻击使用确定性的 output-to-input affine：

- RGB8 输入和输出尺寸保持不变；
- bilinear interpolation、zero padding、`align_corners=true`；
- 输出执行 `[0,1]` clamp、乘 255、floor 后转回 uint8；
- crop、scale、rotation 的范围由
  `configs/experiments/internal_execution_components.json` 冻结并进入摘要。

`AttackArtifact` 与 `GeometricAttackSpec` 在构造和每次 public apply 前复用同一组
无副作用 revalidator：前者按当前 RGB8 像素重算摘要，后者按当前 attack ID、
全部参数边界与活动参数规则重算配置摘要。构造后 tensor 原地修改或字段/digest
漂移必须在 affine/grid 计算前失败。

攻击只依赖 `experiments/protocol/`，不导入项目方法、runtime、metrics、runner 或
治理代码，不读取任何方法私有状态，也不写 governed records。当前未实现非几何攻击。

`contrastive_lf_branch_attribution.py` 另实现 Stage-A 冻结的 identity、Pillow
12.3.0 RGB8 JPEG Q70、Gaussian blur sigma1.0 与 public-seed Gaussian noise
sigma0.01。JPEG Q70/Q90 capability golden 在 real-model preflight 前 fail closed；
noise 由 `(protocol_id, attack_id, source_cluster_id, generation_seed)` 的 stable-JSON
SHA-256 经 counter blocks、open-interval binary64 uniforms 和固定 Box-Muller 顺序生成，
随后一次 cast binary32、按 `0.01*255` 加和、clamp 与 ties-to-even RGB8 roundtrip。
攻击只消费当前 RGB8 与公开 cluster/seed provenance，不读取 method state。
