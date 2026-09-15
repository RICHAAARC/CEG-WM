# Latent-Landmark-Sync-V1：单一首候选与 M0

## 当前结论与授权

本轮用户授权已覆盖实施，随后明确改为 **M0 + Notebook 静态检查后提交推送，不运行 Notebook/M1**。此前两份审计文档的仅文档边界是历史记录，不能阻断当前实施。原有文档保留。本轮没有真实图像输出，M1 的 4 来源、24 图均为 `NOT_EXECUTED`，不能判断载体是否存活或建议直接执行 M2。

实现使用 main 实际 `ContentISSInjectionCallback`：同源 plain 的 host LF 分数决定冻结 ISS beta，content-only 和三个候选使用同一个 beta 和 seed；在其 step 18 返回的内容 latent 上加同步。旧设计引用的 `ContentAdaptiveInjectionCallback` 并非当前 main 实际 writer，故未回退旧内容语义。main SyncSeal 对同源 content-only 最终 RGB 应用 0.75 残差。每源 6 个输出，其中 SyncSeal 复用 content-only 图而不产生额外独立来源。

## 输出前固定的机制

- 坐标：整数像素中心，8 个公开布局坐标按 `63*(x,y)`；最近整数使用 `floor(center+0.5)`，7×7 支持。`r<3.5` 余弦窗 `w=(1+cos(pi*r/3.5))/2`，`g=exp(-r²/2)`；核 `w*(g-sum(w*g)/sum(w))` 后单位 L2。这样去 DC 同时保留边缘衰减。亚像素中心直接采样解析函数，没有先整数写入再插值。全 16×64×64 张量统一总 RMS，非逐点预算。
- 身份：独立域 `CEG-WM/Latent-Landmark-Sync-V1/channel-codes\0`，key 前缀 8 字节大端长度。每 row-major 元素由 SHA256(prefix + 4字节大端计数器) 前 8 字节映到 [-1,1]。NumPy float64 reduced QR，R 对角统一为正；小于 1e-12 拒绝退化。码正交仅是写入空间事实。
- 读出：空间 9×9 零填充均值高通；同样高通整数中心参考核，得到 15×15 模板。按身份投影后相关，以该 15×15 观测局部能量平方根归一化。能量 ≤1e-24 返回 0。无按图白化。只接受完整 15×15 支持，最外 7 格不搜索。滤波零填充不是尚未执行的旋转攻击。
- 峰：原响应 3×3 局部极大 mask 后 NMS 半径 3（含等号），每身份最多 3 个正峰；相等按 y、x 排序。3×3 中央差分二次拟合，Hessian 严格负定且条件数 <1e6，偏移每轴 ≤0.75 才接受，否则保留整数峰并标注；边界同样标注整数回退。0.75 是允许离散非抛物线峰修正的单一界限；M0 初版 0.5 在理想分数核的一个对角峰上不必要回退，故在真实输出前修复，未按真实结果调参。
- 对应：身份不同、提案点距及公开点距均 ≥8；由点对估刚体 observed→canonical H。角度 ±20°、无反射、尺度 1。一致性半径 0.5；跨身份观测距离 ≤0.75 判为同一点冲突。最多 24 峰使用穷举有剪枝的一对一赋值，目标为身份数、响应总和、负残差平方和的词典序。等权 Procrustes 拟合及重新赋值固定 2 次；复查最少 4 身份、角度、0.35×64 的最大间距。
- 次优：所有提案以相同目标排序，8 个公开中心上的最大变换差 ≤0.25 latent 像素视为同一 H。最佳与第一个不同 H 的 `身份数+平均响应` 差必须 >0；单一 H 的 margin=2。这样重复提案不会制造歧义，平票拒绝。范围/覆盖/错误匹配不以内容分数修复。
- RGB 共轭：约定 A 为 `x_rgb=8*x_latent+3.5`（y 同）；`H_rgb=A H_latent A^-1`，再以 N=`2*x/511-1` 共轭为 main 公共 H。仅是固定 cell-center 坐标约定，**不声称尺寸比证明 VAE 实际物理中心**。M0 检查正负旋转、平移和真实 PIL 回正采样方向；物理中心偏差留给未来真实盲点位误差。

## M1 开发规则（尚未执行）

固定清单与预算见 `configs/latent_landmark_sync_v1/candidate.json`，4 个新 prompt/seed 在真实输出前确定，不能替换失败源。三个 RMS 共享单一阈值：12 个 correct-key 非候选查询（plain/content-only/main SyncSeal）与全部 24 图的固定外部 wrong-key 查询，取所有正峰最大响应与 0 的最大值作为严格峰阈值。外部 wrong key 按另一个同步域、root key 和来源 ID 的 SHA256 派生，distinct query 不增加独立来源数。内部 16 wrong-key 内容分数仍由原 main scorer 自行计算。

这个共享阈值是保守、单参数、可复算的开发规则；它保证开发负对照没有可接受峰，可能同时抹去弱真峰。歧义阈值固定 0，不按幅度分别拟合；仍完整保留阈值前全图峰诊断。负查询不全则不能冻结阈值。所有幅度按升序选最小同时满足 4/4 的一项：盲 H 接受、至少 4 身份点误差 ≤0.125 latent 像素、固定 3×3 RGB 控制点 RMSE≤1 像素、质量预算、content-only 原本 pre>tau 的不得丢失。0.125 是与 1 RGB 像素开发定位精度一致的预定正确点判据，非已知内容容差。

PSNR/LPIPS 均相对同源 plain，比较 main SyncSeal：PSNR 最多低 0.5dB；LPIPS 最多 main 的1.1倍+1e-4。LPIPS 为 alex 默认 learned v0.1、RGB[-1,1]；残差记录全图和8个固定57×57 RGB窗口的 RMS/max。完全相同的 PSNR 序列化为 null + identical=true。内容参考阈值复制历史原始文件，tau=1.5545224858704942，SHA256 `88a52daa3398092df4ffa76117692abf9df8fb63f0b775d4a64a54790c1feaf8`；来源 `alive/paper-main-reconstruction-v2-results/02_raw_records/main/threshold.json`。仅用于机制比较，不继承 FPR。

独立候选 detect API 仅接受当前图、key、公开资产与阈值，内部复用原 main 内容 scorer（带 `.value` 的统计对象）并保持 pre>tau 早退且不调用 reader；否则内部从当前图调用一次 reader，再仅一次 H 回正+同 scorer/key/tau post。无 H 或歧义不回正。不接受外传 geometry，故不能由调用方注入真 H。不会把候选塞进 `BlindProductionAssets.geometry_backend`。M1 只需 clean pre 内容保持；不把额外 reader 调用当作生产实际几何调用，也不自动增加强制 post 实验。

## 检查与运行

M0：17 个 CPU 测试通过，含核/正交码/RMS、正负旋转和移动、漏点/错误身份/重复位置/退化/无信号/歧义、合成核盲读、RGB像素中心及PIL方向、callback顺序、真实内容分数对象协议、4×6分母与共享阈值、mock完整实验入口。mock输出仅在 pytest 临时目录，不进入 M1 科学记录。源码默认 main 不变。

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 python -m pytest tests/test_latent_landmark_sync.py -q
python -m experiments.run_latent_landmark_sync_v1 --mode initialize --output /path/to/m1-output
# 本轮不执行下条；未来用户选择真实运行时后使用
python -m experiments.run_latent_landmark_sync_v1 --mode run --output /path/to/m1-output --runtime-root /path/to/runtime
```

`initialize` 固定24行并输出每幅度4源未执行状态；运行失败保留 ERROR 和已存图/盲峰，不自动重试。`summarize` 对缺失也保留逐源状态和可计算指标，全集不完整时不选择 RMS。实际 timesteps 与 callback 序列写入日志；“写后一次更新”的证据来自标准 diffusers step-end callback 语义及 0..19 序列，未声称直接 instrument scheduler.step。

## 同版本双审后的必要修复

A2/A3独立复核、A4综合后，保留物理报告用 `public_h=N H_rgb N^-1`，另在候选 PIL 采样边界使用 `N T(+0.5) H_rgb T(-0.5) N^-1`。main 回正器直接传系数给 PIL，PIL 使用半整数采样中心；只测平移看不出偏差，新增 ±10°和90°围绕单像素脉冲的实际 PIL 测试，适配后中心保持，未适配版本出现偏移。没有改变 main 采样或估计的物理 H。

correct-key reader 完成即原子落盘，再执行外部 wrong-key 查询，防止后续硬中断丢失已有盲峰。全部 JSON 用同目录临时文件加 replace 写入；不引入重试。execution.json 记录源码 exact 与关键文件 SHA256。`negative_evaluated.json` 对既有峰生成 36 项共享阈值后判读（12 非候选正确 key、24 外部 wrong-key），不新增模型调用。Notebook 默认 run 供用户运行全部单元；本轮静态验证不执行。

Notebook 仅 JSON/代码 AST 静态校验，所有 code execution_count=null、outputs=[]，未顶到底执行。使用已推送实现 exact 的入口，与 Notebook 自身承载提交区别记录。WSL 为 torch2.5.1+cpu、CUDA不可用；本地无已配置可用真实模型环境。**本轮未执行的直接原因是用户最新明确停止真实运行，不能把未执行判为机制失败。**
