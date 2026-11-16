# CEG-WM 残余平移／尺度误差诊断

[直接在 Colab 打开](https://colab.research.google.com/github/RICHAAARC/CEG-WM/blob/diagnostics/crop-bottleneck-v1/diagnostics/CropResidual-V1/CEG_WM_crop_residual_v1.ipynb)

沿用已成功的 **Python 3.12 GPU 环境**，Secrets 使用原 `HF_TOKEN`、`CEG_WM_ROOT_KEY`，运行全部单元。L4 可用；这里只加载原模型版本的 VAE，不加载或调用去噪器、文本编码器、DINO、SyncSeal，也没有生成回退。

默认输入：`MyDrive/CEG-WM/crop-bottleneck-v1/attempt01/`（只读）。
默认输出：`MyDrive/CEG-WM/crop-residual-v1/attempt01/`（独立目录）。
缺失或哈希不符时停止，不删除旧结果、不重新生成。Notebook 内嵌诊断脚本，不需要上传额外文件。

## 固定范围

8 个源 × 正负两角色 × 22 条件 = **352 条评分**。

| 条件 | 固定设置 |
|---|---|
| Oracle 重放 | 原裁剪真值及原半像素转换 |
| 预测 H 重放 | 原记录的 H，不重新估计 |
| 坐标约定对照 | Oracle 物理中心 H 直接进入原渲染器，量化半像素转换的影响 |
| 半预测残差 | Oracle 和预测 H 规范化矩阵固定等权插值，诊断实际复合误差，非盲方法 |
| x/y 平移 | 每轴 ±0.5、±2、±4 RGB 像素，共 12 点 |
| 各向同性尺度 | 绕画布中心 ±0.5%、±2%、±4%，共 6 点 |

所有条件都从保存的 `crop_direct` RGB 直接渲染一次，不对已校正图再次插值。
同一组正负图像、检测密钥、原 LF/HF 算法、17-key 顺序和加权间隔不变。352 行不可作为352个独立样本；不挑选最优条件判阳。

## 读出一致性与停止条件

只加载旧 `model_revisions.json` 指定版本的 `AutoencoderKL`，处理器配置与 Diffusers 0.39.0 的 SD3 pipeline 一致。
评分直接调用固定历史提交中的 `_lf`、`_hf`、融合和间隔函数。独立 CPU 检查与原 `score_branches_v2` 的共享观测路径逐项比较。

实际 GPU 上必须先通过全部 **32 个 Oracle／预测 H 重放检查**：

- 重采样后的 RGB 与保存的对应 RGB 逐像素一致；
- 三分支各 17 个密钥分数及最终间隔满足固定 `atol=1e-5, rtol=1e-4`；
- 全部通过才开始扰动。失败会保留差值并停止，不能把环境漂移当作敏感性。

历史阈值仍只作参照；不重新校准、搜索 H 或优化参数。

## 保存的证据

- 每条件完整 LF/HF/融合 17-key 分数、正确密钥与最大错钥的差异，以及相对本图 Oracle 的变化。
- 实际传入渲染器的 H、换算为物理像素中心后的角点 RMSE 和中心位移。
- RGB RMSE 与 VAE 表示相对 L2；另报固定内区 RGB `[64:448]²`／latent `[8:56]²` 以减轻边框解释的混淆。内区不用于评分。
- Oracle 已通过的样本标志与其失效标志；Oracle 未通过的样本仍留在全体分母。

输出包括 `summary.json`、`readout.csv`、`replay_gate.json`、`records/`、`attempts/` 和显式打印并脱敏保存的 `execution_logs/`。
每条评分独立落盘；中断后保留 RUN_ID 重跑，已完成扰动跳过，Oracle 参照会重读以校验环境。
源哈希、代码、脚本或环境改变时拒绝混合续跑。禁止并发写同一输出目录。

## 解释边界

小残差就损伤密钥证据，支持整个读出链对错位敏感；较宽的容忍范围而预测 H 仍失败，则应检查实际复合误差。半残差点帮助检验朝真值方向修正的收益，不能视作可部署的盲修正。
两种原因可同时成立。这里不能单独区分 VAE 非等变性与 HF 载体相位敏感性；各向同性尺度曲线也不覆盖任意各向异性或透视变换。
此诊断不证明扩散独有机制、正式 FPR 或新方法有效。

本地仅运行合成图像及模拟中断的 CPU 工程检查；真实 GPU 残余诊断待 Colab 执行。
