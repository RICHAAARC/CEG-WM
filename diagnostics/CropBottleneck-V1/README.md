# CEG-WM 裁剪瓶颈最小诊断

这是独立诊断交付目录，没有修改 `alive/CEG-WM` 主算法或已有实验结果。

## 使用

1. [直接在 Colab 打开](https://colab.research.google.com/github/RICHAAARC/CEG-WM/blob/diagnostics/crop-bottleneck-v1/diagnostics/CropBottleneck-V1/CEG_WM_crop_bottleneck_v1.ipynb)，或自行上传本目录的 Notebook。
2. 选择 GPU 运行时，建议 A100 和高 RAM。原管线完整驻留 GPU，未加入 CPU offload 或其他运行时方法修改。
3. 在 Secrets 配置 `HF_TOKEN`、`CEG_WM_ROOT_KEY` 并授予 notebook 访问；HF 账户需有 SD3.5 模型读取权限。不要在代码单元粘贴密钥。
4. 运行全部单元并挂载 Drive。默认输出目录：`MyDrive/CEG-WM/crop-bottleneck-v1/attempt01/`。
5. 中断后保持 `RUN_ID` 不变重跑。已完成图像和评分校验哈希后跳过；失败尝试保留。不要同时开两个 notebook 写同一目录。

Notebook 内嵌完整诊断 Python，不需要单独上传脚本，不依赖把新代码推送到 GitHub。
它克隆并检出历史提交 `d9ef4adb4f8424f1102f577bb76347fbba112bd9`，复用原生成器、攻击、SyncSeal 和评分器。
不会调用原完整正式 worker，不运行 1,000 对生成、阈值校准、LPIPS 评价或其他攻击。

## 固定范围

- 正式前 8 个源，种子 `2028030000..2028030007`；每源一对 primary-null 与完整水印图。
- 512×512，原 step-18 内容写入，原 SyncSeal residual multiplier 0.75。
- 保留约 80% 面积中心裁剪：458×458，再 bicubic 恢复 512×512。
- 四路径：完整图直读、裁剪直读、原连续预测 H 校正、攻击真值 H 校正。
- 每路径原 LF/HF、注册密钥及内部 16 错钥，原融合和间隔统计量。
- 历史阈值 `1.5545224858704942` 仅作诊断参照；不重新校准，不宣称本轮达到 0.1% FPR。
- 固定 8 对、64 个路径槽位。无 H、运行失败和未执行分别记录，干净失败不排除。
- C 保存单独 post 分数，没有原流程 early-return。原裁剪流程可由 B 和 C 的分数重建；两者不可混淆。

## Oracle 的坐标约定

物理观测像素中心 `q` 到原图像素中心：`p = 27 + (q + 0.5) * 458/512 - 0.5`。
原 PIL 渲染器直接采用像素边缘采样坐标；传给该渲染器的 Oracle 矩阵按其约定解析转换，确保输出到源的实际映射为 `(x_edge - 27) * 512/458`。
物理矩阵、渲染矩阵分别落盘；预测几何误差针对物理矩阵计算。C 保持原预测 H 和原渲染调用不变。
Oracle 仅重采样被攻击图像并以黑色填充，不读取原图修补缺失区域。support 图只是可观测区域记录，不参与评分。

## 结果与恢复

- `summary.json`：64 条当前状态及固定分母汇总。
- `readout.csv`：紧凑分数表，含 own/wrong、LF/HF 和几何误差。
- `protocol.json`：名册、种子、固定攻击、密钥摘要（无密钥）、源提交与诊断脚本哈希。
- `model_revisions.json`、`environment.json`、`syncseal_identity.json`：本次公开模型 SHA、软件版本、GPU 型号及 SyncSeal 文件哈希。
- `units/`：原始成对 PNG、各路径输入、有效区域图、全部分支/错钥分数、H 与各次尝试状态。
- `last_failure.json`：异常类别和脱敏 traceback。进程硬中断时最后一次尝试会保留 STARTED，不能据此猜测 OOM。

历史模型权重版本未确认。本轮按首次解析出的公开模型 SHA 固定重生成；不得把新 PNG 或新分数称为逐字节历史复现。
恢复时协议、脚本、密钥摘要、依赖、GPU 型号、模型版本不一致会拒绝混用；需要新 RUN_ID。
这避免修改实验定义后沿用部分旧结果。首次模型下载在 Colab 本地缓存，重置运行时后需要重新下载，但沿用已冻结的模型 SHA。

## 判断边界

先检查 A；D 好而 C 差指向同步，C/D 都差才有理由进一步区分观测与载体。
Oracle 失败不等于水印信息论不可恢复。8 源只是机制筛查；内部 16 错钥不是独立非所有者密钥实验。
入口不会自动启动有效区域读法、传播探针、参数搜索或新项目。

本地仅做 CPU 工程检查：Notebook 格式/内嵌一致性、解析坐标、真实 PIL 合成坐标图、缺失区不可恢复、续跑、失败分母及秘密脱敏。
真实 SD3.5 / SyncSeal / CUDA 推理与科学结果尚未运行。
