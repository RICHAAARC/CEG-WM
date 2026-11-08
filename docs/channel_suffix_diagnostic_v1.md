# 源末段三次前向定位

设备恢复修复后的真实记录确认：4 个捕获点、12 份零臂观测和 8 条 RGB 梯度均被复用，
`timesteps`、`sigmas` 都已恢复 CUDA，但四源仍未通过末段 RGB 与旧零臂逐像素一致的检查。
尚无潜变量 VJP、求解或标记输出。当前数据不足以把剩余差异归因于某个算子。

用户已同意先进行本诊断。只选择既有名册第一项 `source05_smooth_stilllife`，
不按误差大小挑来源；使用 `continuation_suffix_device_v1` 中的原始捕获点和零臂。
固定顺序是 NG1（关闭求导）、NG2（关闭求导）、G（开启求导）。同一个已加载的源模型、
相同权重、输入值、布局、dtype 和当前后端设置不变，每次从捕获状态独立恢复 scheduler。
输入叶子始终保留 requires_grad 标志；模式开关控制前向是否记录计算图。
G 沿用现有 CPU saved-tensor 策略，但不调用 backward 或 VJP。

## 最小计算范围

| 工作 | 最多新增次数 |
|---|---:|
| SD3 源模型装载 | 1 |
| step-19 末段前向 / 去噪步 / VAE 解码 | 3 / 3 / 3 |
| 完整生成、SDXL 重建、公开特征读取 | 0 |
| RGB VJP、潜变量 VJP、水印写入 | 0 |

该范围不包含额外预热、重复尝试、有限差分、码格求解或门槛搜索。失败臂保留，
固定三臂不会因结果不理想而替换。载入检查点是 CPU 读取已有证据，不是模型前向。

## 记录与解释

诊断与生产 VJP 使用相同的末段前向实现；诊断只观察失配，生产的像素门槛保持不变。
保存原输入潜变量、引导后的 transformer 预测、scheduler 输出、VAE 输入、原 dtype
浮点解码、归一化浮点 RGB 和最终 uint8 RGB，以及各阶段 dtype、布局和执行模式。
比较 NG1 对 NG2、NG1 对 G 的每层值并报告首个被观察到的分歧；三臂最终 RGB
另外与保存的旧零臂比较。缺失、异常或非有限结果不能自动解释为相等。
若输入潜变量或时间步本身不一致，单独标记对照条件不一致，不能归因于求导模式。

- NG1 与 NG2 有差异：先检查重复性或隐藏状态，不能直接归因于开启求导。
- NG1 与 NG2 一致，但与旧零臂不同：检查恢复路径、重载和执行环境；这本身不能证明
  某一项状态恢复有错。
- NG1、NG2 与旧零臂一致，G 不同：支持差异与求导代理执行路径相关；该路径还包含
  autograd、张量保存和内核选择，单次 G 不证明其稳定性或唯一原因。
- 三者都与旧零臂一致：仅说明本来源本次前向一致；其他来源和梯度可靠性仍未验证。

逐像素相等不是代理梯度有效的数学必要条件，小像素误差也不能保证梯度可靠。
本诊断不改变该门槛，不设新的 PSNR 容差，不进行水印成功率、FPR 或创新性裁决。

## 用户运行入口

`notebooks/channel_suffix_diagnostic_v1_colab.ipynb` 内嵌源码，首代码单元仅挂载 Drive。
选择 GPU 并设置原有 `HF_TOKEN` 后 Run all。输入和输出分别为：

```text
/content/drive/MyDrive/CEG-WM/channel-precompensation-v1/precompensation-20261108/
  continuation_suffix_device_v1/       # 只读原检查点
  suffix_diagnostic_v1/                # 新建三臂诊断
```

结果目录在任何模型加载前创建并写入初始状态；已有目录只读，不重新执行三臂。
每臂在调用前落盘 attempted 状态，随后保存 trace、图像或失败信息。进程中断时，最后
记录的计数不能被报告为完整物理调用总数。启动 stdout/stderr 和返回码也写入 ZIP。

结果包仅包含本次诊断、必要的 source05 原输入、配置、历史摘要及本次源码／环境／日志，
原批次保持原样。历史累计计数、本次新增计数与诊断比较分别报告。

本地默认入口只显示计划，不加载模型：

```bash
PYTHONPATH=src:. python -m experiments.run_channel_suffix_diagnostic_v1 \
  --resume-from <run-root>/continuation_suffix_device_v1
```

真实执行需显式添加 `--execute --output <new-directory>`；本地交付仅作 CPU/static/fake
验证，不自动运行 Colab 或 GPU。

交付验证：原预补偿与本诊断共七个相关测试文件合计 **150 项工程检查通过**。
包括生产像素门槛保留、同源三臂、无反向、逐层失败保存、输入控制失效分类、中断未知计数、
配置属性保留、隔离源码包默认计划及 notebook 源码一致性。真实旧检查点的只读计划已通过；
真实三次 GPU 前向尚未执行，工程检查不代表诊断结论。
