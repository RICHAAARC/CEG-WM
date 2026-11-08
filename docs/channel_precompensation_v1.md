# 单次双通道预补偿：首候选实现与最小否定验证

本候选已获用户采纳。首次用户运行在写入前的普通／代理前向检查处停止，尚未取得
标记输出或比特恢复结果。当前交付修正一个已定位的预处理布局差异，并从已保存的
零臂继续；修复在真实 L4 上的效果尚未验证。旧测量、负结果和旧 LF/HF 路径均保留。
本轮用户明确选择：**图像质量只报告，不设质量失败门槛、不据此筛除来源。**

## 具体机制

普通 RGB 经同一个 SD3.5 VAE posterior mode 及原 shift/scale 约定得到完整特征。
密钥派生单位 Rademacher 读向量和独立量化相位；当前配置固定
`key_seed=2026110830`、`delta=1`、内区间半宽 `delta/4`。不拟合读者、特征白化、
频带或格距。接收端仅使用图像、密钥和固定公共资产；比特真值只用于离线报表，
不参与解码。

写端在同一未标记宿主上读取干净分数和设计种子重建分数。两条局部代理敏感度
分别描述这两个标量对 step-18 写入潜变量的响应。每个通道独立选择最近的目标
比特码格，精确平局选择较大格点；此后不换格点。

- **joint：**在两个线性化内区间约束下求最小 L2 扰动。用至多两个活跃约束的
  SVD 解避免显式平方条件数；保留退化、不可达、超预算和数值未解状态。
- **gradient：**共用相同分数、敏感度、目标格点、内区间和预算，从两个中心残差
  的平方损失形成一条合成梯度射线，在该射线上求最短可行步。没有可调学习率，
  也不使用额外模型反馈搜索步长。

两者都只写一次。写入位置仍是 20 步生成的 step 18 末回调，之后实际执行 step 19。
可写空间对两法相同，不限制为此前的固定 HF 模板。名义相对 L2 上限为 0.01；
量化后实际扰动单独记录，实际超预算不会被隐式缩放回预算。
原图已位于某比特目标区间时允许有意零写入，但每个来源必须同时测试两个比特。
请求非零而实际 dtype 消失的写入另标为 `quantization_collapsed`。

## 真实路径与代理的边界

1. 每个来源先完整生成未标记 RGB，捕获写入前潜变量、条件、生成器和 scheduler 状态。
2. 干净读者 VJP 从普通 RGB 出发。设计零臂先由普通 no-grad/PIL SDXL 路径重建并读取；
   然后另外运行一次可微代理重建用于求导。
3. 代理重建调用 Diffusers 0.39.0 官方 SDXL img2img `__call__.__wrapped__`，
   包含输入 VAE 随机重参数化、实际去噪步骤和输出 VAE；没有把整个攻击替换成恒等 STE。
   只有 uint8 取整边界使用直通近似。设计种子的普通重建和代理重建要求逐像素相等；
   不一致时保留两份图像和读数并停止该来源的线性化。
4. 从完整生成捕获点执行可微 SD3 末段。其量化 RGB 必须逐像素复现未标记完整生成。
   一份末段前向接受两条 RGB 余切，得到两次潜变量 VJP。
5. 每个可解写入重新执行完整同 seed 生成，核对写入前状态、scheduler、实际 dtype 差值。
   正式输出及两个种子的重建都从保存为普通 RGB 的路径重新读特征。

这些是 **代理敏感度**，不是离散 FP16/uint8 系统的精确 Jacobian。保留代理／实际标量
差值、实际扰动后的线性预测、梯度 dtype 和零值情况。FP16 梯度全零可能来自数值下溢，
不能直接当作物理通道为零。前向等价性检查失败属于实现／代理限制，不能直接宣判
该科学机制不存在。

## 固定最小名册和调用预算

模型、revision、512×512、SD3.5 20 步和 guidance 7、SDXL strength 0.3／20 步／guidance 1／
空 prompt 沿用上一轮明确配置。新来源和种子在取得任何新结果前固定：

| 来源 | 内容 | 生成 seed |
|---|---|---|
| source05_smooth_stilllife | 红苹果、白瓷盘、简单背景 | 2026110831 |
| source06_fur_knit | 金毛犬、针织靠垫 | 2026110832 |
| source07_coastal_scene | 海岸悬崖、草地和海浪 | 2026110833 |
| source08_indoor_structure | 书架、窗框、地砖和阴影 | 2026110834 |

设计重建 seed 为 `2026110840`；独立评价 seed 为 `2026110841`。求导、求解和格点选择
不接触独立 seed。完整英文 prompts 在 `configs/channel_precompensation_v1_colab.json`。

| 工作 | 全部来源可写时的次数 |
|---|---:|
| 正式完整生成：4 个零臂 + 16 个标记输出 | 20 |
| 正式普通 RGB 重建 | 40 |
| 正式公开特征读取 | 60 |
| 求导附带的完整 SDXL 重建前向 | 4 |
| 完整 SDXL 重建前向物理合计 | 44 |
| SD3 捕获点之后的辅助末段前向 | 4 |
| 潜变量 VJP | 8 |
| RGB VJP（干净读者 4 + 完整重建读者 4） | 8 |

另有代理读者前向及 ordinary/proxy 对照读取，全部独立计数；一次反向内同时经过
读者和 SDXL 的组件计数不能相加为两次独立反向。原始 component 计数和 attempted／returned
均保存。不可达写入不实际生成，但每方法 **8 个比特试次、4 个双比特来源**的分母不变。
运行中断时最后落盘计数不冒充完整物理调用总数。

## Colab 使用

入口为 `notebooks/channel_precompensation_v1_colab.ipynb`。这是**内含本次源码快照**的
未执行 notebook，不依赖尚未发布的 Git ref，也不要求上传代码 ZIP 或手动改源码版本。
在 Colab 打开此 notebook、选择 GPU、设置有模型访问权限的 `HF_TOKEN` Secret 后 Run all。
第一个代码单元仅挂载 Drive。

结果位置：

```text
/content/drive/MyDrive/CEG-WM/channel-precompensation-v1/precompensation-20261108/
  config.json
  source_snapshot.zip
  environment.json
  batch/
    summary.json
    reader.pt
    source05_smooth_stilllife/...
  measurement_summary.json
```

末尾打包格只读取结果并下载 `precompensation-20261108-results.zip`。出错后可单独运行
打包格。已有批次无论完整、失败还是中断都只读，不自动重算。此次入口不进行首次成功
样本筛选，不以读错、不可达、质量或方向响应为依据换来源。

SD3 和 SDXL 大组件分时装载；只有同一个 SD3 VAE 读者跨阶段保留。梯度阶段移除推理
offload hooks、冻结权重、把保存的激活放到 CPU。稳定冻结权重的 CPU 副本复用；临时改变
dtype 的 SDXL VAE 不使用这个缓存。SD3 末段求导时文本编码器留在 CPU。
首次 L4 运行已完成零臂与多数 RGB 反向；**完整写入流程及本次修复后的显存和主机 RAM
需求仍未验证。**

## 2026-11-08 首次运行与续跑

已读取用户回传的原始结果包。4 次零臂完整生成、8 次正式零臂重建、4 次代理重建及
12 次公开特征读取均返回；7 条 RGB VJP 返回，潜变量 VJP 为 0。四个来源均在
`gradient SDXL forward does not exactly reproduce ordinary design RGB` 停止；
source07 另有干净读者的 ordinary/proxy feature allclose 失败。没有求解或标记图。
因此两方法的 `failed: 8` 是固定名册内的工程失败，不是实测比特恢复率为零。

普通／代理未标记重建图的 PSNR 为 59.04、49.90、50.44、49.81 dB；对应密钥分数差
为 +0.01059、−0.00844、+0.01614、−0.02086。这些数值描述两条前向路径的差别，
不描述水印质量或补偿效果。原记录未保存 source07 失败处的干净特征，不能补造该误差。

Diffusers 0.39.0 的 PIL 预处理给出 channels-last 布局，Tensor 入口经拼接后给出
连续 NCHW；在本地读取原图验证了像素相同而 stride 不同。修复只把写端预处理输出
复制到普通 PIL 路径的布局，复制对输入保留恒等导数。shape、dtype 和数值必须相同；
未改变读者、模型、随机种子、码格、预算、特征 allclose 容差或逐像素检查。
布局差异可能改变半精度核的数值行为，但尚不能认定它解释了 L4 上的全部失败。
现在失败前保存特征、标量差、布局和执行策略；成功求出的 RGB 梯度也立即保存。

本次使用 `notebooks/channel_precompensation_v1_continue_colab.ipynb`，从原
`precompensation-20261108/batch` 读取配置、读者、4 个捕获点及 12 份零臂 RGB/特征。
原 batch、源码快照和环境记录不覆盖，新结果写入同级 `continuation_layout_v2/`。
旧版本没有保存梯度，所以本次重新计算 8 条 RGB VJP（含 4 次代理重建），
随后执行尚未发生的末段求导和标记写入。全部可写时新增上限：

| 工作 | 新增次数 |
|---|---:|
| 标记完整生成 | 16 |
| 正式普通 RGB 重建 | 32 |
| 正式公开特征读取 | 48 |
| 求导代理重建 | 4 |
| 完整重建前向物理合计 | 36 |
| 源末段前向／潜变量 VJP／RGB VJP | 4 / 8 / 8 |

摘要分别记录旧调用、新调用、累计调用和复用数量；原失败摘要也原样保留。
每方法 8 个比特试次、4 个双比特来源的分母不变。旧失败不能回写为成功。
此入口只恢复写入前检查点；若已有标记写入尝试，不会静默重做。已有本次续跑目录
仍只读，不自动再次运行。打包下载包含原失败批次和本次新增结果。

## 本地工程验证

在当前 checkout 中：

```bash
PYTHONPATH=src:. python -m experiments.run_channel_precompensation_v1 \
  --config configs/channel_precompensation_v1_colab.json
```

默认仅显示计划，不加载模型、不创建结果目录。真实执行必须显式添加
`--execute --output <new-directory>`；本轮交付未执行该模式。

只读检查已有结果的续跑计划：

```bash
PYTHONPATH=src:. python -m experiments.run_channel_precompensation_v1 \
  --resume-from <original-run>/batch
```

```bash
python scripts/build_channel_precompensation_notebook.py
python -m pytest -o addopts= \
  tests/unit/test_channel_precompensation.py \
  tests/unit/test_channel_precompensation_runner.py \
  tests/unit/test_channel_precompensation_notebook.py \
  tests/integration/test_channel_precompensation_runtime.py \
  tests/integration/test_channel_precompensation_provider.py -q
```

本地验证包括打包后隔离源码的默认入口、多层假模型的完整反向、实际 dtype、
普通／代理前向不一致、布局复制的数值和导数、分时装载、计数、检查点恢复和失败保存。
上述五个测试文件合计 **109 项 CPU/static/fake 检查通过**；原始结果 ZIP 的只读恢复加载
及两个 notebook 的 nbformat 验证也通过。本次未执行真实模型或 GPU。
这不是科学 PASS，也不是完整 OptMark 或多种子 EOT 复现。

## 结果判读和停止条件

逐试次分别保留：求解状态、实际预算、干净／设计／独立 seed 的固定盲读结果与裕量、
零写入／量化消失、RGB RMSE／PSNR。独立 seed 结果只用于评价。质量始终只报告；同一潜变量
预算不保证同等 RGB 失真，所以不能把更大实际失真造成的收益解释为机制创新。

- 线性预测可达，而真实 RGB 设计通道不能支持双比特写入：停止此一次校正版。
- 设计 seed 可读，独立 seed 不可读：本配置未建立所测 seed 迁移能力。
- 联合法没有优于同信息梯度对照的实际裕量／成功情况／失真代价：不支持专门预补偿贡献。

正结果也只说明值得继续。两个方法都使用每图敏感度，当前小对照不能单独证明宿主适配
是收益原因；一个独立 seed 不能证明一般随机重建稳定性。QIM、反拉梯度和区间最小范数
求解本身均不构成创新。一比特可控性不等于水印存在性、密钥归属或固定 FPR。
