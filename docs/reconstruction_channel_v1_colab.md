# 重建通道最小 Colab 验证

入口：[在 Colab 打开](https://colab.research.google.com/github/RICHAAARC/CEG-WM/blob/dev/reconstruction-channel-v1/notebooks/reconstruction_channel_v1_colab.ipynb)。
Notebook 自动获取公开仓库中指定版本的候选源码和固定配置，无需上传代码、配置或 ZIP，
无需手动填写版本。它不会调用旧 formal worker，也不读取旧水印私钥或阈值。

## 怎么运行

1. 在 Colab 打开 notebook，选择 GPU 运行时。在 Colab Secrets 中准备有模型访问权限的
   `HF_TOKEN`。沿用此前 SD3.5 notebook 使用的 Hugging Face 账号即可；如模型访问尚未获准，
   需先在该账号下取得访问权限。
2. 选择 Run all。第一个代码单元只挂载 Drive。后续自动获取 notebook 中指定版本的源码、
   安装运行依赖并读取固定配置，不使用旧 Drive 缓存代码包。获取的代码版本会记录到结果目录。
3. notebook 固定先执行第一来源，再执行其余三个来源。只有运行状态、同状态比较和完整读出
   有效时才进入后三个来源；响应为零、实际扰动量化消失或失真偏大都不会被当作筛选来源的依据。
4. 已存在完整结果只读取、不重算。已有失败或未完成的目录保留并停止，不自动重试。
   若有故障，请直接回传已有结果与报错，不修改名册、幅度或输出目录来隐藏失败。

默认结果位置：

```text
/content/drive/MyDrive/CEG-WM/reconstruction-channel-v1/measurement-20261108/
  first_source/
  remaining_sources/
  measurement_summary.json
  environment.json
```

首来源计入总共四个来源，不重复生成。最终汇总单元会在上级目录生成
`measurement-20261108-results.zip`，包含普通 RGB、原始特征、实际扰动和状态、逐来源指标及汇总。
如果运行中止，可单独运行最后的汇总/打包单元，保留失败和未运行来源；回传该 zip 即可。
环境安装在执行模型之前失败时，也请保留并回传 Colab 报错。

## 本轮固定配置

完整配置位于指定代码版本的 `configs/reconstruction_channel_v1_colab.json`，notebook 会公开显示。

| 项目 | 本轮选择与依据 |
|---|---|
| 生成模型 | `stabilityai/stable-diffusion-3.5-medium`，revision `b940f670f0eda2d07fbb75229e779da1ad11eb80`，复用现有 baseline worker 的明确版本；不宣称旧主生成已固定该版本 |
| 生成参数 | 512×512、20 步、FP16、guidance 7.0，末回调 step 18 注入后继续 step 19；7.0 对应当前 Diffusers SD3 默认值，本轮显式写入配置 |
| 探针 | 一个共享 HF-family 单位方向，公开 seed `2026110800`；名义相对 L2 为 `0.01` |
| 幅度依据 | 在观察新结果前选定的 1% 相对 L2 有限幅度测量点，便于解释归一化扰动；不继承旧 `0.012`，也不据数值小就假设线性或感知无损。实际 dtype 的幅度和偏离另行测量 |
| 读出 | 同一个固定 SD3.5 VAE 的完整 posterior mode，经其 shift/scale 处理；输入仅为普通 RGB |
| 重建模型 | `stabilityai/stable-diffusion-xl-base-1.0`，revision `462165984030d82259a11f4367a4eed129e94a7b` |
| 重建参数 | strength 0.3、20 步、guidance 1、空 prompt，沿用旧正式实验的重建设置；全局配对 seed `2026110810` |
| 内存策略 | CPU offload；本批全部生成结束后，移除 hooks 并释放 SD3.5 大组件，再加载 SDXL，保留同一 SD3.5 VAE 做读出 |

固定名册（完整英文 prompts 在配置中，不从既有图像结果筛选）：

| 顺序 | 来源 | 生成 seed |
|---|---|---|
| 1 | 素色陶瓷碗与简单背景 | 2026110801 |
| 2 | 猫毛与织物纹理 | 2026110802 |
| 3 | 山湖、树林与云层 | 2026110803 |
| 4 | 自行车、砖墙与栅栏 | 2026110804 |

这四个来源是预先选择的内容覆盖样例，不是具有统计代表性的样本。
每个来源三次完整同 seed 生成、每臂一次配对重建，因此总预算为
**12 次生成、12 次重建、24 次 VAE 特征读取**。不另外生成预检图；首来源就是名册第一项。
模型加载、卸载和特征计算不包含隐藏的生成重试。

## 环境与证据边界

挂载、子进程、`HF_TOKEN` 与 Drive 保存方式参考已用的
`paper_main_worker_colab.ipynb` 和 `paper_reconstruction_worker_colab.ipynb`。
依赖选择为 Diffusers 0.39.0、Transformers 4.57.6、Accelerate 1.1.1；前两者与本地核查的
API 对齐，Accelerate 版本也见旧 baseline notebook 配置。这不等于该组合已经通过真实 GPU 验证。
保留 Colab 自带 CUDA PyTorch，运行前检查 PyTorch ≥2.6 与 CUDA；不自动重装 PyTorch/CUDA。

本次交付只进行了 CPU/fake、语法、路径和入口验证，**未执行真实 GPU/Colab 实验**。
一个重建 seed 只能支持固定随机条件下的通道观察。四来源、单方向、单幅度不能证明共享子空间、
比特可传输、水印可检出或固定 FPR；不拟合 U/V/Γ，也不开发编码器。

如果特征完整、状态匹配但实际探针幅度归零，数据收集可完成，而有限差分仍标为无效；这是需保留的
方法负结果。端点偏离也不独立证明非线性，需和实际扰动偏离方向的能量一起解释。

Notebook 和实际运行代码分别记录 Git 版本：发布的 notebook 固定获取已经提交的候选代码，
不会跟随 main 改变。版本用于追溯，不另设文件名、字节长度或额外哈希访问门槛。
若读取此前已有的结果，本次取得的代码版本不倒推为旧结果的生产版本。

本入口固定获取的测量代码版本为
[`85d6566a571c81205063d313398bf2e459473680`](https://github.com/RICHAAARC/CEG-WM/commit/85d6566a571c81205063d313398bf2e459473680)。
该值已写在 notebook 中，用户无需填写或修改。
