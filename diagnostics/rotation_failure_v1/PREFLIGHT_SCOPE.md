# 阶段 3 真实预检入口与待授权范围

当前状态：ENTRY_PREPARED / REAL_PREFLIGHT_NOT_STARTED。
当前授权仅为入口准备与范围冻结；尚不包含真实 GPU 预检、100 对诊断、方法修改或新正式实验。

## 固定范围

- 环境：Google Colab GPU runtime，不限制 GPU 型号或显存规格。硬件信息仅供记录，不作为预检阻断项。
  沿用现有 CUDA runtime，不添加 CPU 替代路线；若实际执行发生资源或依赖错误，保留实际错误记录。
- 最长 30 分钟，包含模型加载；TERM 后最多 30 秒强制退出。无自动重试，不购买额度或资源。
- 输入：seed=20260906 的 1 张固定 512x512 RGB 合成图，未攻击和固定 +10°旋转共 2 条记录。
- science_denominator=0，diagnostic_pair_count=0；不读取 100 对、旧 N=1000、校准集或新测试集。
- 不调用生成或嵌入；只使用 main 的检测、内容评分、SyncSeal 几何及同采样器 oracle 诊断。
- runtime 资产目录：`/content/rotation-diagnostic-runtime`。content model 与 public assets 沿用 main；HF 缓存按执行环境配置，不自动迁移已有实验数据。
- 输出：`/content/drive/MyDrive/CEG-WM/RotationFailure-Diagnostic-V1/preflight-v1`。
  新目录 create-only，已存在则停止；不会覆盖源目录清单、图片或旧结果。
- CEG_WM_ROOT_KEY、HF_TOKEN 从环境读取；不打印或写入 notebook/结果。

## 通过与失败

通过要求：真实 runtime 构造成功；两条件 reference/pre/oracle 分数有限；生产 detector 方法完成；
几何调用完成，合法 H 对应的 post 分数可用。UNSUPPORTED 几何、阴性分数或低几何质量本身不是性能门槛。
拒绝 ERROR、缺失分数、合法 H 不可用和运行中断。

产物：started.json、两条条件记录、result.json。合成数据/注入 backend 标识与真实 runtime 标识分开。
超时或进程被杀可能仅留下 started.json 与部分记录；没有 result.json 不得称为通过。
失败或超时后的重试须先审议原因并另行授权，不能通过换目录跳过失败。

## 执行前准备

本提交仅在本地，未推送。Colab 代码交付方式仍须明确；不能假定 GitHub 已有此提交。
先交付并核对诊断分支提交，再按仓库既有依赖准备环境；入口不会自动安装新依赖。
必须先挂载 Drive。若后续制作 Colab notebook，首个独立代码单元必须恰为：

```python
from google.colab import drive
drive.mount('/content/drive')
```

在已获真实预检授权且准备完成的 Colab 终端执行：

```bash
bash diagnostics/rotation_failure_v1/run_preflight.sh --execute-authorized-preflight
```

run_preflight.sh 使用上述默认目录和硬超时。环境变量仅提供经审议的路径覆盖方式；本次范围仍以默认路径为准。

## 独立授权

下一步可批准：代码交付方式、上述时限/输入/输出范围内的一次真实预检。
预检通过只证明阶段 3 运行链路，不触发 100 对诊断。完整诊断执行仍需下一次明确授权。
