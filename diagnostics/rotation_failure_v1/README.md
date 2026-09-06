# RotationFailure-Diagnostic-V1

本分支从经本地与远端核验的 main `e12c7eae91cc36edc5d1a1d96249780a3925eccb` 创建。
授权仅覆盖独立分支/worktree、诊断实现和 CPU 验证。真实模型/GPU/Colab 诊断尚未执行，仍需单独授权。
方法修改须另建 BlindDetection-V2；阶段 5 须采用全新独立校准/测试样本并另行授权。

## 范围与输入

`input_reference.json` 固定 Drive 目录/清单引用及 100 对样本 ID、层归属。图片、模型资产和逐样本运行结果均留在 Git 外。
外部数据根目录当前为 `/home/richar/projects/CEG-WM/diagnostics/RotationFailure-Diagnostic-V1`，包含：

- 原 `manifest.json`、`drive_index.json`；
- `clean/` 和 `watermarked/`，各 100 张原始 RGB 512x512 PNG；
- `implementation/source_rows.json`：此前从 Drive 原始结果抽取的 400 条历史记录，只读取数据，不调用该目录内旧代码。

本分支不依赖 PaperFPR checkout。内容 scorer、检测器、校正器和 runtime builder 均调用本 worktree 的 main 接口。
历史正式 producer `9ec454055c74cf4ed89001387c9f700e9ba5aef0` 只标识来源结果及攻击定义，不作为分支基线。
未攻击/+10°旋转的 replay 定义位于诊断程序内，已对照历史正式攻击验证。
运行时使用历史正式阈值 `1.2657276026437319`，不加载 N_dev=256 的工程阈值，不重新校准。

## 阶段 2 冻结说明

100 对、200 张图片均成功解码；无缺失、解码失败或额外文件；五层为 4/24/24/24/24。
现有 typical 层恰好对应前三层选完后，剩余 948 个样本按旋转分数排序的第 463—486 位。
这与 manifest 中“距中位数最近的 24 个”有两个成员差异：现有名单包含 0738、0875，数值距离规则会选 0717、0992。
保持现有名单及层归属，不修改原 manifest。上述为可复现的重构解释，原生成算法仍未核验。

## 阶段 3 固定合同

- `POSTHOC_DIAGNOSTIC_ONLY`、`science_denominator=0`。
- 100 对 x 2 条件（未攻击、正式 +10°旋转）x 2 侧 = 400 条图像条件记录；配对表 200 条。
- 固定 detector/key/public assets/preprocessing/threshold；严格 `score > tau`。
- 原始 1000 对正式结果不改写，诊断集不能进入论文主分母。
- 记录 reference/pre/SyncSeal post/oracle post、阈值裕量、预测 H、角点 RMSE/最大误差、匹配负样本与全部错误。
- 生产 route/decision/score 独立保存。直接阳性后强制计算的 post 明确标记，不改变生产判定。
- oracle 与正负配对差只用于诊断；几何仅恢复坐标，不能投票阳性。
- 原始正式行的 replay 比较预设 abs_tol=1e-5、rel_tol=1e-5；不匹配需解释，不能按结果重试。
- 所有原因归类保持 UNADJUDICATED；不在看真实数据前指定主导瓶颈或阶段 4 数值标准。

## oracle 与失败记录

正式攻击使用 bicubic、reflect padding、center crop；校正复用 main 的 bilinear perspective 黑色填充采样器。
`H_truth_pixel_centers_normalized` 用于像素中心坐标误差；`H_oracle_sampler_observed_to_canonical` 适配 Pillow 边界坐标采样。
二者通过 `T(-0.5) D T(0.5)` 区分，避免半像素约定混淆。这是 oracle 的诊断适配，不改生产方法。
oracle 不能恢复裁失内容或抵消插值损失。有效支持比例和 RGB MAE 仅供解释，不用于遮罩内容评分。

每条记录 create-only；中断后同目录续写，所有已写行（包括失败）均跳过。最终产物为逐行 JSON、paired_differences.json、summary.json。
生产方法不完整与诊断步骤失败分别记录，不把运行错误算成阴性。注入测试 backend 明确标记 INJECTED_TEST_BACKEND。

## 本次验证

- 9 项 CPU unittest 全部通过：实际旋转系数、四个合成点的逆变换、错误方向反例、identity、坐标约定、oracle 决策隔离、失败保留和中断后固定 400 行。
- 与历史正式攻击比较：2 张 CPU 合成图片 x 2 条件，4/4 像素一致。
- main runtime builder 导入通过；未调用 builder、未加载模型。
- 外部输入复核 200/200 可解码，400 条来源记录身份完整。

在本目录运行 CPU 测试：

```bash
/home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python -m unittest -v test_diagnostic
```

输入审计（output 须为不存在的新文件）：

```bash
python diagnostic.py --root /path/to/external/diagnostic-data audit-inputs --output /path/to/new/input-audit.json
```

后续真实诊断接口，仅在另行授权后运行：

```bash
python diagnostic.py --root /path/to/external/diagnostic-data run-diagnostic \
  --runtime-root /path/to/external/runtime-assets \
  --output /path/to/external/diagnostic-output \
  --execute-authorized-diagnostic
```

使用原有 CEG_WM_ROOT_KEY、HF_TOKEN 环境变量，不写入日志。builder 可能加载生成 pipeline，但本程序不调用生成。
运行参数不能改阈值、样本数或攻击条件；数据/运行资产/输出路径拒绝位于本 Git worktree 内。
CPU 检查不代表真实模型诊断通过；阶段 3 的最终通过还需完整真实记录、oracle 正确性和原因审议。

真实预检入口已单独准备：`preflight.py`、`run_preflight.sh`；范围与执行前置见 `PREFLIGHT_SCOPE.md`。
新增预检检查后合计 13 项 CPU 测试通过，shell 语法检查通过。真实预检仍未启动，不能自动进入 100 对诊断。
