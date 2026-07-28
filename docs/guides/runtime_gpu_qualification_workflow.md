# Runtime And GPU Qualification Workflow

## Purpose And Boundary

本指南指导 Codex 从当前 `method_implemented` 状态建设真实 runtime、优先完成本地
CPU 验证、在必须使用 GPU 时停止并通知用户，最后依据 Colab 结果申请进入
`runtime_verified`。

本指南强调有限、可执行的方法推进：

- 先实现真实 runtime 业务路径，再补对应测试和最小记录；
- 能在本地 CPU/mock 完成的工作不占用 Colab；
- 只有真实 SD3.5、actual dtype、VAE 和 Q/K 等必须依赖 GPU 的检查才交给 Colab；
- Notebook 只是固定运行入口，不承载方法、runtime 或判定逻辑；
- Google Drive 只保存 Notebook 副本、执行包和小型结果压缩包；
- 模型权重、Hugging Face cache、pip cache 和临时张量只放 Colab 临时磁盘，
  每个新会话重新下载，不保存到 Google Drive；
- 不要求对下载的模型权重逐文件计算或核验 hash；
- 只要求使用候选规格登记的模型 ID/revision 和关键运行参数，不把旧环境快照、
  GPU 型号、驱动或 CUDA minor 的逐项一致设为硬门。

本指南本身不授权修改 runtime、运行 GPU 或迁移阶段。开始 runtime 构建、通知用户
运行 Colab、进入 `runtime_verified` 仍是三个独立权限。

## Current Starting Point

开始 runtime 工作前确认：

- `project_stage: method_implemented`；
- `implementation_status: implemented`；
- 13 项职责、27 个 CPU/synthetic 行为节点和 method readiness 仍有效；
- 正式 detector 仍为 HF-only；
- LF/routing/组合尚未实验晋升，`full_ceg_wm_eligible=false`；
- 当前任务只建设和验证 runtime，不开始候选选择、正式 calibration、攻击矩阵、
  baseline 或论文实验。

## Reference Colab Environments

以下 2026-07-07 Colab 快照作为优先设计参考，不作为硬门禁：

| snapshot | GPU | GPU memory | observed base |
| --- | --- | ---: | --- |
| `colab_env_snapshot_20260707T161828Z_T4.zip` | Tesla T4 | 15360 MiB | standard |
| `colab_env_snapshot_20260707T162014Z_T4 high RAM.zip` | Tesla T4 | 15360 MiB | high-RAM session |
| `colab_env_snapshot_20260707T195908Z_L4.zip` | NVIDIA L4 | 23034 MiB | standard |

三份快照共同记录：

```text
Python 3.12.13
torch 2.11.0+cu128
torchvision 0.26.0+cu128
torchaudio 2.11.0+cu128
diffusers 0.38.0
transformers 5.12.1
accelerate 1.14.0
huggingface-hub 1.20.1
safetensors 0.8.0
numpy 2.0.2
Pillow 11.3.0
CUDA runtime 12.8
```

`StableDiffusion3Pipeline` 和 `StableDiffusion3Img2ImgPipeline` 在三份快照中均可导入。
这只能证明当时的导入环境可用，不证明 SD3.5 模型加载、显存足够或 CEG-WM runtime
已经通过。

环境设计原则：

- 优先沿用上述 Python 和项目候选依赖版本；
- 完整 qualification 优先选择 L4；T4 可先运行 smoke，并在资源允许时运行完整
  qualification；
- Colab 当前驱动、CUDA minor、Python patch 或 GPU 型号合理变化时先尝试受支持的
  运行路径，不因与旧快照不完全一致直接失败；
- 项目 Python 依赖若无法解析、模型 ID/revision 无法加载或关键算子不可用，应报告
  runtime/resource failure，不得静默更换模型或方法；
- 不要求保存或比较旧快照 zip 的 hash，也不要求计算模型权重 hash。

## Minimal Advancement Flow

```text
用户授权 runtime 构建
  ↓
本地实现 runtime + CPU/mock tests
  ↓
完整 CPU 三门
  ↓
中文 revision + 代码审查
  ↓
冻结一个 Colab execution package 和一个薄 Notebook
  ↓
Codex 通知用户需要 GPU
  ↓
用户在 Colab 运行 smoke
  ↓
smoke 通过后运行 qualification
  ↓
Colab 将小型结果 zip 保存到 Google Drive
  ↓
Codex 从本地映射读取并审核结果
  ↓
失败则回本地修复并生成新 revision
  ↓
通过后另行申请 runtime_verified 阶段迁移
```

默认不强制第三次完整 GPU replay。只有结果含混、独立审计提出具体问题，或需要确认
Colab 重启重放时，才使用同一 Notebook 的 `replay` profile。

## Local CPU-First Construction

### Batch 1: Runtime Configuration And Adapter Skeleton

- 固定 SD3.5 model ID/revision、pipeline、scheduler、steps、guidance、resolution；
- 固定 callback index、VAE 路径、latent/template dtype 和 Q/K 登记层；
- runtime 只依赖 `main/` 公开接口；
- 建立配置解析、设备选择和 fail-closed 错误；
- 使用 mock backend 完成导入和控制流测试；
- 不下载模型，不调用 GPU。

### Batch 2: Content Write And VAE Path

- 建立 clean/watermarked 同基础 latent 配对；
- 在 callback index 18 物化 `main` 返回的内容 delta；
- 计算 `delta_content_actual` 和 realized combined total norm/relative L2；
- generation decode 使用冻结 VAE scaling/shift；
- detection encode 使用 VAE posterior mode，不采样；
- CPU/mock 覆盖 callback 未触发、重复/错误 index、dtype 写入消失、非有限量、
  overflow 和 deterministic binary16 replay；
- 未预登记 actual-dtype budget acceptance rule 前只返回 realized 测量与
  `budget_acceptance_status=not_evaluated`，不得由 runtime 声称预算合格。

### Batch 3: Q/K Observation Path

- 从普通待检图像重新建立检测 latent；
- 使用冻结检测 schedule、公开确定性噪声和登记 Q/K 层；
- runtime 只返回方法需要的 observation；
- 不读取生成缓存、embed record、参考图或私有嵌入状态；
- CPU/mock 覆盖缺层、重复 hook、shape/dtype 错误和非有限量；
- 真实 Q/K 内容留到 Colab GPU 验证。

### Batch 4: Runner, Result Zip And Thin Notebook

- 建立一个可从命令行运行的 runtime qualification runner；
- runner 提供 `smoke`、`qualification`，必要时提供 `replay`；
- runner 自己生成结果文件、失败状态和 zip，Notebook 不手写结果；
- 建立最小 execution package，供没有 Git remote 的 Colab 会话使用；
- 创建唯一 `notebooks/colab/runtime_qualification.ipynb`；
- Notebook boundary、smoke/integration 选择和 execution package 在本地先检查。

### Local Tests

真实 backend、GPU、network、large model、smoke 和 integration 不得进入默认 pytest。
每个本地批次先运行最小定向 CPU/mock 测试；完成时按项目合同且只运行一个注册
validation profile：

```bash
conda run -n CEG-WM python governance/tools/run_validation_profile.py method
conda run -n CEG-WM python governance/tools/run_validation_profile.py full
```

research/runtime 代码、对应测试及普通非语义引用文档（包括字段登记和测试清单）
使用 `method`；只有修改 registered design、readiness、stage/research-state、
pytest selection 或实际跨治理平面时才使用 `full`。不得把两个 profile 都当作同一
完成门重复运行。

CPU 通过只表示可以申请 Colab/GPU 检查，不表示 runtime 已验证。

## Runtime Method Flow

### Generation

```text
Prompt / negative Prompt / seed
  ↓
固定 SD3.5 + FlowMatch
  ↓
callback 写入前 latent 和必要 observation
  ↓
main 计算 routing、LF/HF directions 和共同总预算 delta
  ↓
runtime 在 callback index 18 以 actual dtype 写入
  ↓
runtime 返回实际写入量
  ↓
main 判定预算是否合格（仅在另行预登记 acceptance rule 后；
当前 Batch 2 状态为 not_evaluated）
  ↓
VAE 解码普通 RGB 图像
```

### Detection And Q/K

```text
普通待检 RGB 图像 + 检测 key
  ↓
VAE posterior mode
  ├── final-image latent → content score
  └── 检测 schedule → 真实 Q/K observation → geometry chain
```

runtime 不选择 `a`、组合函数、`tau`、`tau_rescue` 或 geometry reliability；不产生
最终阳性。Q/K 只服务同步和回正，不是第三种水印票。

## Codex Notification Before GPU

本地实现、CPU 三门和代码审查通过后，Codex 必须停止并通知用户：

```text
GPU_REQUIRED
runtime_candidate_revision: <revision>
execution_package: <Google Drive path or local file for upload>
notebook: runtime_qualification.ipynb
profile: smoke | qualification | replay
expected_result_directory: <Google Drive path>
expected_result_name: <zip name>
recommended_gpu: L4 preferred; T4 accepted for smoke and qualification if resources allow
```

通知中同时说明：

- 本次只验证 runtime，不做 LF/routing 候选选择；
- 模型和缓存每个 Colab 会话重新下载到 `/content`；
- 不要把 Hugging Face cache、pip cache、模型权重或临时 tensor 指向 Google Drive；
- 如果 OOM、模型下载失败或检查失败，保存失败结果 zip，不修改 Notebook、不换模型。

用户未确认 Colab 完成前，Codex 不得假设 GPU 已运行或推进阶段。

## Fixed Minimal Notebook

唯一 Notebook：

```text
notebooks/colab/runtime_qualification.ipynb
```

Notebook 只包含：

1. 显示任务范围和当前 profile；
2. 挂载 Google Drive；
3. 检查 GPU/CUDA 和可用磁盘；
4. 创建 Colab 临时 cache：

   ```text
   /content/ceg_wm_runtime/
   /content/hf_cache/
   /content/pip_cache/
   ```

5. 从 Drive 读取 execution package 并解包到 `/content`；
6. 安装优先参考版本或 runner 指定依赖；
7. 从 Colab Secret 读取必要访问 token；
8. 调用包内唯一 runner；
9. 把 runner 生成的结果 zip 写入 Drive；
10. 显示结果路径和简短成功/失败摘要。

Notebook 不得包含：

- 方法、runtime、Q/K hook、阈值或统计实现；
- 手写 records、通过判定或 zip 文件清单；
- 模型 fallback、自动降级或结果后改参数；
- token、root key、模型权重或私有数据；
- 把 `/root/.cache/huggingface`、`HF_HOME`、pip cache 或临时模型目录设置到 Drive。

Notebook 源 cells 首次审核后保持冻结。Colab 自动写入的 outputs 和 execution counts
不算源逻辑变化，不回写仓库；后续功能变化修改 runner/config，不复制新的 Notebook。

## Colab Storage Rules

### Temporary Colab Storage

每次会话下载并在会话结束后丢弃：

- SD3.5 模型权重；
- Hugging Face cache；
- pip/wheel cache；
- VAE、text encoder 和 Transformer 临时文件；
- latent、Q/K tensors 和中间图像；
- 解包后的 execution package 工作目录。

不把模型缓存挂载到 Google Drive，也不为了节省下次下载时间复制模型权重。

### Google Drive Storage

Windows 根目录：

```text
G:\我的云端硬盘\CEG-WM\runtime_qualification\
```

推荐结构：

```text
runtime_qualification\
├── execution_packages\
│   └── <runtime_candidate_revision>\
│       └── ceg_wm_runtime_execution_<revision>.zip
└── runs\
    └── <runtime_candidate_revision>\
        └── <run_id>\
            └── ceg_wm_runtime_qualification_<run_id>.zip
```

Notebook 工作副本可放在：

```text
G:\我的云端硬盘\Colab Notebooks\CEG-WM\runtime_qualification.ipynb
```

Drive 中只保存：

- 小型 execution package；
- 固定 Notebook 工作副本；
- smoke/qualification/replay 结果 zip；
- 用户明确要求保留的少量示例图。

不得保存模型权重、HF cache、pip cache、完整 latent、原始 Q/K tensor 或大规模生成
数据集。

本机 Windows 可以看到 `G:\我的云端硬盘`，但 WSL 不保证存在 `/mnt/g`。Codex 本地
检查结果前先确认 Windows 文件已经同步；若需要 WSL 解包，再通知用户授权临时映射
或把单个 zip 复制到未跟踪临时目录。不得在 Drive 原目录内解压和修改结果。

## Colab Validation Profiles

### Smoke

smoke 使用最小样本检查：

- 当前 GPU 能被 PyTorch 使用；
- SD3.5 指定 model ID/revision 能下载和加载；
- pipeline、scheduler 和关键模块可调用；
- callback index 18 真实触发；
- actual-dtype delta 可测；
- VAE decode 和 posterior-mode encode 可执行；
- 至少一个登记层产生真实 Q/K observation；
- 结果 zip 能写入 Drive。

T4 或 L4 都可以运行 smoke。资源不足保存 `resource_failure`，不解释为方法失败。

### Qualification

smoke 通过后检查：

- clean/watermarked 同基础 latent 配对；
- callback 不缺失、不重复；
- actual-dtype 写入没有消失，预算量可测；
- VAE 路径和检测输入边界正确；
- 登记 Q/K 层、shape、dtype 和检测端重建路径正确；
- 同 Prompt/seed/key 重复运行在合理容差内一致；
- correct-key/wrong-key 的最小 runtime 路径均能执行；
- OOM、下载失败、中断和非有限量有明确失败状态；
- runtime 不改变 detector、阈值或最终判定语义。

优先使用 L4 运行完整 qualification。T4 若能按同一方法和参数完成，也可接受；不能
为了通过而更换模型、resolution、steps、callback 或方法。

### Optional Replay

仅在以下情况使用：

- qualification 结果含混；
- 需要确认 Colab restart 后的重放；
- 独立审计指出一个具体检查需要复验。

replay 使用同一 Notebook 和 runner，不创建第三个 Notebook。

## Minimal Result Zip

结果 zip 由 runner 自动生成，至少包含：

```text
run_summary.json
environment_summary.json
runtime_checks.jsonl
failures.jsonl
console.log                 # 可选诊断附件
artifacts/                  # 少量必要示例，可选
```

`run_summary.json` 至少记录：

- profile、run ID、开始/结束时间和完成状态；
- runtime candidate revision；
- model ID 和请求的 revision；
- Python、PyTorch、diffusers、transformers、CUDA 和 GPU 名称；
- callback、actual dtype、VAE、Q/K、determinism 和结果打包的摘要；
- 成功、runtime failure、resource failure 或 incomplete。

不要求：

- 模型权重文件 hash；
- Hugging Face cache hash；
- 整个 Python 环境 hash；
- 与旧 Colab 快照逐包逐版本完全一致；
- 为了 runtime qualification 保存完整模型、缓存或大规模中间张量。

可以为结果 zip 自动生成普通 SHA-256 sidecar 以检查 Drive 传输完整性，但 sidecar
不是方法或模型身份硬门；本地也可以直接重算。

## User Completion And Local Check

用户运行完成后返回：

```text
COLAB_RUN_COMPLETE
profile: <smoke | qualification | replay>
result_zip: G:\我的云端硬盘\CEG-WM\runtime_qualification\runs\...\<result>.zip
```

失败也保存并返回 zip，不只发送截图。

Codex 本地检查重点是：

1. zip 可读取且运行已明确结束；
2. runtime candidate revision 与本地目标一致；
3. model ID/revision 和关键方法参数没有被替换；
4. callback、actual dtype、VAE 和真实 Q/K 均有结果；
5. 失败、OOM、中断没有被隐藏；
6. 没有把 runtime 检查冒充 LF/routing 晋升、FPR 或论文证据；
7. 结果包不包含模型权重、cache、token、root key 或私有嵌入状态。

不因 GPU 名称、驱动、CUDA minor、Python patch 或非关键包与旧快照不同自动拒绝。
只有实际不兼容、关键方法身份变化或必需检查失败才返修。

如果需要修代码，回到本地仓库形成新 revision，再重新运行受影响的 smoke 或
qualification；不在 Notebook 中临时补丁。

## Entering Runtime Verified

可以申请进入 `runtime_verified` 的最低条件：

- runtime 真实实现存在并通过本地 CPU 三门；
- runtime candidate revision 已完成代码审查；
- Colab smoke 通过；
- Colab qualification 证明 SD3.5、callback、actual dtype、VAE、真实 Q/K、
  基本确定性和失败语义可用；
- 结果 zip 已在本地检查，失败没有被隐藏；
- runtime 没有改变方法判定职责；
- 用户明确授权阶段迁移。

阶段迁移使用独立 revision，建议：

```text
chore: 进入 runtime_verified 运行时验证阶段
```

迁移 revision 不混入 runtime 修复、Notebook 修改、GPU 重跑、LF/routing 候选选择或
正式实验。迁移后重新运行 CPU 三门并接受独立阶段审核。

`runtime_verified` 只表示真实 runtime 边界可用，不表示 LF、routing、组合、FPR、
鲁棒性或完整 CEG-WM 科学有效。

## Stop And Notify Rules

Codex 必须停止并通知用户：

- 本地 CPU 工作完成，需要首次运行 Colab/GPU；
- Colab execution package 和 Notebook 已准备好；
- 模型下载、OOM、callback、VAE 或 Q/K 检查失败；
- 需要更换登记模型、revision、scheduler、steps、dtype 或 callback；
- qualification 完成，结果 zip 已可本地检查；
- 需要进入 `runtime_verified`；
- 工作意图扩展到 LF/routing 候选选择、calibration 或正式实验。
