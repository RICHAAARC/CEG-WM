# Runtime And GPU Qualification Workflow

## Purpose And Boundary

本指南记录从 `method_implemented` 建设真实 runtime、优先完成本地 CPU 验证、
在必须使用 GPU 时停止并通知用户，最后依据 Colab 结果进入 `runtime_verified`
的完整流程。当前项目已完成该流程；一般门序继续适用于未来 runtime 身份变化后的
重新 qualification。

本指南强调有限、可执行的方法推进：

- 先实现真实 runtime 业务路径，再补对应测试和最小记录；
- 能在本地 CPU/mock 完成的工作不占用 Colab；
- 只有真实 SD3.5、actual dtype、VAE 和 Q/K 等必须依赖 GPU 的检查才交给 Colab；
- Notebook 只是固定运行入口，不承载方法、runtime 或判定逻辑；
- Google Drive 只保存 Notebook 副本、独立可信 bootstrap、执行包和小型结果压缩包；
- 模型权重、Hugging Face cache、pip cache 和临时张量只放 Colab 临时磁盘，
  每个新会话重新下载，不保存到 Google Drive；
- 不要求对下载的模型权重逐文件计算或核验 hash；
- 只要求使用候选规格登记的模型 ID/revision 和关键运行参数，不把旧环境快照、
  GPU 型号、驱动或 CUDA minor 的逐项一致设为硬门。

本指南本身不授权修改 runtime、运行 GPU 或迁移阶段。开始 runtime 构建、通知用户
运行 Colab、进入 `runtime_verified` 始终是三个独立权限；当前三项权限已分别使用，
不自动授权再次运行或后续实验。

## Current Starting Point

当前已闭合的 runtime 检查点为：

- `project_stage: runtime_verified`；
- `implementation_status: implemented`；
- 13 项职责、27 个 CPU/synthetic 行为节点和 method readiness 仍有效；
- runtime candidate
  `8b2344756c4c247906ff0d4eab68e46a773e13f5` 的 execution package SHA-256 为
  `8290abeed79931eb7208ac9ca280f1ea401f4725abfead35f12617a0ef54dd38`；
- qualification run `20260729T110628Z` 为 `passed`，result ZIP SHA-256 为
  `d9b7d91d41cc963098c077268445ad80e9994c809227ca2f68615a37ac93ac37`；
- 正式 detector 仍为 HF-only；
- LF/routing/组合尚未实验晋升，`full_ceg_wm_eligible=false`；
- `negative_identity` 只证明 runtime/key identity control，不是 wrong-key FPR；
- 当前阶段不开始候选选择、正式 calibration、攻击矩阵、baseline 或论文实验。

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
- dependency lock 继续把 PyTorch 公共版本精确冻结为 `2.11.0`。runner 只对
  `torch` 接受完整实际版本 `2.11.0`，或接受 `2.11.0+<local>`，其中
  `<local>` 必须完整匹配
  `[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*`；实际读取的含 local label 版本原样写入
  dependency evidence。该例外不归一化公共版本、不接受 prerelease，也不适用于
  其他依赖；
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
Colab 重启重放时，才形成固定 `replay` profile 的新 Notebook revision，并重新
完成实施者、独立审计者和 gatekeeper 审核。

## Local CPU-First Construction

### runtime_configuration_and_adapter: Runtime Configuration And Adapter Skeleton

- 固定 SD3.5 model ID/revision、pipeline、scheduler、steps、guidance、resolution；
- 固定 callback index、VAE 路径、latent/template dtype 和 Q/K 登记层；
- runtime 只依赖 `main/` 公开接口；
- 建立配置解析、设备选择和 fail-closed 错误；
- 使用 mock backend 完成导入和控制流测试；
- 不下载模型，不调用 GPU。

### content_write_and_vae: Content Write And VAE Path

- 建立 clean/watermarked 同基础 latent 配对；
- 在 callback index 18 按 `main` 请求的 binary32 scale 物化 nominal content delta；
- 计算 `delta_content_actual`、realized combined total norm/relative L2 和 replay
  identity；
- generation decode 使用冻结 VAE scaling/shift；
- detection encode 使用 VAE posterior mode，不采样；
- CPU/mock 覆盖 callback 未触发、重复/错误 index、dtype 写入消失、非有限量、
  overflow 和 deterministic binary16 replay；
- `content_relative_l2_nominal=content_relative_l2_limit=3/250`；runtime 只物化、
  测量和执行完整性检查，`main.content_embedder` 独占 hard-budget 接受、重试、
  scale 选择和最终 fail-closed；
- CPU/property 覆盖 row-major binary32 算术、binary16 RNE/subnormal/overflow、
  预算谓词单调性、无新 midpoint 终止、最大非零可行选择、zero plateau、轻微超限
  和无非零可行写入；
- 不设置 `tau_actual_budget`、`q_budget` 接近门、经验 tolerance 或 actual 强度
  下限；ratio/utilization 只作诊断，低 utilization 不得结果后筛除。

content_write_and_vae 的本地实现与测试通过本身不表示本批完成；当前精确 candidate 的真实
SD3.5 callback、actual float16、VAE 路径已经由后续 GPU qualification 闭合。

### qk_observation: Q/K Observation Path

- 从普通待检图像重新建立检测 latent；
- 使用冻结检测 schedule、公开确定性噪声和登记 Q/K 层；
- runtime 只返回方法需要的 observation；
- 不读取生成缓存、embed record、参考图或私有嵌入状态；
- CPU/mock 覆盖缺层、重复 hook、shape/dtype 错误和非有限量；
- 真实 Q/K 内容留到 Colab GPU 验证。

当前本地实现提供可直接绑定真实 attention module 的 `to_q`/`to_k` hook 接口，
并已覆盖 posterior-mode、public-noise、schedule/conditioning identity 和上述
fail-closed 路径；精确 candidate 的两登记层真实 Q/K 已由后续 Colab GPU
qualification 捕获并核验。

### runtime_qualification_delivery: Runner, Result Zip And Thin Notebook

- 建立一个可从命令行运行的 runtime qualification runner；
- runner 提供 `smoke`、`qualification`，必要时提供 `replay`；
- runner 自己生成结果文件、失败状态和 zip，Notebook 不手写结果；
- 建立 execution package 外、绑定 package schema version 1 的独立可信 bootstrap；
- 建立最小 execution package，供没有 Git remote 的 Colab 会话使用；
- 创建唯一 `notebooks/colab/runtime_qualification.ipynb`；
- Notebook boundary、bootstrap 安全边界、smoke/integration 选择和 execution
  package 在本地先检查。

当前本地实现已提供上述 backend、runner、revision-bound package builder 和唯一薄
Notebook 源。可信 bootstrap 不从待验证 package 导入，也不进入 package 自验证；
它们仍须完成本批 CPU/mock 审计与注册 `full` profile 后才能固化。
runner 可捕获普通 Python/runtime 失败并写出最小 failure zip；若解释器硬崩溃、
进程被系统直接杀死或结果存储不可写，进程内打包不可能完成，必须诚实登记为
`incomplete` 或 `resource_failure`，不得由 Notebook 伪造通过记录。

runtime_qualification_delivery 最终 revision 提交且工作树干净后，只能从 exact HEAD 的 tracked blobs
构建 execution package。`<输出目录>` 必须在仓库外：

```bash
RUNTIME_CANDIDATE_REVISION="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
PYTHONDONTWRITEBYTECODE=1 python \
  scripts/experiment_execution/build_runtime_qualification_package.py \
  --root . \
  --runtime-candidate-revision "${RUNTIME_CANDIDATE_REVISION}" \
  --output-zip \
  "<输出目录>/ceg_wm_runtime_execution_${RUNTIME_CANDIDATE_REVISION}.zip"
```

Drive 中 execution package 的身份与 Notebook 入口按以下固定规则处理：

1. 唯一权威、不可变 archive 是
   `execution_packages/<runtime_candidate_revision>/ceg_wm_runtime_execution_<runtime_candidate_revision>.zip`；
   已冻结的 revision-specific archive 不得覆盖或改写。
2. `execution_packages/current/ceg_wm_runtime_execution.zip` 只是固定 Notebook
   ingress alias，不是 revision 或证据权威。
3. alias 必须从上述权威 archive 逐字节复制；不得为 alias 重建、重新打包、重新
   压缩或修改 zip。
4. 复制后必须分别计算权威 archive 与 alias 的 SHA-256，并确认两者完全相同。
5. current 路径不提供运行身份；Notebook 解包并校验后，只以包内
   `runtime_execution_manifest.json` 的 `runtime_candidate_revision` 作为运行身份。
6. Notebook 把结果写入
   `runs/<runtime_candidate_revision>/<run_id>/`，其中 revision 来自已校验的
   manifest。更换候选时，只能用另一个已冻结的权威 archive 覆盖 alias，并重新
   核对两者 SHA-256 相同；不得改写既有权威 archive 或历史 results。

package 进入 runner 前必须经过独立可信 bootstrap：

1. bootstrap 固定支持 package schema version 1，其仓库 revision 和完整文件
   SHA-256 由独立审核给出；bootstrap 文件不能来自待验证 package。
2. Notebook 对 Drive bootstrap 只读取一次；先比较该 bytes 的 SHA-256，再以 `xb`
   写入全新的 `/content` 本地快照并复核同一摘要，随后只执行本地快照。摘要不匹配
   时不得启动任何 subprocess。完整 package archive SHA-256 必须来自独立审核并
   固定在该 Notebook revision 中；不得自动信任与 archive 同目录的可替换
   sidecar，也不得由用户在运行时替换。
3. bootstrap 只用 Python 标准库执行预信任阶段。在任何 pip、requirements、
   package import 或 runner 启动前，先把 Drive archive 单次流式复制到新建
   ephemeral `xb` 快照并同步计算完整 SHA-256；不匹配时删除快照且不解包。匹配后
   只从本地快照检查 ZIP traversal、绝对/Windows drive 路径、反斜线、重复成员、
   symlink、成员/总大小，安全解包并验证 manifest
   schema/profile/readiness/revision、allowlist、完整文件集及逐文件 hash/size。
4. 全部检查通过后才安装冻结 requirements，并调用包内 runner。runner 继续独立
   复核 package/dependency identity，不把 bootstrap 当作方法或 runtime 证据。

runner 的退出码 `0/1/2` 分别表示通过、已完成的失败、incomplete/preflight 失败；
这三种情况只要 runner 形成正式 result zip，bootstrap 都独立检查 result schema
version 2、身份、固定文件集和退出状态，再复制到 revision/run-id Drive 目录。若在
runner 可启动或形成正式结果前发生 archive ingress、解包、manifest、pip 或启动
失败，bootstrap 写入独立的
`ceg_wm_runtime_bootstrap_failure_<run_id>.zip`，其中只含
`bootstrap_failure.json`。该诊断不是 qualification result，不得伪装通过或支撑
`runtime_verified`。

runner 的 `result_zip`、`ephemeral_root` 和 `persistent_root` 没有默认值，调用方
必须显式提供。结果 zip 必须严格位于 ephemeral root 内；ephemeral 与 persistent
root 在相等和两个祖先方向都必须不相交。只有 `replay` profile 可以提供
`replay_source`，且 source 必须严格位于 persistent root 内；smoke/qualification
携带 replay source 必须 fail closed。runtime backend 接收同一个 persistent root，
并独立拒绝与其相等、包含它或被它包含的模型 cache root。bootstrap 因此先让 runner
在 `/content` 临时根生成结果，独立核验后再逐字节复制到 manifest
revision/run-id 对应的 Drive 目录；这不改变上述 archive/alias 身份规则。

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

### Actual-Dtype Semantic Revision Closure

actual-dtype 预算语义属于 registered design 与 readiness 受保护实现，必须使用两个
独立 revisions 闭合：

1. candidate_semantics_revision 同步 registered design、`main/content_chain/embedder.py`、runtime handshake
   和真实行为/property tests；只运行定向 CPU/static 检查。旧
   `method_readiness.yaml` 在 candidate_semantics_revision 后暂时 stale 是预期事实，不得运行 completion
   profile 或声称 readiness 闭合。
2. 独立语义审计必须绑定 candidate_semantics_revision exact revision、新 candidate SHA 和全部受影响
   方法/测试路径，给出 `APPROVE` 或 `REQUEST CHANGES`。
3. 仅在 `APPROVE` 后创建 readiness_rebinding_revision；readiness_rebinding_revision 只更新 readiness 的 candidate SHA、reviewed
   revision、真实审核引用及必要纯状态绑定，不夹带方法/runtime 修复。
4. readiness_rebinding_revision 在登记 `CEG-WM` Conda 环境运行唯一 `full` profile，再由独立 gatekeeper
   核对 candidate_semantics_revision→readiness_rebinding_revision、candidate digest、protected paths、工作树和 stage 仍为
   `method_implemented`。未通过不得进入 qk_observation。

该两-revision 闭环不等于 content_write_and_vae、GPU qualification 或 `runtime_verified` 完成。

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
runtime 在 callback index 18 按 main 请求的 scale 以 actual dtype 物化
  ↓
runtime 返回 actual 张量、replay、integrity 和 realized 测量
  ↓
main 直接执行 hard-budget 比较；超限时驱动 binary32 最大可行 scale 搜索
  ↓
VAE 解码普通 RGB 图像
```

冻结物化/预算摘要为：

```text
z_s = cast_binary16_RNE(fp32(z0) + f32(s*delta_content_nominal))
delta_actual_s = fp32(z_s) - fp32(z0)
A = norm32_row_major(delta_actual_s)
L = f32(norm32_row_major(fp32(z0)) * f32(3/250))
accept iff A <= L
```

full scale 超限时，`main.content_embedder` 在 binary32 `[0,1]` 上使用冻结 midpoint
二分，直到 midpoint 与边界 bitwise 相同；zero plateau 不算可行写入，返回最大
非零可行 scale 或 fail closed。runtime 不输出独立 budget decision。hard limit
只约束 LF/HF/routing 最终 combined content delta；nominal directions 不是 actual
branch decomposition，geometry budget 独立。

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

1. 显示任务范围并挂载 Google Drive；
2. 使用独立审核后固定在 Notebook 源中的 candidate revision、`PROFILE`、package
   路径、expected package SHA-256 和 replay source；
3. 检查 GPU/CUDA 和可用临时磁盘；
4. 从 Colab Secrets 读取 `HF_TOKEN` 与 `CEG_WM_ROOT_KEY`；
5. 单次读取独立可信 bootstrap，比较其 SHA-256，以 `xb` 写入新的 `/content`
   本地快照并复核摘要；
6. 只用上述本地可信快照，以显式 package、expected SHA、ephemeral/persistent
   roots 调用 bootstrap；
7. 显示 bootstrap 返回的正式结果或诊断包路径和简短状态。

Notebook 不得包含：

- package manifest schema、allowlist、逐文件 hash/size 或安全解包实现；
- 正式 result zip 文件清单、result schema 或通过判定；
- 方法、runtime、Q/K hook、阈值或统计实现；
- 手写 records、通过判定或 zip 文件清单；
- 模型 fallback、自动降级或结果后改参数；
- token、root key、模型权重或私有数据；
- 把 `/root/.cache/huggingface`、`HF_HOME`、pip cache 或临时模型目录设置到 Drive。

当前 Notebook revision 固定绑定 candidate
`8b2344756c4c247906ff0d4eab68e46a773e13f5`、`PROFILE="qualification"`、`current`
package 路径、独立审核的完整 package SHA-256
`8290abeed79931eb7208ac9ca280f1ea401f4725abfead35f12617a0ef54dd38` 和
`REPLAY_SOURCE=None`。该冻结快照已经完成一次授权的 **Run all**；不要重复运行或
编辑并保存 Notebook 源。该 candidate 的 smoke 与 qualification 结果均保存在独立
revision/run-id 目录，互不覆盖。后续切换
`smoke`、`replay` 或
候选 package 时，必须由实施者修改固定快照，经独立审计者和 gatekeeper 审核后形成
新的 Notebook revision。Colab 自动写入的 outputs 和 execution counts 不回写仓库；
后续功能变化修改 bootstrap/runner/config 并重新审核 trust anchor。

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
├── bootstrap\
│   └── package_schema_1\
│       └── runtime_qualification_bootstrap.py
├── execution_packages\
│   ├── <runtime_candidate_revision>\
│   │   └── ceg_wm_runtime_execution_<runtime_candidate_revision>.zip
│   └── current\
│       └── ceg_wm_runtime_execution.zip
├── runs\
    └── <runtime_candidate_revision>\
        └── <run_id>\
            └── ceg_wm_runtime_qualification_<run_id>.zip
└── bootstrap_failures\
    └── <run_id>\
        └── ceg_wm_runtime_bootstrap_failure_<run_id>.zip
```

revision-specific archive 是不可变权威对象；`current` 文件只是在运行前从所选
权威 archive 逐字节复制得到的 Notebook ingress alias。两份文件的 SHA-256 必须
相同；实际 revision 必须从 alias 包内已校验 manifest 读取，结果按该 revision
归档。切换候选时只覆盖 alias 并重核 SHA-256，不覆盖历史权威 archive 或 results；
同时必须形成绑定新 archive SHA-256 的三角色审核 Notebook revision，不能由用户
只改 alias 或 Notebook 常量。

Notebook 工作副本可放在：

```text
G:\我的云端硬盘\Colab Notebooks\CEG-WM\runtime_qualification.ipynb
```

Drive 中只保存：

- 独立审核并由 Notebook 固定 SHA-256 的 package-schema-v1 bootstrap；
- 小型 execution package；
- 固定 Notebook 工作副本；
- smoke/qualification/replay 结果 zip；
- runner 前失败产生的小型 `bootstrap_failure` 诊断 zip；
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
- actual-dtype 写入没有消失，独立 replay 通过，`main` 返回 accepted 的
  nonzero maximal feasible scale，预算状态与尝试身份可追溯；
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

replay 继续使用唯一 Notebook 路径和同一 runner，但必须形成固定 replay 参数的新
Notebook revision；不新增第二种 Notebook 入口。

## Minimal Result Zip

正式 schema version 2 结果 zip 由 runner 自动生成，文件集必须精确为：

```text
run_summary.json
environment_summary.json
runtime_checks.jsonl
failures.jsonl
```

不得加入 `console.log`、`artifacts/` 或其他可选成员。额外诊断只能通过既有四个成员的
登记字段表达；runner 启动前失败则使用独立、只含 `bootstrap_failure.json` 的诊断包。

`run_summary.json` 按 schema version 2 的冻结字段集记录：

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
- 与旧 Colab 快照逐包逐版本完整字符串一致；PyTorch 仍必须满足上述冻结公共版本
  与 local build label 规则；
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

当前精确 candidate 已满足上述条件，并由独立 stage-only revision 同步
`runtime_verified`。qualification 没有具体含混点，因此 Optional Replay 不构成本次
阶段迁移的强制前置条件。`negative_identity` 也只证明 key identity control 的
runtime 分离，不得解释为 wrong-key FPR 或 attribution 科学结果。

## Stop And Notify Rules

Codex 必须停止并通知用户：

- 本地 CPU 工作完成，需要首次运行 Colab/GPU；
- Colab execution package 和 Notebook 已准备好；
- 模型下载、OOM、callback、VAE 或 Q/K 检查失败；
- 需要更换登记模型、revision、scheduler、steps、dtype 或 callback；
- qualification 完成，结果 zip 已可本地检查；
- 需要进入 `runtime_verified`；
- 工作意图扩展到 LF/routing 候选选择、calibration 或正式实验。
