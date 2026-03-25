## 1. 文档目的

本文档用于定义 `notebook_v2/` 目录下各个 Colab Notebook 的职责边界、输入输出、运行顺序、依赖关系、建议的 Cell 组织方式，以及实现约束。其目标不是解释算法细节，而是为第三方提供一份**可直接据此构建各 Notebook Cell** 的工程说明文档。

本文档面向如下场景：

1. 项目所有正式生成、测试、审计与论文数字生产均在 **Google Colab** 中完成。
2. 本地无 GPU 环境。
3. 默认所有中间产物与最终产物都写入 **Google Drive 的 `project_root`**，以支持多 Notebook、多会话复用。
4. Notebook 尽可能只作为**流程编排层（orchestration layer）**，不在 Notebook 中重写项目核心机制，不直接修改 `main/` 中算法实现。
5. 如可行，Notebook 仅通过 **live `scripts/`** 或 **`main.cli.*`** 调用项目逻辑。
6. `archive/` 目录已被忽略，不作为任何 Notebook 的依赖来源。
7. 模型与权重下载必须在 Notebook 中显式完成，不依赖本地预置 `models/` 目录。
8. 当前项目中，`Paper_Full_Cuda.ipynb` 已具备下载 `InSPyReNet` 权重并移动到对应目录的逻辑；`notebook_v2/` 应沿用该思路。

---

## 2. 总体设计原则

### 2.1 分层原则

`notebook_v2/` 采用四层结构：

1. `core/`  
   项目论文机制主链路，`Full model` 作为基线。

2. `ablation/`  
   论文消融层，区分：
   - 主结构消融；
   - 修复机制消融。

3. `eval/`  
   论文数字生产层，用于生成 `LPIPS`、`SSIM`、`FID`、`TPR@FPR`、攻击矩阵等最终论文结果。

4. `release/`  
   发布与可复现收口层，用于构建 `repro bundle`、审计与最终 `signoff`。

---

### 2.2 Notebook 职责边界

每个 Notebook 只负责一个清晰阶段，不承担多个大型阶段的混合逻辑。禁止以下做法：

1. 将主链生成、攻击生成、质量评估、表格导出全部塞进一个 Notebook。
2. 在 Notebook 内复制项目算法逻辑。
3. 通过 Notebook 临时修改 `main/` 中核心实现。
4. 依赖 `archive/` 中脚本。

Notebook 应只承担以下任务：

1. 挂载 Google Drive；
2. 拉取或更新代码；
3. 安装环境；
4. 下载模型与权重；
5. 预检输入文件与路径；
6. 生成 config snapshot；
7. 调用 live `scripts/` 或 `main.cli.*`；
8. 读取并汇总结构化输出；
9. 将摘要写入 `project_root`。

---

### 2.3 持久化原则

默认所有结果写入 Google Drive 下的：

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/CEG_WM_Project_V2")
````

任何 Notebook 不应默认把正式产物保存在 Colab 临时路径中。Colab 临时路径只用于：

1. 仓库工作副本；
2. Python 环境；
3. 缓存中的临时计算；
4. 模型下载后向项目目录中的复制。

---

### 2.4 模型与权重原则

任何需要运行 embed / detect / experiment matrix / ablation 的 Notebook，都必须显式完成如下准备：

1. 下载或复用 Hugging Face 模型缓存；
2. 显式下载 `InSPyReNet` 权重；
3. 在仓库目录下创建：

   ```text
   <REPO_ROOT>/models/inspyrenet/ckpt_base.pth
   ```
4. 保证配置中的 `mask.semantic_model_path` 最终指向该绝对路径。

---

### 2.5 配置快照原则

Notebook 不直接修改仓库中的原始配置文件。任何 Notebook 在调用 live `scripts/` 或 `main.cli.*` 前，都应生成一份**本轮运行专用的 config snapshot**，写入：

```text
project_root/config_snapshots/<layer>/
```

必须优先将以下路径改为绝对路径：

1. `mask.semantic_model_path`
2. `inference_prompt_file`
3. `calibration.detect_records_glob`
4. `evaluate.detect_records_glob`
5. `evaluate.thresholds_path`

---

## 3. 目录结构

```text
notebook_v2/
├── README.md
├── core/
│   ├── 01_Paper_Full_Cuda.ipynb
│   ├── 02_Parallel_Attestation_Statistics.ipynb
│   └── 03_Experiment_Matrix_Full.ipynb
├── ablation/
│   ├── 00_Ablation_Manifest_And_Baseline.ipynb
│   ├── structural/
│   │   ├── 01_Geometry_And_Anchor_Ablation.ipynb
│   │   ├── 02_LF_HF_Content_Ablation.ipynb
│   │   └── 03_Structural_Ablation_Tables.ipynb
│   └── repair/
│       ├── 01_GEO_Repair_Ablation.ipynb
│       ├── 02_LF_Exact_Repair_Ablation.ipynb
│       └── 03_Repair_Ablation_Tables.ipynb
├── eval/
│   ├── 00_Manifest_And_Splits.ipynb
│   ├── 01_Clean_Generation.ipynb
│   ├── 02_Readonly_Calibration.ipynb
│   ├── 03_Quality_Metrics.ipynb
│   ├── 04_Attack_Generation.ipynb
│   ├── 05_Detect_And_TPR.ipynb
│   └── 06_Tables_And_Figures.ipynb
└── release/
    ├── 00_Repro_Bundle.ipynb
    ├── 01_All_Audits.ipynb
    └── 02_Publish_And_Signoff.ipynb
```

---

## 4. `project_root` 建议结构

```text
project_root/
├── cache/
│   ├── hf/
│   └── inspyrenet/
│       └── ckpt_base.pth
├── repo_state/
│   └── git_version_info.json
├── config_snapshots/
│   ├── core/
│   ├── ablation/
│   ├── eval/
│   └── release/
├── manifests/
│   ├── ablation/
│   └── eval/
├── runs/
│   ├── core/
│   │   ├── paper_full_cuda/
│   │   └── experiment_matrix_full/
│   ├── ablation/
│   │   ├── baseline/
│   │   ├── structural/
│   │   └── repair/
│   └── eval/
│       ├── clean_generation/
│       ├── readonly_calibration/
│       ├── quality_metrics/
│       ├── attack_generation/
│       ├── detect_and_tpr/
│       └── tables/
├── reports/
│   ├── ablation/
│   └── eval/
├── release/
│   ├── repro_bundle/
│   ├── audits/
│   └── signoff/
└── exports/
    ├── zip/
    └── tables/
```

---

## 5. 所有 Notebook 通用的 Cell 模板

每个 Notebook 应按需要复用以下通用 Cell 类型。第三方在实现 Notebook 时，应首先补齐这些标准 Cell，再实现阶段专属 Cell。

### Cell A0：Notebook 作用说明

功能如下：

1. 说明 Notebook 目的；
2. 列出输入与输出；
3. 说明是否需要 GPU；
4. 说明是否需要模型和权重；
5. 说明是否需要 `CEG_WM_K_*` 环境变量；
6. 说明上游依赖。

---

### Cell A1：挂载 Google Drive 与定义路径

功能如下：

1. 挂载 Google Drive；
2. 定义：

   * `PROJECT_ROOT`
   * `CACHE_ROOT`
   * `RUNS_ROOT`
   * `MANIFESTS_ROOT`
   * `CONFIG_SNAPSHOTS_ROOT`
   * `REPORTS_ROOT`
   * `EXPORTS_ROOT`
3. 创建缺失目录。

---

### Cell A2：拉取或更新仓库代码

功能如下：

1. 将仓库 clone / pull 到 Colab 本地，例如：

   ```text
   /content/CEG-WM
   ```
2. 若仓库已存在，则执行 `git fetch`、`git reset --hard`；
3. 将当前 commit、branch、时间写入：

   ```text
   project_root/repo_state/git_version_info.json
   ```

---

### Cell A3：安装环境与依赖

功能如下：

1. 执行 `pip install -e .`；
2. 安装 Notebook 所需附加依赖；
3. 记录安装结果。

说明如下：

* 基础 Notebook 一般需要：`pyyaml`、`pandas`、`pyarrow`；
* 质量指标 Notebook 需要额外安装：`lpips`、`pytorch-fid`；
* 需要 SD3 推理的 Notebook 需要保证项目运行时依赖完整。

---

### Cell A4：预检 live `scripts/`、`main.cli.*` 与配置文件

功能如下：

1. 检查项目必要入口是否真实存在；
2. 检查配置文件是否存在；
3. 检查 Python 导入路径是否正确；
4. 对缺失的 live 脚本执行 `fail-fast`。

---

### Cell A5：显式下载模型与权重

仅对需要 embed / detect / matrix / ablation 的 Notebook 保留。

功能如下：

1. 设置 Hugging Face 缓存目录到：

   ```text
   project_root/cache/hf/
   ```
2. 显式下载所需模型；
3. 显式下载 `InSPyReNet` 的 `ckpt_base.pth`；
4. 创建：

   ```text
   <REPO_ROOT>/models/inspyrenet/
   ```
5. 将权重复制到：

   ```text
   <REPO_ROOT>/models/inspyrenet/ckpt_base.pth
   ```

---

### Cell A6：准备环境变量

仅对 formal 路径相关 Notebook 保留。

功能如下：

1. 检查：

   * `CEG_WM_K_MASTER`
   * `CEG_WM_K_PROMPT`
   * `CEG_WM_K_SEED`
2. 可选检查 Hugging Face token；
3. 对格式错误或缺失执行 `fail-fast`。

---

### Cell A7：运行前总预检

功能如下：

1. 检查 GPU 是否可用；
2. 检查上游产物是否存在；
3. 检查配置 snapshot 是否准备好；
4. 打印最终执行参数与输出目录；
5. 明确是否允许覆盖已有结果。

---

## 6. 当前 live 仓库中已存在的入口

以下入口当前可直接用于 Notebook 编排：

1. `scripts/run_paper_full_cuda.py`
2. `scripts/run_parallel_attestation_statistics.py`
3. `scripts/run_experiment_matrix.py`
4. `scripts/run_all_audits.py`
5. `main.cli.run_embed`
6. `main.cli.run_detect`
7. `main.cli.run_calibrate`
8. `main.cli.run_evaluate`

---

## 7. 当前 live 仓库中缺失、但 Notebook 设计所要求的脚本

为了保证 Notebook 仅做流程编排，以下 live 脚本需要先补齐：

1. `scripts/run_eval_detect_dataset.py`
2. `scripts/run_quality_metrics.py`
3. `scripts/run_attack_generation.py`
4. `scripts/run_eval_tables.py`
5. `scripts/run_repro_bundle.py`
6. `scripts/run_publish_signoff.py`

### 7.1 `scripts/run_eval_detect_dataset.py`

职责如下：

1. 读取 clean 或 attacked 图片 manifest；
2. 基于图片生成 detect 记录；
3. 为每条记录绑定 `is_watermarked` 标签；
4. 输出 detect manifest 与 detect records 目录；
5. 支持 calibration、clean eval、attack eval 三种场景。

---

### 7.2 `scripts/run_quality_metrics.py`

职责如下：

1. 读取 paired clean manifest；
2. 计算逐样本 `LPIPS` 与 `SSIM`；
3. 计算集合级 `FID`；
4. 输出 `per_sample_metrics.parquet` 与 `quality_summary.json`。

---

### 7.3 `scripts/run_attack_generation.py`

职责如下：

1. 读取 clean paired manifest 与 attack plan；
2. 对正样本与负样本图片施加攻击；
3. 输出 attacked images；
4. 生成 `attack_manifest.parquet` 与 `attack_generation_summary.json`。

---

### 7.4 `scripts/run_eval_tables.py`

职责如下：

1. 读取 `quality_summary.json`；
2. 读取 clean / attack evaluation reports；
3. 输出主表、攻击矩阵表、LaTeX 表与图。

---

### 7.5 `scripts/run_repro_bundle.py`

职责如下：

1. 汇总正式 run 及论文结果；
2. 收集 configs、records、manifests、reports；
3. 生成最终 `repro bundle`；
4. 输出 `bundle_manifest.json`。

---

### 7.6 `scripts/run_publish_signoff.py`

职责如下：

1. 读取 `repro bundle` 与审计报告；
2. 汇总最终发布级结论；
3. 输出 `signoff_summary.json`；
4. 形成最终发布目录。

---

## 8. `core/` 层说明

---

### 8.1 `core/01_Paper_Full_Cuda.ipynb`

#### 目的

该 Notebook 是项目正式 `Full model` 主链 Notebook，用于生成项目全机制完整产物，并作为论文中 `Full` 基线的唯一正式来源。

#### 实现方式

该 Notebook 只做流程编排，最终调用：

```text
scripts/run_paper_full_cuda.py
```

#### 输入

1. `configs/paper_full_cuda.yaml`
2. 正式 prompt 文件
3. SD3 模型
4. `InSPyReNet` 权重
5. `CEG_WM_K_*` 环境变量

#### 输出目录

```text
project_root/runs/core/paper_full_cuda/
```

#### 建议 Cell 组成

1. A0：Notebook 说明
2. A1：挂载 Drive
3. A2：拉取代码
4. A3：安装环境
5. A4：预检 live 脚本与 config
6. A5：下载模型与 `InSPyReNet`
7. A6：准备环境变量
8. A7：运行前总预检
9. 生成 config snapshot，写入：

   ```text
   project_root/config_snapshots/core/paper_full_cuda.yaml
   ```
10. 调用：

    ```bash
    python scripts/run_paper_full_cuda.py --config <snapshot> --run-root <run_root>
    ```
11. 读取结构化输出：

    * `embed_record.json`
    * `detect_record.json`
    * `calibration_record.json`
    * `evaluate_record.json`
    * `workflow_summary.json`
12. 写出 Notebook 级摘要。

---

### 8.2 `core/02_Parallel_Attestation_Statistics.ipynb`

#### 目的

该 Notebook 从正式 `Full model` 的主 `run_root` 派生，独立运行 `parallel attestation statistics`，用于生成事件级 attestation 统计结果。

#### 实现方式

调用：

```text
scripts/run_parallel_attestation_statistics.py
```

#### 上游依赖

```text
project_root/runs/core/paper_full_cuda/
```

#### 输出目录

位于主 `run_root` 下，例如：

```text
project_root/runs/core/paper_full_cuda/outputs/parallel_attestation_statistics/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. 读取主 `run_root` 路径
7. 预检：

   * 主 `run_root` 是否存在；
   * `detect_record.json` 是否存在。
8. 调用 detached script
9. 读取 parallel summary
10. 写出 Notebook 摘要。

---

### 8.3 `core/03_Experiment_Matrix_Full.ipynb`

#### 目的

该 Notebook 用于在 `Full model` 条件下执行正式 `experiment matrix`，为攻击矩阵与 full baseline robustness 结果提供专项输出。

#### 实现方式

调用：

```text
scripts/run_experiment_matrix.py
```

#### 输入

1. `configs/paper_full_cuda.yaml` 的 matrix 专用 snapshot
2. SD3 模型
3. `InSPyReNet` 权重
4. `CEG_WM_K_*`

#### 输出目录

```text
project_root/runs/core/experiment_matrix_full/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. A5
7. A6
8. A7
9. 基于 `paper_full_cuda.yaml` 生成 matrix 专用 config snapshot
10. 调用 `run_experiment_matrix.py`
11. 读取：

    * `grid_summary.json`
    * `aggregate_report.json`
12. 写出 Notebook 摘要。

---

## 9. `ablation/` 层说明

---

### 9.1 `ablation/00_Ablation_Manifest_And_Baseline.ipynb`

#### 目的

该 Notebook 负责为 repair 型消融提供共同基线对象，包括：

1. ablation manifest；
2. baseline embed；
3. baseline full detect；
4. 可复用的 `base_embed_record`。

#### 实现方式

通过 `main.cli.run_embed` 与 `main.cli.run_detect` 完成，不依赖额外脚本。

#### 输出目录

```text
project_root/runs/ablation/baseline/
project_root/manifests/ablation/baseline_manifest.json
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. A5
7. A6
8. A7
9. 基于 `configs/ablation/paper_ablation_cuda.yaml` 生成 baseline config snapshot
10. 运行 baseline embed
11. 运行 baseline detect
12. 写出 `baseline_manifest.json`
13. 抽检 `embed_record` 与 `detect_record`。

---

## 9.2 `ablation/structural/01_Geometry_And_Anchor_Ablation.ipynb`

#### 目的

该 Notebook 负责结构性消融中的：

1. `Full`
2. `w/o Geometry`
3. `w/o Anchor`

#### 实现方式

使用 `scripts/run_experiment_matrix.py`，通过专用 config snapshot 限定 `ablation_variants`。

#### 输出目录

```text
project_root/runs/ablation/structural/geometry_anchor_matrix/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. A5
7. A6
8. A7
9. 生成只包含 `Full / w/o Geometry / w/o Anchor` 的 snapshot
10. 调用 `run_experiment_matrix.py`
11. 读取矩阵结果
12. 输出 reduced summary。

---

## 9.3 `ablation/structural/02_LF_HF_Content_Ablation.ipynb`

#### 目的

该 Notebook 负责结构性消融中的：

1. `Full`
2. `w/o LF`
3. `w/o HF`
4. 可选：`w/o Content`

#### 实现方式

同样调用 `scripts/run_experiment_matrix.py`，但变体为 `enable_lf` / `enable_hf` / `enable_content`。

#### 输出目录

```text
project_root/runs/ablation/structural/lf_hf_content_matrix/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. A5
7. A6
8. A7
9. 生成 LF/HF/content 专用 snapshot
10. 调用 matrix 脚本
11. 读取结果
12. 写出 reduced summary。

---

## 9.4 `ablation/structural/03_Structural_Ablation_Tables.ipynb`

#### 目的

汇总结构消融结果，导出论文中的结构消融表格与图。

#### 实现方式

该 Notebook 不运行模型，不调用重型流程，只做结果读取、合并与导出。

#### 输出目录

```text
project_root/reports/ablation/structural/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. 预检上游 reduced summary 文件
7. 合并并规范化标签
8. 输出：

   * `.csv`
   * `.tex`
   * 图。

---

## 9.5 `ablation/repair/01_GEO_Repair_Ablation.ipynb`

#### 目的

对同一对象执行 detect rerun，比对：

1. `GEO repair on`
2. `GEO repair off`

#### 实现方式

通过 `main.cli.run_detect` 重跑 detect，并使用来自 `ablation/00` 的 `base_embed_record`。

#### 输出目录

```text
project_root/runs/ablation/repair/geo_repair/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. A5
7. A6
8. A7
9. 读取 `baseline_manifest.json`
10. 生成 `geo_repair_on.yaml` 与 `geo_repair_off.yaml`
11. 分别调用 `run_detect`
12. 读取并比较 detect records
13. 写出 compare 表。

---

## 9.6 `ablation/repair/02_LF_Exact_Repair_Ablation.ipynb`

#### 目的

对同一对象执行 detect rerun，比对：

1. `LF exact repair on`
2. `LF exact repair off`

#### 实现方式

通过 `main.cli.run_detect` 重跑 detect，并使用同一个 `base_embed_record`。

#### 输出目录

```text
project_root/runs/ablation/repair/lf_exact_repair/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. A5
7. A6
8. A7
9. 读取 baseline manifest
10. 生成 on/off snapshot
11. 两次调用 detect
12. 比较 detect records
13. 写出 compare 表。

---

## 9.7 `ablation/repair/03_Repair_Ablation_Tables.ipynb`

#### 目的

汇总 repair 类消融结果。

#### 实现方式

只做文件读取与结果导出。

#### 输出目录

```text
project_root/reports/ablation/repair/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. 预检 compare 文件
7. 合并结果
8. 导出 `.csv`、`.tex`。

---

## 10. `eval/` 层说明

---

### 10.1 `eval/00_Manifest_And_Splits.ipynb`

#### 目的

冻结论文评测的样本口径，生成：

1. `calibration_manifest.parquet`
2. `clean_eval_manifest.parquet`
3. `attack_plan.parquet`
4. `paired_index.csv`

#### 实现方式

该 Notebook 不运行模型，只在 Python 中构建 manifest。

#### 输出目录

```text
project_root/manifests/eval/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. 读取 prompt 列表与 seeds
7. 构建 split
8. 写出 manifests
9. 检查 split 不重叠并写出 summary。

---

### 10.2 `eval/01_Clean_Generation.ipynb`

#### 目的

基于 `eval/00` 的 manifests 批量生成：

1. baseline clean images
2. watermarked clean images
3. paired clean manifest

#### 实现方式

通过循环调用 `main.cli.run_embed` 实现。

#### 输出目录

```text
project_root/runs/eval/clean_generation/
project_root/manifests/eval/paired_clean_manifest.parquet
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. A5
7. A6
8. A7
9. 读取 eval manifests
10. 为每个样本生成 embed config snapshot
11. 循环调用 `run_embed`
12. 从 sample run 中收集 baseline / watermarked 图片路径
13. 写出 paired clean manifest
14. 写出 Notebook 摘要。

---

### 10.3 `eval/02_Readonly_Calibration.ipynb`

#### 目的

在 `calibration split` 上生成 detect dataset，并执行只读阈值校准。

#### 实现方式

该 Notebook 要求存在：

```text
scripts/run_eval_detect_dataset.py
```

随后再调用：

```text
main.cli.run_calibrate
```

#### 输出目录

```text
project_root/runs/eval/readonly_calibration/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. A5
7. A6
8. A7
9. 预检缺失脚本是否存在
10. 读取 `paired_clean_manifest.parquet`
11. 调用 `run_eval_detect_dataset.py` 生成 calibration detect dataset
12. 生成 calibrate config snapshot
13. 调用 `run_calibrate`
14. 读取 thresholds artifact
15. 写出 calibration summary。

---

### 10.4 `eval/03_Quality_Metrics.ipynb`

#### 目的

计算：

1. `LPIPS`
2. `SSIM`
3. `FID`

#### 实现方式

该 Notebook 要求存在：

```text
scripts/run_quality_metrics.py
```

#### 输出目录

```text
project_root/runs/eval/quality_metrics/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. 安装 `lpips` 与 `pytorch-fid`
7. 预检 `paired_clean_manifest.parquet`
8. 调用 `run_quality_metrics.py`
9. 读取 `per_sample_metrics.parquet`
10. 读取 `quality_summary.json`
11. 写出 Notebook 摘要。

---

### 10.5 `eval/04_Attack_Generation.ipynb`

#### 目的

根据 `attack_plan` 对 clean images 施加攻击，生成 attacked image set。

#### 实现方式

该 Notebook 要求存在：

```text
scripts/run_attack_generation.py
```

#### 输出目录

```text
project_root/runs/eval/attack_generation/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. 预检：

   * `attack_plan.parquet`
   * `paired_clean_manifest.parquet`
   * `configs/attack_protocol.yaml`
7. 调用 `run_attack_generation.py`
8. 读取 `attack_manifest.parquet`
9. 写出摘要。

---

### 10.6 `eval/05_Detect_And_TPR.ipynb`

#### 目的

读取只读阈值，对 clean eval 与 attacked eval 进行 detect 与 evaluate，生成：

1. clean `TPR@FPR`
2. attack `TPR@FPR`

#### 实现方式

依赖：

1. `scripts/run_eval_detect_dataset.py`
2. `main.cli.run_evaluate`

#### 输出目录

```text
project_root/runs/eval/detect_and_tpr/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. A5
7. A6
8. A7
9. 预检：

   * `readonly_calibration` 的 thresholds；
   * `attack_manifest.parquet`；
   * 缺失脚本。
10. 生成 clean detect dataset
11. 生成 attack detect dataset
12. 生成 evaluate config snapshot
13. 对 clean 调用 `run_evaluate`
14. 对 attack 调用 `run_evaluate`
15. 读取 evaluation report
16. 写出 notebook summary。

---

### 10.7 `eval/06_Tables_And_Figures.ipynb`

#### 目的

汇总论文中的最终结果表格与图。

#### 实现方式

要求存在：

```text
scripts/run_eval_tables.py
```

#### 输出目录

```text
project_root/runs/eval/tables/
project_root/exports/tables/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. 预检：

   * `quality_summary.json`
   * clean evaluation report
   * attack evaluation report
   * live table script
7. 调用 `run_eval_tables.py`
8. 展示主表与攻击矩阵表
9. 复制导出到 `exports/tables/`。

---

## 11. `release/` 层说明

---

### 11.1 `release/00_Repro_Bundle.ipynb`

#### 目的

构建最终可复现 bundle。

#### 实现方式

要求存在：

```text
scripts/run_repro_bundle.py
```

#### 输出目录

```text
project_root/release/repro_bundle/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. 预检核心上游结果
7. 调用 `run_repro_bundle.py`
8. 读取 `bundle_manifest.json`
9. 输出 bundle 摘要。

---

### 11.2 `release/01_All_Audits.ipynb`

#### 目的

执行统一审计入口。

#### 实现方式

调用：

```text
scripts/run_all_audits.py
```

#### 输出目录

```text
project_root/release/audits/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. 定义 audit output 路径
7. 调用 `run_all_audits.py`
8. 读取 audit report
9. 输出 FAIL / BLOCK 摘要。

---

### 11.3 `release/02_Publish_And_Signoff.ipynb`

#### 目的

读取 repro bundle 与 audit results，生成最终发布级 signoff。

#### 实现方式

要求存在：

```text
scripts/run_publish_signoff.py
```

#### 输出目录

```text
project_root/release/signoff/
```

#### 建议 Cell 组成

1. A0
2. A1
3. A2
4. A3
5. A4
6. 预检：

   * repro bundle；
   * audit report；
   * signoff script。
7. 调用 `run_publish_signoff.py`
8. 读取 `signoff_summary.json`
9. 输出最终发布级结论。

---

## 12. 运行顺序

推荐的总体运行顺序如下。

### 12.1 核心机制基线

1. `core/01_Paper_Full_Cuda.ipynb`
2. `core/02_Parallel_Attestation_Statistics.ipynb`
3. `core/03_Experiment_Matrix_Full.ipynb`

其中，`core/02` 与 `core/03` 可在 `core/01` 完成后并行。

---

### 12.2 消融层

1. `ablation/00_Ablation_Manifest_And_Baseline.ipynb`
2. 并行运行：

   * `ablation/structural/01_Geometry_And_Anchor_Ablation.ipynb`
   * `ablation/repair/01_GEO_Repair_Ablation.ipynb`
   * `ablation/repair/02_LF_Exact_Repair_Ablation.ipynb`
3. `ablation/structural/02_LF_HF_Content_Ablation.ipynb`
4. 汇总：

   * `ablation/structural/03_Structural_Ablation_Tables.ipynb`
   * `ablation/repair/03_Repair_Ablation_Tables.ipynb`

---

### 12.3 论文数字生产层

1. `eval/00_Manifest_And_Splits.ipynb`
2. `eval/01_Clean_Generation.ipynb`
3. `eval/02_Readonly_Calibration.ipynb`
4. 并行运行：

   * `eval/03_Quality_Metrics.ipynb`
   * `eval/04_Attack_Generation.ipynb`
5. `eval/05_Detect_And_TPR.ipynb`
6. `eval/06_Tables_And_Figures.ipynb`

---

### 12.4 发布层

1. `release/00_Repro_Bundle.ipynb`
2. `release/01_All_Audits.ipynb`
3. `release/02_Publish_And_Signoff.ipynb`

---

## 13. 上游产物调用关系

### 13.1 `core/01` 的产物

以下 Notebook 会直接或间接读取 `core/01` 的正式结果：

1. `core/02`
2. `release/00`
3. 可选：`ablation/structural/03` 中的 `Full` 基线行
4. 可选：`eval/06` 中的背景说明与 baseline 对照

---

### 13.2 `ablation/00` 的产物

以下 Notebook 必须读取：

1. `ablation/repair/01`
2. `ablation/repair/02`

其核心输入为：

```text
project_root/manifests/ablation/baseline_manifest.json
```

---

### 13.3 `eval/00` 的产物

以下 Notebook 必须读取：

1. `eval/01`
2. `eval/02`
3. `eval/04`

---

### 13.4 `eval/01` 的产物

以下 Notebook 必须读取：

1. `eval/02`
2. `eval/03`
3. `eval/04`
4. `eval/05`

其核心共享产物为：

```text
project_root/manifests/eval/paired_clean_manifest.parquet
```

---

### 13.5 `eval/02` 的产物

以下 Notebook 必须读取只读阈值：

1. `eval/05`

---

### 13.6 `eval/04` 的产物

以下 Notebook 必须读取：

1. `eval/05`

其核心共享产物为：

```text
project_root/runs/eval/attack_generation/attack_manifest.parquet
```

---

### 13.7 `eval/03` 与 `eval/05` 的产物

以下 Notebook 必须读取：

1. `eval/06`
2. `release/00` 可选读取

---

## 14. 实现约束与禁止事项

### 14.1 允许事项

1. 在 Notebook 中生成 config snapshot；
2. 在 Notebook 中组织 shell 调用；
3. 在 Notebook 中对结果做轻量读取、汇总和导出；
4. 在 Notebook 中显式下载模型与权重；
5. 在 Notebook 中进行 fail-fast 预检。

---

### 14.2 禁止事项

1. 在 Notebook 中复制 `main/` 中算法逻辑；
2. 在 Notebook 中热修项目机制代码；
3. 在 Notebook 中依赖 `archive/`；
4. 在 Notebook 中直接写死临时路径而不经 `project_root`；
5. 在 Notebook 中绕过 live `scripts/` 与 `main.cli.*` 私自拼装核心 records。

---

## 15. 第三方实现时的最低完成标准

第三方若依据本 README.md 构建 `notebook_v2/`，至少应满足以下条件：

1. 所有 Notebook 均包含标准前置 Cell；
2. 所有 Notebook 均将结果持久化到 `project_root`；
3. 所有 Notebook 在运行前执行真实存在性预检；
4. 凡是缺失 live script 的 Notebook，均在预检时 fail-fast，而不是静默跳过；
5. Notebook 不修改项目核心代码；
6. Notebook 可在 Colab 多会话下分批独立运行；
7. Notebook 的输出可被后续 Notebook 通过 manifest 或 summary 文件稳定复用。

---

## 16. 本 README 的使用方式

推荐的实际使用方式如下：

1. 先根据本 README 建立 `notebook_v2/` 空目录结构；
2. 先实现 `core/` 中三个 Notebook；
3. 再实现 `ablation/` 中不依赖缺失脚本的 Notebook；
4. 对 `eval/` 与 `release/` 中依赖缺失脚本的部分，先补 live script，再构建对应 Notebook；
5. 每实现一个 Notebook，都应先在 `README.md` 对应条目下核对：

   * 输入；
   * 输出；
   * Cell 组成；
   * 上游依赖；
   * 是否需要模型；
   * 是否需要环境变量。

---

## 17. 最终说明

`notebook_v2/` 的定位不是重新实现项目，而是将已有 live 机制重构为**可在 Colab 中稳定执行的论文级流程编排体系**。
其核心原则是：

1. `core/` 证明正式机制成立；
2. `ablation/` 证明组件贡献；
3. `eval/` 生产论文数字；
4. `release/` 完成可复现收口。

若未来 live `scripts/` 增加，则本 README 中“缺失脚本”相关说明应同步更新。
