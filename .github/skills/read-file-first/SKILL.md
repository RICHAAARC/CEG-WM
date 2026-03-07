---
name: read-file-first
description: Enforce mandatory file reading discipline before any audit, gap analysis, or code review. Use when users ask for audit, gap review, code trace, or any analysis that requires reading actual file content instead of relying on conversation history or memory summaries.
---
# read-file-first

Enforce real file reads before any analysis. Conversation history and memory summaries are NOT acceptable substitutes.

## Execute

### 强制读取清单（按顺序执行）

**第一批：配置文件（必须全文读取）**
- `configs/paper_full_cuda.yaml`
- `configs/frozen_contracts.yaml`
- `configs/policy_path_semantics.yaml`
- `configs/runtime_whitelist.yaml`

**第二批：核心流程文件（读取关键段落）**

| 文件 | 必读位置 |
|------|----------|
| `main/cli/run_embed.py` | `content_inputs` 构造、`plan_digest` 分支、fallback 分支 |
| `main/cli/run_detect.py` | `preplan` 调用、主链路 `content_inputs` 构造、real/fallback 门控 |
| `main/watermarking/embed/orchestrator.py` | `_build_content_inputs_for_embed`、`_resolve_embed_input_image_path` |
| `main/watermarking/detect/orchestrator.py` | `_build_content_inputs_for_detect`、`_resolve_detect_image_path`、real 模式门控、`_build_mismatch_fusion_decision` |
| `main/watermarking/content_chain/semantic_mask_provider.py` | `extract()`、`_probe_model_v2_availability()` |
| `main/watermarking/content_chain/content_detector.py` | `extract()` 的期望输入字段 |
| `main/watermarking/fusion/decision.py` | `decision_status` 所有可能值 |
| `main/watermarking/geometry_chain/sync/latent_sync_template.py` | uncertainty 门控、`sync_strength` 默认值 |

**第三批：真实产物（字段级逐项对比）**
- `outputs/GPU Outputs/colab_run_paper_full_cuda/records/embed_record.json`
- `outputs/GPU Outputs/colab_run_paper_full_cuda/records/detect_record.json`
- `outputs/GPU Outputs/colab_run_paper_full_cuda/records/calibration_record.json`
- `outputs/GPU Outputs/colab_run_paper_full_cuda/records/evaluate_record.json`

## Rules

### 禁止项
- **不得**依赖对话历史中的摘要或代码片段引用替代真实读取。
- **不得**以"之前已读过"为由跳过任何文件。
- **不得**在未读产物文件的情况下引用产物字段值。
- **不得**混用两套产物：`colab_run_paper_full_cuda/` 与 `audit_diagnostics/` 是**不同场景的不同产物**，字段不可交叉引用。

### 产物边界隔离
| 产物集 | 场景 | 允许用途 |
|--------|------|----------|
| `colab_run_paper_full_cuda/records/` | 真实 GPU Colab 运行 | 论文链路字段级证据 |
| `audit_diagnostics/` | 本地无图像输入诊断运行 | 仅用于诊断本地无图像场景 |

### `scripts/` 边界
- `scripts/` 中的代码**仅作为审计结构辅助**，不得将其作为产物字段追踪链路的一部分。
- `run_onefile_workflow.py` 是 Colab 辅助脚本，不是核心流程。
- 若 `run_onefile_workflow.py` 中涉及输入构造或输出写入，必须回到 `main/` 对应文件确认，并以 `main/` 为准。

### 核心发布边界
- 项目核心发布仅包含 `configs/` 和 `main/`。
- 所有核心流程分析必须基于 `configs/paper_full_cuda.yaml` + `main/` 目录代码，禁止依赖 `scripts/` 逻辑。

## Verification Checklist

读取完成后，在分析报告开头输出以下核验清单：

```
[read-file-first 核验]
□ configs/paper_full_cuda.yaml        — 已读 / 未读
□ configs/frozen_contracts.yaml       — 已读 / 未读
□ configs/policy_path_semantics.yaml  — 已读 / 未读
□ configs/runtime_whitelist.yaml      — 已读 / 未读
□ main/cli/run_embed.py               — 已读 / 未读
□ main/cli/run_detect.py              — 已读 / 未读
□ main/watermarking/embed/orchestrator.py   — 已读 / 未读
□ main/watermarking/detect/orchestrator.py  — 已读 / 未读
□ main/watermarking/content_chain/semantic_mask_provider.py — 已读 / 未读
□ main/watermarking/content_chain/content_detector.py       — 已读 / 未读
□ main/watermarking/fusion/decision.py      — 已读 / 未读
□ main/watermarking/geometry_chain/sync/latent_sync_template.py — 已读 / 未读
□ outputs/.../embed_record.json       — 已读 / 未读
□ outputs/.../detect_record.json      — 已读 / 未读
□ outputs/.../calibration_record.json — 已读 / 未读
□ outputs/.../evaluate_record.json    — 已读 / 未读
产物集隔离确认：colab_run_paper_full_cuda / audit_diagnostics 未混用
```

## Repository Anchors

始终以如下文件作为分析锚点：
- `configs/paper_full_cuda.yaml` — 主配置真理源
- `main/watermarking/` — 核心算法实现
- `main/policy/` — 决策策略层
- `main/evaluation/` — 评估指标计算
