# CEG-WM: Content-Evidence Geometry Watermarking

融合内容证据与几何证据的鲁棒水印系统。支持扩散模型（Stable Diffusion 3）中的水印嵌入与检测。

## 核心模块

### 水印系统
- `main/watermarking/embed/`: 嵌入层（pattern、injector、plan、router、subspace）
- `main/watermarking/detect/`: 检测层（decoder、evidence、inversion、trace）
- `main/watermarking/fusion/`: 融合层（decision、policy、calibrate、audit）
- `main/watermarking/content_chain/`: 内容证据链路（interfaces、content_baseline_extractor、statistics）
- `main/watermarking/geometry_chain/`: 几何证据链路

### 扩散模型集成
- `main/diffusion/sd3/`: SD3 推理、hook、权重快照、溯源与加载
- `main/diffusion/schedulers/`: 调度器与 DDIM 反演

### 核心基础设施
- `main/core/`: 配置加载、契约、摘要、记录、路径指纹、状态
- `main/registries/`: 运行时注册表与能力解析
- `main/policy/`: 冻结门禁、路径策略、白名单与覆盖规则

### 评估与协议
- `main/evaluation/`: 攻击、指标、协议、报告与表格导出

## 快速开始

### 安装依赖
项目依赖由内部环境统一管理；如需单独安装，请根据实际环境补齐依赖。

### 配置
编辑 `configs/default.yaml` 配置模型、水印强度、生成参数等。

### 水印嵌入
```bash
python main/cli/run_embed.py --config configs/default.yaml
```

### 水印检测
```bash
python main/cli/run_detect.py --config configs/default.yaml
```

### 评估与校准
```bash
python main/cli/run_evaluate.py --config configs/default.yaml
python main/cli/run_calibrate.py --config configs/default.yaml
```

## 审计与质量门禁

### Policy Path 审计
验证融合决策的规范性与冻结白名单合规性：
```bash
python scripts/audits/audit_policy_path_semantics_binding.py --run_dir outputs_demo --strict_freeze 1 --strict_source_audit 1
```

输出报告：
- `policy_path_audit.json`: 完整审计结果与时间戳
- `policy_path_counts.csv`: 策略路径分布统计
- `policy_path_violations.csv`: 违规详情
- `policy_path_source_violations.csv`: 源码静态审计违规

## 项目结构

```
CEG-WM/
├── configs/              # 配置文件（default.yaml、frozen_contracts.yaml、policy_path_semantics.yaml）
├── doc/                  # 文档与审计说明
├── main/                 # 核心实现
│   ├── cli/              # 命令行入口
│   ├── watermarking/     # 水印系统（embed、detect、fusion、content_chain、geometry_chain）
│   ├── diffusion/        # 扩散模型集成（SD3、DDIM 反演）
│   ├── evaluation/       # 指标与评估
│   ├── registries/       # 注册表与能力解析
│   ├── policy/           # 策略与冻结门禁
│   └── core/             # 核心基础设施
├── scripts/              # 审计脚本与运行入口
│   └── audits/           # 审计集合
├── tests/                # 测试用例
└── pyproject.toml        # 测试与工具配置
```

## 文件级说明（main 与 configs）

### configs
- [configs/default.yaml](configs/default.yaml): 运行默认配置，包含 policy_path、target_fpr、链路开关与占位 impl_id。
- [configs/frozen_contracts.yaml](configs/frozen_contracts.yaml): 冻结契约权威来源，定义 gate 规则、digest 口径与 records schema。
- [configs/model_sd3.yaml](configs/model_sd3.yaml): SD3 模型溯源字段模板，记录模型来源与权重校验信息。
- [configs/policy_path_semantics.yaml](configs/policy_path_semantics.yaml): policy_path 语义映射、链路需求、审计义务与枚举约束。
- [configs/runtime_whitelist.yaml](configs/runtime_whitelist.yaml): 运行期白名单、override 允许范围、枚举闭包与 impl_id 允许集合。

### main/cli
- [main/cli/__init__.py](main/cli/__init__.py): CLI 入口执行方式校验，强制模块方式启动。
- [main/cli/run_calibrate.py](main/cli/run_calibrate.py): 校准 CLI 占位流程，加载事实源与配置，写入校准记录与闭包。
- [main/cli/run_common.py](main/cli/run_common.py): CLI 共享工具，包含字段路径写入、impl 绑定与失败状态整理。
- [main/cli/run_detect.py](main/cli/run_detect.py): 检测 CLI 占位流程，构建 pipeline，执行推理烟测并写入记录。
- [main/cli/run_embed.py](main/cli/run_embed.py): 嵌入 CLI 占位流程，构建 pipeline，执行推理烟测并写入记录。
- [main/cli/run_evaluate.py](main/cli/run_evaluate.py): 评估 CLI 占位流程，写入评估记录与闭包。

### main/core
- [main/core/__init__.py](main/core/__init__.py): 包初始化占位。
- [main/core/config_loader.py](main/core/config_loader.py): YAML 统一加载入口，计算 provenance 与 cfg_digest，并应用 override。
- [main/core/contracts.py](main/core/contracts.py): 冻结契约加载与解释面构建，绑定契约字段到记录。
- [main/core/digests.py](main/core/digests.py): 规范化 JSON 与 SHA256 digest 统一入口。
- [main/core/env_fingerprint.py](main/core/env_fingerprint.py): 环境指纹采集与摘要计算。
- [main/core/errors.py](main/core/errors.py): 冻结契约与门禁相关异常定义与失败原因枚举。
- [main/core/input_provenance.py](main/core/input_provenance.py): 外部输入来源溯源记录与摘要。
- [main/core/records_bundle.py](main/core/records_bundle.py): records bundle 闭包与 manifest 生成。
- [main/core/records_io.py](main/core/records_io.py): 统一写盘通道与 gate 触发，提供原子写入与事实源上下文。
- [main/core/schema.py](main/core/schema.py): records 与 run_closure schema、字段访问器与校验入口。
- [main/core/status.py](main/core/status.py): run_closure 生成、环境审计与状态枚举校验。
- [main/core/time_utils.py](main/core/time_utils.py): 时间戳统一入口与 RNG 种子占位机制。

### main/diffusion
- [main/diffusion/__init__.py](main/diffusion/__init__.py): 包初始化占位。
- [main/diffusion/schedulers/ddim_inversion.py](main/diffusion/schedulers/ddim_inversion.py): 空文件，占位 DDIM 反演实现。
- [main/diffusion/sd3/__init__.py](main/diffusion/sd3/__init__.py): 包初始化占位。
- [main/diffusion/sd3/attention_registry.py](main/diffusion/sd3/attention_registry.py): 空文件，占位注意力注册表。
- [main/diffusion/sd3/diffusers_loader.py](main/diffusion/sd3/diffusers_loader.py): diffusers 受控导入与 SD3 pipeline 构造封装。
- [main/diffusion/sd3/hooks.py](main/diffusion/sd3/hooks.py): 空文件，占位 hook 扩展点。
- [main/diffusion/sd3/infer_runtime.py](main/diffusion/sd3/infer_runtime.py): SD3 推理烟测与运行期元信息收集。
- [main/diffusion/sd3/infer_trace.py](main/diffusion/sd3/infer_trace.py): 推理轨迹构造与摘要计算。
- [main/diffusion/sd3/latent_io.py](main/diffusion/sd3/latent_io.py): 空文件，占位 latent IO。
- [main/diffusion/sd3/pipeline_factory.py](main/diffusion/sd3/pipeline_factory.py): pipeline 壳构造、溯源与权重快照绑定。
- [main/diffusion/sd3/provenance.py](main/diffusion/sd3/provenance.py): pipeline 溯源对象构造与 digest。
- [main/diffusion/sd3/weights_snapshot.py](main/diffusion/sd3/weights_snapshot.py): 权重快照离线摘要与元信息构建。

### main/evaluation
- [main/evaluation/__init__.py](main/evaluation/__init__.py): 包初始化占位。
- [main/evaluation/attacks.py](main/evaluation/attacks.py): 空文件，占位攻击模拟集合。
- [main/evaluation/metrics.py](main/evaluation/metrics.py): 空文件，占位指标计算。
- [main/evaluation/protocol.py](main/evaluation/protocol.py): 空文件，占位评估协议。
- [main/evaluation/reporting.py](main/evaluation/reporting.py): 空文件，占位报告生成。
- [main/evaluation/table_export.py](main/evaluation/table_export.py): 空文件，占位表格导出。

### main/policy
- [main/policy/__init__.py](main/policy/__init__.py): 包初始化占位。
- [main/policy/freeze_gate.py](main/policy/freeze_gate.py): 写盘前 gate 校验与契约化闭包策略执行。
- [main/policy/override_rules.py](main/policy/override_rules.py): CLI override 解析、白名单校验与审计段生成。
- [main/policy/path_policy.py](main/policy/path_policy.py): 输出路径布局、越界与 symlink 校验。
- [main/policy/runtime_whitelist.py](main/policy/runtime_whitelist.py): 白名单与语义表加载、digest 计算与一致性校验。

### main/registries
- [main/registries/__init__.py](main/registries/__init__.py): 包初始化占位。
- [main/registries/capabilities.py](main/registries/capabilities.py): impl 能力声明与兼容性 gate。
- [main/registries/content_registry.py](main/registries/content_registry.py): 内容链与子空间规划 registry，占位实现注册。
- [main/registries/fusion_registry.py](main/registries/fusion_registry.py): 融合规则 registry，占位实现注册。
- [main/registries/geometry_registry.py](main/registries/geometry_registry.py): 几何链与同步模块 registry，占位实现注册。
- [main/registries/impl_identity.py](main/registries/impl_identity.py): impl 身份与元数据结构与 digest 计算。
- [main/registries/pipeline_registry.py](main/registries/pipeline_registry.py): pipeline 壳 registry，SD3 占位实现注册。
- [main/registries/registry_base.py](main/registries/registry_base.py): registry 基座，支持 seal 与解析。
- [main/registries/runtime_resolver.py](main/registries/runtime_resolver.py): 从 cfg 解析 impl 并构造实现集合，聚合能力校验。

### main/watermarking
- [main/watermarking/__init__.py](main/watermarking/__init__.py): 包初始化占位。

#### main/watermarking/content_chain
- [main/watermarking/content_chain/__init__.py](main/watermarking/content_chain/__init__.py): 包初始化占位。
- [main/watermarking/content_chain/interfaces.py](main/watermarking/content_chain/interfaces.py): 内容证据结构与提取器协议定义。
- [main/watermarking/content_chain/content_baseline_extractor.py](main/watermarking/content_chain/content_baseline_extractor.py): 空文件，占位内容链实现。
- [main/watermarking/content_chain/statistics.py](main/watermarking/content_chain/statistics.py): 空文件，占位统计模块。
- [main/watermarking/content_chain/subspace/planner_interface.py](main/watermarking/content_chain/subspace/planner_interface.py): 空文件，占位子空间规划接口。
- [main/watermarking/content_chain/subspace/subspace_planner_impl.py](main/watermarking/content_chain/subspace/subspace_planner_impl.py): 空文件，占位子空间规划实现。

#### main/watermarking/detect
- [main/watermarking/detect/__init__.py](main/watermarking/detect/__init__.py): 包初始化占位。
- [main/watermarking/detect/orchestrator.py](main/watermarking/detect/orchestrator.py): detect/evaluate/calibrate 占位编排与记录字段构造。
- [main/watermarking/detect/baseline_stub.py](main/watermarking/detect/baseline_stub.py): 空文件，占位检测侧实现。

#### main/watermarking/embed
- [main/watermarking/embed/__init__.py](main/watermarking/embed/__init__.py): 包初始化占位。
- [main/watermarking/embed/orchestrator.py](main/watermarking/embed/orchestrator.py): 嵌入占位编排与记录字段构造。
- [main/watermarking/embed/baseline_stub.py](main/watermarking/embed/baseline_stub.py): 空文件，占位嵌入侧实现。

#### main/watermarking/fusion
- [main/watermarking/fusion/__init__.py](main/watermarking/fusion/__init__.py): 包初始化占位。
- [main/watermarking/fusion/decision.py](main/watermarking/fusion/decision.py): 空文件，占位融合决策逻辑。
- [main/watermarking/fusion/interfaces.py](main/watermarking/fusion/interfaces.py): 融合决策结构与规则协议定义。
- [main/watermarking/fusion/neyman_pearson.py](main/watermarking/fusion/neyman_pearson.py): NP 阈值占位实现与 digest 口径。
- [main/watermarking/fusion/baseline_fusion_stub.py](main/watermarking/fusion/baseline_fusion_stub.py): 空文件，占位融合实现。

#### main/watermarking/geometry_chain
- [main/watermarking/geometry_chain/__init__.py](main/watermarking/geometry_chain/__init__.py): 包初始化占位。
- [main/watermarking/geometry_chain/canonicalize.py](main/watermarking/geometry_chain/canonicalize.py): 空文件，占位几何规范化。
- [main/watermarking/geometry_chain/interfaces.py](main/watermarking/geometry_chain/interfaces.py): 几何证据结构与提取器协议定义。
- [main/watermarking/geometry_chain/baseline_impl.py](main/watermarking/geometry_chain/baseline_impl.py): 空文件，占位几何链实现。
- [main/watermarking/geometry_chain/sync/__init__.py](main/watermarking/geometry_chain/sync/__init__.py): 包初始化占位。
- [main/watermarking/geometry_chain/sync/baseline_sync.py](main/watermarking/geometry_chain/sync/baseline_sync.py): 空文件，占位同步模块实现。
- [main/watermarking/geometry_chain/sync/sync_interface.py](main/watermarking/geometry_chain/sync/sync_interface.py): 空文件，占位同步协议定义。

## 关键特性

✓ **双证据融合**: 内容证据 + 几何证据  
✓ **策略路由**: 15+ 个预定义 Policy Path  
✓ **冻结机制**: 严格白名单约束（仅 `content_evidence_only`）  
✓ **占位实现**: 基线占位标记（embed_mode、detect_feature）  
✓ **源码静态审计**: Policy Path 配置暴露与透传检测  
✓ **工程门禁**: PASS/FAIL 审计结果与时间戳  

## 通用配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_id` | Stable Diffusion 模型 | `stabilityai/stable-diffusion-3.5-medium` |
| `embed_mode` | 嵌入模式 | `post_vae_encode_baseline` |
| `pattern_id` | 水印模式 | `gaussian_iid` |
| `strength` | 水印强度 | `0.1` |
| `height` / `width` | 生成分辨率 | `1024` |
| `strict_freeze_policy` | 评估冻结模式 | `true` |

## 许可证

内部项目

