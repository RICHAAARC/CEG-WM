# 文件组织契约

## 研究项目本体

```text
main/                   CEG-WM 最小方法包边界
  shared/               两链共享类型与密钥语义
  content_chain/        router、LF/HF carrier、embedder、LF/HF detector 与 content detector
  geometry_chain/       Q/K 同步、几何估计、独立 reliability 与回正
  joint_decision/       近阈值门控与同阈值重判
runtime/                只依赖 main 的运行能力
experiments/            协议、适配、攻击、指标与 runners
paper_artifacts/        从冻结 records 重建论文产物
configs/                研究与运行配置
notebooks/              Notebook / Colab 薄编排入口
infrastructure/         环境、调度和远程执行入口
scripts/                研究执行与 artifact rebuild 辅助命令
templates/              研究协议或交付模板
docs/                   设计、指引、决策和研究参考
tests/                  仅测试研究项目本体
third_party/            可选固定来源外部源码
```

## 可拆卸外层

```text
.agents/skills/         Codex 项目工作流
.codex/                 当前项目合同与阶段审计元数据
governance/             契约、policy、harness、自测和治理说明
```

外层还包含 `governance/tools/` 下的拆包工具。外层可以读取研究本体；研究代码、可交付脚本和 Notebook 可执行代码不得导入 `governance`。删除全部外层目录后，核心导入、项目测试、实验协议和 artifact rebuild 仍应可用。

精确根目录与依赖边界分别以 `governance/policies/project_roots.yaml` 和 `governance/policies/dependency_rules.yaml` 为准。
