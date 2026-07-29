# Notebook Entrypoints

此目录保存 Jupyter 与 Colab 的薄编排入口。Notebook 只能进行环境准备、配置选择、repository module 调用、轻量检查和展示；协议、方法、攻击、指标、records 和 artifact rebuild 的唯一实现必须留在可测试模块中。

## 目录约定

- `colab/`：Colab 入口。
- `support/`：只有出现真实重复的 Notebook 环境或展示代码后才创建；不得放科学计算核心。

提交的 Notebook 必须清空 cell outputs 和 execution count。运行输出写入未提交目录，正式 records 与 artifacts 通过 repository modules 生成。

当前唯一入口是 `colab/runtime_qualification.ipynb`。它只收集运行时参数、挂载
Drive、检查资源与 Secret、单次读取并核对独立 bootstrap 摘要、建立新的
`/content` 可信快照并只调用该快照；execution
package 的安全解包、身份校验、依赖安装、runner 调用和结果归档均不在 Notebook
源中实现。
