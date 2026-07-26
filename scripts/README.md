# Command-Line Support Scripts

此目录保存数据准备、实验执行和重建辅助命令，不保存核心方法或只能从脚本进入的实验实现。它属于研究项目可交付外围代码，不得依赖任何外层控制平面。

当前 CEG-WM 尚未提供研究执行脚本。后续按真实用途放置：

- `experiment_execution/`：服务器或 Colab 执行所需研究脚本；
- `artifact_rebuild/`：只从冻结 records 重建产物的脚本。

拆包工具属于可拆卸外层，位于 `governance/tools/`，不在本目录保存。
