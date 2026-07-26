# Infrastructure Entrypoints

此目录保存环境定义、作业调度、远程执行和 GPU 服务器入口。它可以调用 runtime 与 experiment runners，但不得保存唯一的方法、协议、攻击、指标或证据构建实现。

环境文件应固定可复现依赖；调度配置不得包含凭据、私有路径或运行结果。当前 CEG-WM 只定义边界，不提供基础设施。
