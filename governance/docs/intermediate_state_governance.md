# Intermediate State Governance

写入研究配置、records、manifests、tables、reports，或跨模块、进程、Notebook 边界传递的字段必须登记；纯函数局部变量通常不登记。

| category | suffix | rule |
| --- | --- | --- |
| intermediate | `_intermediate` | 跨步骤保存、尚未成为稳定协议字段。 |
| temporary | `_temporary` | 可清理的临时 artifact 标记。 |
| cache | `_cache` | 可由输入、配置和代码重建。 |

Temporary 和 cache 不得支持 claim；成为稳定协议后应改用实际语义字段。Placeholder 与 random trace 规则见 `governance/docs/placeholder_random_governance.md`。
