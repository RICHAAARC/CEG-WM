# Project Tests

此目录只测试研究项目本体，不依赖任何外层控制平面。默认 pytest 运行标记为 `unit`、`constraint` 或 `quick` 的研究测试；`integration`、`smoke`、`slow` 与 `formal` 必须显式选择。

| path | purpose | default_run |
| --- | --- | --- |
| `unit/` | 纯函数、算法局部行为和 schema。 | yes |
| `functional/` | 小型合成输入上的轻量功能。 | yes |
| `integration/` | 跨层或真实组件集成。 | no |
| `smoke/` | 真实 backend 的关键可用性。 | no |
| `formal/` | 冻结协议下的证据与发布门禁。 | no |
| `fixtures/` | 小型测试数据。 | not applicable |
| `helpers/` | 非测试辅助模块。 | not applicable |

`constraint` 是按测试目的使用的 marker，不要求单独建立目录；研究协议、schema 或文件边界的轻量约束测试仍按所属功能放入 `unit/` 或 `functional/`。治理控制平面约束测试只放在 `governance/tests/`。

当前研究测试入口见 `docs/reference/test_inventory.md`。
