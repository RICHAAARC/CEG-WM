# CEG-WM Historical Method Provenance State

本文件记录历史来源的 revision、文件摘要、工作树观察和复用许可边界。它是随审计
变化的 provenance 状态，不是 CEG-WM 方法设计。

## Historical Source Audit

### Read-Only Revisions

| source | read-only revision | state observed | authority |
| --- | --- | --- | --- |
| `SLM-WM-FlowHF` | `a7f33825d0913d4707af5723b236beb65f53f4e5` | tracked worktree clean | historical DirectHF source for the CEG-WM HF candidate only |
| `SLM-WM` | `47bd9a1850c434aa47ee03caa7377706f4d283de` | tracked files clean; `.codex/config.toml` and `docs/ceg_wm_direct_hf_scope_decision.md` untracked | LF/routing/QK candidate source only |
| `SLM-WM` FlowHF baseline | `34825098553d22f68f188afcd938d0aa72132caf` | Git object verified | upstream identity referenced by FlowHF |

本表是 provisional provenance：它证明读过哪个历史 Git object，不证明任何代码已经迁入 CEG-WM，也不替代未来 CEG-WM revision。

两个历史仓库根目录均未发现许可证或 copying 文件。用户可以在此缺口仍存在时授权建立 CEG-WM 版本身份，但在实际复制历史代码前必须由用户确认复用权，或明确授权按本文公式进行不复制源码的独立重写。缺口未关闭时，代码迁移 fail closed。

### Historical DirectHF Source Files

`SLM-WM-FlowHF` revision `a7f33825...`：

| path | SHA-256 |
| --- | --- |
| `flowhf/hf_injector.py` | `03dab6c32d801b712362264584c8b30567e2ab44b88678af2e0c44f27c433cf4` |
| `flowhf/direct_detector.py` | `ea5c5d8ffa34faea4cf7b88d03f78296a3ddd9e44cfbc3e767c366898ea9fd1c` |
| `flowhf/evaluate_keys.py` | `3ce54b65f72f59ac0cde7c132cb58947c05f3af2a1012a1f8b1d78b49a5f372d` |
| `flowhf/key_plan.py` | `c83808d07a6400cfeb3405be5faaeb893d5cb408485a18fb58661ab48f3a9837` |
| `flowhf/model_runtime.py` | `35fcb73c5c78250fc7ea11620f8d1ceb360c13dd298d81a2bbe914c39d7f6de9` |
| `flowhf/run_spec.py` | `cccd166439f0f0be5cfa5281ce8d6eaf9a61005dd8f8452b22516c14a19aee9c` |
| `tests/fixtures/hf_template_golden.json` | `d3f7e9c77ffeecd6f0a5615582bb09b1a2aa170169a71ef4da30ed7ad5483b25` |

FlowHF 只提供 historical DirectHF 的四 Prompt 小样本来源证据。它不提供 CEG-WM HF 成功结论，也不提供 population、fixed-FPR、攻击鲁棒性、LF、路由或 Q/K 成功证据。

### LF, Routing And Q/K Candidate Files

`SLM-WM` revision `47bd9a...`：

| responsibility | path | SHA-256 |
| --- | --- | --- |
| LF template | `main/methods/carrier/low_frequency.py` | `c5d2a4f7cf0879987801372e135e5e537ea2bbe28b3c505300e2759add95bf24` |
| routed composition | `main/methods/carrier/content_update.py` | `f85f2bee8efa5019f1cf34b9e02035b2bf50baec4b81a5cfe87faa22e9f1d170` |
| S/T/R/Q routing | `main/methods/content/routing.py` | `37bf9eac26f85ff667d99dc23678486d9e7ee2962c53547211de18a4d4f3a97a` |
| semantic observation | `main/methods/content/saliency.py` | `07ff1e94fea816333269ca77a3fc89ce54463e92d31e0ce067326b63a82578dc` |
| semantic runtime | `main/methods/content/prompt_saliency_runtime.py` | `47dcd16391a46142dafd8058a414866d672b12b92fb33d2e5093bbe24eeba1b0` |
| texture observation | `main/methods/content/texture.py` | `584d3f6ce24d6a86bacc2f5a46f7a3d69cc2362133c79aa0c4ade5df6b8e2122` |
| response observation | `main/methods/content/latent_response.py` | `947af3114806c50984123b6f6b475ad9de753ea007b7675d4067619b9711f736` |
| sensitivity observation | `main/methods/content/local_sensitivity.py` | `de0eee215e1fe77ba7559c99a7fed7747d09d22da40328690516e7ddf4316331` |
| reference P95 rule | `experiments/protocol/content_routing_reference_quantile.py` | `a9f1d407b08e3ba59a7354a3b804048e5ab823350230f572072400295ae538fd` |
| historical runtime config | `configs/model_sd35.yaml` | `dabebea3fa5c9c06fdc880f093debec6913bf5ce4da31f00be51578bfe2e1670` |
| direct Q/K relation | `main/methods/geometry/differentiable_attention.py` | `6c48f69e005b2c3f450de1ec2531910b9f076d25a60e03bee1ac2db61ee138b3` |
| Q/K synchronization write | `main/methods/geometry/sync_update.py` | `1590ac04e9bcdbc265e62383469808a06cefbd68457903e86a63afbc557863cc` |
| affine estimation and rectification | `main/methods/geometry/attention_alignment.py` | `134fd1e32b4542c7904540093a1279b85a36908c44dc2f37f36c5ac9bae2c8c2` |
| Q/K runtime protocol | `experiments/protocol/method_runtime_config.py` | `8619aa4e4ec3e87d1b80558878ff1e91e6f6c501c2c70534dd59b59df16a2da9` |
| image-only Q/K extraction source | `experiments/runners/semantic_watermark_runtime.py` | `87ec13fc86b843289505cb855f232fd6a6cea494265c2ab16370ba1295866424` |
| keyed PRG | `main/core/keyed_prg.py` | `9fd5f24023862afef4743dc6aca1cf0b4401f1ffb8d848c4d52f86616945cea2` |
| normal quantile table | `main/core/normal_quantile_table.py` | `e98c2a0d76080d5080b8d22eb20cb7559c8291a668cf810aa508d89bc7b8776e` |

FlowHF 明确把旧 LF/HF 双载体和 Q/K coupled route 记为 historical non-passing route。因此这些文件只提供具体候选语义，不能向 CEG-WM 传递成功、阈值或论文证据。
